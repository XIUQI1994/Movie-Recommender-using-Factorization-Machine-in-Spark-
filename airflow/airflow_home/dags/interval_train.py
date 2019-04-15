import sys
import json
from pyspark import SparkContext, SparkConf
from pyspark.mllib.util import MLUtils
from uuid import uuid1
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from time import sleep
from pymongo import MongoClient
import time
import os
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import numpy as np

mongo_client = MongoClient('localhost', 27017)
db = mongo_client.movies
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
total_feas = int(db.total_feas.find_one(
            {'key': 'total_feas'})['value'])
sse = SparkSession .builder .appName("myApp") .config(
    "spark.mongodb.input.uri",
    "mongodb://127.0.0.1/movies.live") .config(
    "spark.mongodb.output.uri",
    "mongodb://127.0.0.1/movies.ratings") .config(
    "spark.jars.packages",
    "org.mongodb.spark:mongo-spark-connector_2.11:2.4.0").getOrCreate()
sc = sse.sparkContext
sc.addPyFile(
    "/home/xiuqi/Dropbox/MLE/myProject/airflow/airflow_home/dags/fm/fm_parallel_extend.py")
from fm_parallel_extend import trainFM_parallel_sgd, predictFM, evaluation

class train:
    def load_weights(self):
        weights_r = db.weights.find_one()
        b = np.array(weights_r['b'])
        w = np.array(weights_r['w']).reshape(-1, 5)
        return (w, b)

    def train(self, weights):

        def change_labelPoint(rdd, total_feas):
            label_point = LabeledPoint(
                rdd['label'], SparseVector(
                    total_feas, rdd['pos'], rdd['val']))
            return label_point

        df = sse.read.format("com.mongodb.spark.sql.DefaultSource").option(
            'database', 'movies'). option('collection', 'live').load()
        db.live.drop()
        data = df.rdd.map(lambda x: change_labelPoint(x, total_feas))

        if data.isEmpty():
        	return weights


        evalTraining = evaluation(data, 'reg', 'mse')
        evalTraining.modulo = 1
        weights = trainFM_parallel_sgd(
            sc,
            data,
            weights=weights,
            iterations=1,
            iter_sgd=1,
            factorLength=5,
            verbose=True,
            evalTraining=evalTraining,
            mode='reg',
            loss='mse')

        return weights

    def update_weights(self, weights):
        record = {}
        record['w'] = weights[0].reshape(-1).tolist()
        record['b'] = weights[1].reshape(-1).tolist()
        db.weights.drop()
        db.weights.insert_one(record)
        return weights

    def update_recom(self, weights):

        def labelPoint(row, user_pos, total_feas):
            pos = row['pos']
            val = row['val']
            pos = list(map(lambda x: int(x), pos.strip().split(' ')))
            val = list(map(lambda x: int(x), val.strip().split(' ')))
            pos.append(user_pos)
            val.append(1)
            label_point = LabeledPoint(0, SparseVector(total_feas, pos, val))
            return label_point


        pipeline = "{ '$match': {'imdb_score': {'$gt': 5}}}"
        df = sse.read.format("com.mongodb.spark.sql.DefaultSource").option(
            'database', 'movies'). option(
            'collection', 'movies').option(
            "pipeline", pipeline).load()

        users = db.to_update.find()
        updated = set()

        for user in users:
            user_pos = user['pos']
            userId = user['userId']
            if userId in updated:
                pass
            else:
                updated.add(userId)
                all_movies = df.rdd.map(
                    lambda x: labelPoint(
                        x, user_pos, total_feas))
                preds = predictFM(all_movies, weights[0], weights[1]).collect()

                seen_r = db.seen.find_one({'userId': userId})

                if not seen_r:
                    seen = set()
                else:
                    seen = set(seen_r['seen'])

                top = np.argsort(preds)[-350:][::-1]
                ids = df.select('movieId').collect()
                ids_array = np.array([int(row.movieId) for row in ids])[top]
                ids_array = list(set(ids_array) - seen)

                top_ids = np.random.choice(
                    ids_array, 40, replace=False).tolist()
                top_ids = list(map(lambda x: int(x), top_ids))

                new_rec = {'userId': userId, 'rec': top_ids}

                if db.recommendation.find_one({'userId': userId}):
                    db.recommendation.update_one(
                        {'userId': userId},
                        {'$set': {'rec': top_ids}})
                else:
                    db.recommendation.insert_one(new_rec)
        sc.stop()
        return

