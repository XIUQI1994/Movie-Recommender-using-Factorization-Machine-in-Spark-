
from flask import Flask,jsonify,json
from flask import render_template, request
from flask_pymongo import PyMongo
import numpy as np

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'movies'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/movies'

mongo = PyMongo(app)


@app.route('/')
@app.route('/index')
def index():
    movie_list = mongo.db.AllMovieTitle.find()[0]['title']
    return render_template("index.html", movie_list = movie_list)


@app.route('/show')
def showData():
    user = request.args.get("user")
    user_r = mongo.db.users.find_one({'name': user})
    if user_r and mongo.db.recommendation.find_one({'userId': user_r['userId']}):
        userId = user_r['userId']
        mongo.db.recommendation.find_one({'name': user})
        recommendations = mongo.db.recommendation.find_one({'userId': userId})['rec']
        to_show = np.random.choice(recommendations, 10, replace=False)

        output = []
        for movieId in to_show:
            movie = mongo.db.movies.find_one({'movieId': int(movieId)})
            output.append({'title': movie['title'], 'genres': movie['genres'], 'director_name': movie['director_name'],
                    'actor1_name': movie['actor_1_name'], 'actor2_name': movie['actor_2_name'], 'country': movie['country'], 'imdb_score': movie['imdb_score']})
    else:
        to_show = mongo.db.movies.aggregate([{ '$match': {'imdb_score': {'$gt': 8}}}, 
          {'$sample': {'size': 10}} 
        ])

        output = []
        for movie in to_show:
            output.append({'title': movie['title'], 'genres': movie['genres'], 'director_name': movie['director_name'],
                    'actor1_name': movie['actor_1_name'], 'actor2_name': movie['actor_2_name'], 'country': movie['country'], 'imdb_score': movie['imdb_score']})
 
    return render_template(
    "showData.html",
    user=user,
     json_data=output )

@app.route('/history')
def showHistory():
    user = request.args.get("user")
    user_r = mongo.db.users.find_one({'name': user})
    res = []
    if user_r: 
        userId = user_r['userId']
        history = set(mongo.db.seen.find({'userId': userId})['seen'])
        for h in history:
            res.append({'title': mongo.db.movies.find_one({'movieId': h['movieId']})['title'], 'rating': h['rating']})
    return render_template(
    "showHistory.html",
    user=user,
     json_data=res )    

@app.route('/new_rating')
def addRating():
    user = request.args.get("user")
    movie = request.args.get("movie")
    rating = request.args.get("rating")
    rating = float(rating)
    user_r = mongo.db.users.find_one({'name': user})
    if not user_r:
        max_fea = mongo.db.total_feas.find_one({'key': 'total_feas'})['value']
        mongo.db.users.insert_one({'name':user, 'userId': max_fea, 'pos':max_fea, 'val':1})        
        mongo.db.total_feas.update(
                {'key' : 'total_feas'},
                {'$set': { 'value':max_fea+1}})
        
        userId, userpos, userval = max_fea, max_fea, 1
    else:
        userId, userpos, userval = user_r['userId'], user_r['pos'], user_r['val']
    
    movie_r = mongo.db.movies.find_one({'title': movie})
    movieId, moviepos, movieval = movie_r['movieId'], movie_r['pos'], movie_r['val']
    # moviepos = list(map(lambda x: int(x), moviepos.split(' ')))
    # movieval = list(map(lambda x: int(x), movieval.split(' ')))
    mongo.db.ratings.insert({'userId': userId, 'movieId':movieId, 'rating':rating})
    pos = moviepos + [userpos]
    val = movieval + [userval]
    mongo.db.live.insert({'label':rating, 'pos': pos, 'val': val})
    mongo.db.to_update.insert({'userId': userId, 'pos':userpos})

    seen_r = mongo.db.seen.find_one({'userId':userId})
    if seen_r: 
        seen = set(seen_r['seen'])
    else:
        seen = set()
    
    seen.add(movieId)
    # print(seen)
    mongo.db.seen.update(
                {'userId' : userId},
                {'$set': { 'seen': list(seen)}})    
    movie_list = mongo.db.AllMovieTitle.find()[0]['title']
    return render_template('index.html',movie_list = movie_list) 


if __name__ == '__main__':
    app.run(debug=True)
