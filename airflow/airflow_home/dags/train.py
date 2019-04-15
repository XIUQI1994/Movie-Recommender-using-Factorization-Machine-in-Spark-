from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from interval_train import train
import os
from datetime import datetime


def train_task(**kwargs):
    a = train()
    a.do()


workflow = train()


def load_w(**kwargs):
    return workflow.load_weights()


def retrain_model(**kwargs):
    ti = kwargs['task_instance']
    weights = ti.xcom_pull(task_ids='load_w')
    return workflow.train(weights)


def update_w(**kwargs):
    ti = kwargs['task_instance']
    weights = ti.xcom_pull(task_ids='retrain_model')
    return workflow.update_weights(weights)


def update_rec(**kwargs):
    ti = kwargs['task_instance']
    weights = ti.xcom_pull(task_ids='update_w')
    return workflow.update_recom(weights)


dag = DAG('train', description='oneDAG',
          schedule_interval='0 12 * * *',
          start_date=datetime(2019, 4, 13), catchup=False)

# dummy_operator = DummyOperator(task_id='dummy_task', retries=3, dag=dag)

# hello_operator = PythonOperator(task_id='train_task', python_callable=train_task, dag=dag)
load_w_operator = PythonOperator(
    task_id='load_w',
    python_callable=load_w,
    dag=dag,
    provide_context=True)
retrain_model_operator = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
    provide_context=True)
update_w_operator = PythonOperator(
    task_id='update_w',
    python_callable=update_w,
    dag=dag,
    provide_context=True)
update_rec_operator = PythonOperator(
    task_id='update_rec',
    python_callable=update_rec,
    dag=dag,
    provide_context=True)


load_w_operator >> retrain_model_operator >> update_w_operator >> update_rec_operator
