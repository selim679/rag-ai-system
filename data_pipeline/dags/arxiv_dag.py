from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.append("/opt/airflow")

from data_pipeline.arxiv_fetcher import fetch_arxiv, save_to_json

def extract():
    papers = fetch_arxiv("machine learning", 10)
    save_to_json(papers)

def transform():
    print("Cleaning data... (to be improved in DAY 3)")

def load():
    print("Loading into vector DB... (DAY 3)")

with DAG(
    dag_id="arxiv_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="transform", python_callable=transform)
    t3 = PythonOperator(task_id="load", python_callable=load)

    t1 >> t2 >> t3
