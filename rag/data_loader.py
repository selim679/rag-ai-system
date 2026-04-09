import json

def load_data():
    with open("data_pipeline/arxiv_data.json", "r", encoding="utf-8") as f:
        return json.load(f)
