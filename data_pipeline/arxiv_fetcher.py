import requests
import xml.etree.ElementTree as ET
import json
import os

BASE_URL = "http://export.arxiv.org/api/query"

def fetch_arxiv(query="transformer NLP attention BERT GPT", max_results=100):


    url = f"{BASE_URL}?search_query=all:{query}&start=0&max_results={max_results}"

    response = requests.get(url)
    root = ET.fromstring(response.content)

    papers = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):

        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        published = entry.find("{http://www.w3.org/2005/Atom}published").text

        paper = {
            "title": title.strip(),
            "summary": summary.strip(),
            "published": published
        }

        papers.append(paper)

    return papers


def save_to_json(papers, filename="data_pipeline/arxiv_data.json"):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=4)

    print(f"Saved {len(papers)} papers to {filename}")


if __name__ == "__main__":
    papers = fetch_arxiv(
        "transformer NLP attention BERT GPT machine translation",
        100
    )

    # FILTER NLP-RELEVANT PAPERS
    papers = [
        p for p in papers
        if is_nlp_related(p["title"] + p["summary"])
    ]

    save_to_json(papers)


def is_nlp_related(text):
    keywords = [
        "transformer", "attention", "nlp",
        "bert", "gpt", "language"
    ]
    return any(k in text.lower() for k in keywords)
