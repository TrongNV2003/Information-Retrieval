import json
from datasets import Dataset
from typing import Dict, Set, Tuple
from transformers import AutoTokenizer


def load_train_dataset(
    json_file: str,
    tokenizer: AutoTokenizer,
    include_title: bool = False
) -> Dataset:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    sep_token = tokenizer.sep_token
    for item in data:
        anchor_text = item["question"]
        positive_article = item["relevant_articles"][0]
        
        if include_title:
            positive_text = f"{positive_article['title']}{sep_token}{positive_article['text']}"
        else:
            positive_text = positive_article["text"]

        # negative_texts = []
        for neg_article in item['hard_negatives']:
            if include_title:
                negative_text = f"{neg_article['title']}{sep_token}{neg_article['text']}"
            else:
                negative_text = neg_article['text']
            # negative_texts.append(negative_text)
        
        processed_data.append({
            "anchor": anchor_text,
            "positive": positive_text,
            "negative": negative_text,
        })

    return Dataset.from_list(processed_data)


class EvalDataLoader:
    def __init__(
        self,
        json_file: str,
        corpus_file: str,
        tokenizer: AutoTokenizer,
        include_title: bool = False
    ):
        self.json_file = json_file
        self.corpus_file = corpus_file
        self.tokenizer = tokenizer
        self.include_title = include_title
        self.queries, self.corpus, self.relevant_docs = self._load_data()

    def _load_data(self) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
        with open(self.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)

        queries: Dict[str, str] = {}
        corpus: Dict[str, str] = {}
        relevant_docs: Dict[str, Set[str]] = {}

        for idx, item in enumerate(data):
            query_id = f"query_{idx}"
            queries[query_id] = item["question"]
            relevant_docs[query_id] = {f"{article['law_id']}_{article['article_id']}" for article in item["relevant_articles"]}

        for item in corpus_data:
            doc_id = f"{item['law_id']}_{item['article_id']}"
            corpus[doc_id] = f"{item['title']} {self.tokenizer.sep_token} {item['text']}" if self.include_title else item["text"]
            
        return queries, corpus, relevant_docs

    def get_data(self):
        return self.queries, self.corpus, self.relevant_docs
    
