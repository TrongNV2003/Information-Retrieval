import json
from datasets import Dataset
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

        negative_texts = []
        for neg_article in item['hard_negatives']:
            if include_title:
                negative_text = f"{neg_article['title']}{sep_token}{neg_article['text']}"
            else:
                negative_text = neg_article['text']
            
            negative_texts.append(negative_text)
        
        
        processed_data.append({
            "query": anchor_text,
            "positive": positive_text,
            # "negative": negative_texts,
        })

    return Dataset.from_list(processed_data)


def load_val_dataset(
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

        negative_texts = []
        for neg_article in item['hard_negatives']:
            if include_title:
                negative_text = f"{neg_article['title']}{sep_token}{neg_article['text']}"
            else:
                negative_text = neg_article['text']
            
            negative_texts.append(negative_text)

        processed_data.append({
            "query": anchor_text,
            "positive": [positive_text],
            "negative": negative_texts,
        })

    return Dataset.from_list(processed_data)
