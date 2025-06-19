import json
from typing import List, Dict, Mapping

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CrossEncoderDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        include_title: bool = False
    ) -> None:
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.max_length = max_length
        self.include_title = include_title
        
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        query = item["question"]
        positive_article = item["relevant_articles"][0]
        if self.include_title:
            positive_text = f"{positive_article['title']} {self.sep_token} {positive_article['text']}"
        else:
            positive_text = positive_article["text"]

        negative_texts = []
        for neg_article in item['hard_negatives']:
            if self.include_title:
                neg_text = f"{neg_article['title']} {self.sep_token} {neg_article['text']}"
            else:
                neg_text = neg_article['text']
            negative_texts.append(neg_text)

        return query, positive_text, negative_texts


class CrossEncoderCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        queries, positives, negatives = zip(*batch)
        pos_pairs = []
        neg_pairs = []
        for query, positive, negative_list in zip(queries, positives, negatives):
            pos_pairs.append((query, positive))
            neg_pairs.extend([(query, neg) for neg in negative_list])
            
        positive_inputs = self.tokenizer(
            pos_pairs,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        negative_inputs = self.tokenizer(
            neg_pairs,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        return positive_inputs, negative_inputs


# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
#     dataset = CrossEncoderDataset("dataset/ZaloTextRetrieval/dataset_negatives/test_negatives.json", tokenizer)

#     for i in range(1):
#         query, positive_text, negative_texts = dataset[i]
#         print(f"Query: {query}")
#         print(f"Positive Text: {positive_text}")
#         print(f"Negative Texts: {negative_texts}")
