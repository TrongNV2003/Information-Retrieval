import json
from typing import List, Dict, Mapping

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BiEncoderDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        include_title: bool = True
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


class Collator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        queries, positives, negatives = zip(*batch)
        
        flattened_negatives = [neg for neg_list in negatives for neg in neg_list]

        query_inputs = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        positive_inputs = self.tokenizer(
            positives,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        negative_inputs = self.tokenizer(
            flattened_negatives,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        labels = torch.arange(len(batch), dtype=torch.long)

        return query_inputs, positive_inputs, negative_inputs

# if __name__ == "__main__":
#     data_file = "dataset/ZaloTextRetrieval/dataset_negatives/test_negatives.json"
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
#     dataset = BiEncoderDataset(json_file=data_file, tokenizer=tokenizer, max_length=256, include_title=True)

#     for i in range(1):
#         query, positive_text, negative_texts = dataset[i]
#         print(f"Query: {query}")
#         print(f"Positive Document: {positive_text}")
#         print(f"Negative Documents:{negative_texts}")