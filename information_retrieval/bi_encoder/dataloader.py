import json
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sentence_transformers.readers import InputExample

class BiEncoderDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        tokenizer: AutoTokenizer,
        include_title: bool = True,
    ) -> None:
        self.sep_token = tokenizer.sep_token
        self.include_title = include_title
        
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> InputExample:
        item = self.data[index]
        query = item["question"]
        positive_article = item["relevant_articles"][0]
        if self.include_title:
            positive_text = f"{positive_article['title']} {self.sep_token} {positive_article['text']}"
        else:
            positive_text = positive_article["text"]

        texts = [query, positive_text]
        for neg_article in item['hard_negatives']:
            if self.include_title:
                neg_text = f"{neg_article['title']} {self.sep_token} {neg_article['text']}"
            else:
                neg_text = neg_article['text']
            texts.append(neg_text)

        return InputExample(texts=texts, label=0)


class BiEncoderCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[InputExample]) -> tuple:
        # Mỗi InputExample.texts là một list [query, positive, neg1, neg2, ...]
        
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for i, text in enumerate(example.texts):
                texts[i].append(text)
            labels.append(example.label)

        tokenized_texts = []
        for text_col in texts:
            tokenized = self.tokenizer(
                text_col,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            tokenized_texts.append(tokenized)
            
        labels = torch.tensor(labels, dtype=torch.long)
        
        return tokenized_texts, labels

# if __name__ == "__main__":
#     data_file = "dataset/ZaloTextRetrieval/dataset_negatives/test_negatives.json"
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
#     dataset = BiEncoderDataset(json_file=data_file, tokenizer=tokenizer, include_title=True)

#     for i in range(1):
#         query, positive_text, negative_texts = dataset[i]
#         print(f"Query: {query}")
#         print(f"Positive Document: {positive_text}")
#         print(f"Negative Documents:{negative_texts}")