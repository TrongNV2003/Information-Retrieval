import json
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BiEncoderValDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        corpus_meta_file: str,
    ) -> None:
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        with open(corpus_meta_file, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        self.id2idx = {(d["law_id"], d["article_id"]): i for i, d in enumerate(self.corpus)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        query = item["question"]
        rel_articles = item["relevant_articles"]
        
        pos_articles = rel_articles[0]
        pos_idx = self.id2idx.get((pos_articles["law_id"], pos_articles["article_id"]))
        
        return query, pos_idx


class BiEncoderValCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]):
        queries, positive_idxs = zip(*batch)

        query_inputs = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        positive_idxs = torch.tensor(positive_idxs, dtype=torch.long)

        return query_inputs, positive_idxs, queries
    

# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
#     dataset = BiEncoderValDataset(json_file="dataset/ZaloTextRetrieval/dataset_norm/test.json", corpus_meta_file="legal_corpus_docs.json")

#     for i in range(1):
#         query, candidate_indices, positive_index = dataset[i]
#         print(f"Query: {query}")
#         print(f"Candidate Indices: {candidate_indices}")
#         print(f"Positive Index: {positive_index}")