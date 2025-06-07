import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from information_retrieval.utils.norm import normalize


BATCH_SIZE = 8
MODEL = "hiieu/halong_embedding"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()
model = model.to(device)
output_dir = "embedding_corpus"

model_name = MODEL.split('/')[-1]
save_dir = f"{output_dir}/{model_name}"
os.makedirs(save_dir, exist_ok=True)

"""
Format of the legal corpus
"""

with open("dataset/ZaloTextRetrieval/legal_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

docs = []
for law in corpus:
    law_id = law["law_id"]
    for art in law["articles"]:
        docs.append({
            "law_id": law_id,
            "article_id": art["article_id"],
            "title": art["title"],
            "text": art["text"]
        })

# with open(f"{save_dir}/legal_corpus_docs.json", "w", encoding="utf-8") as f:
#     json.dump(docs, f, ensure_ascii=False, indent=2)


"""
Encode the legal corpus
"""

embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Encoding documents"):
        batch_docs = docs[i:i+BATCH_SIZE]
        batch_texts = [doc["text"] for doc in batch_docs]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        normalized_emb = normalize(emb)
        embeddings.append(normalized_emb.cpu().numpy())

embeddings_np = np.concatenate(embeddings, axis=0)
np.save(f"{save_dir}/legal_corpus_embeddings.npy", embeddings_np)
