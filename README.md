# Information Retrieval
Information retrieval with Encoder models, this IR systems combine both of these approaches by using a Bi-encoder to quickly retrieve a set of candidates, then using a Cross-encoder to re-rank candidate more accurately.

## System Overview
A typical high-performance search system operates in two stages to balance speed and accuracy:

Retrieval (Candidate Generation): A fast model (the Bi-Encoder) scans a massive corpus of documents (e.g., million documents) and quickly retrieves a smaller, relevant subset of candidates (e.g., top 10).

Re-ranking: A slower but more powerful model (the Cross-Encoder) carefully examines this small subset of candidates and re-orders them to produce the final, highly accurate ranking. In this phase, i use mine_hard_negatives of sentence_transformers for better performance of models.

## Dataset Usage
In this repo, i use dataset: Zalo-AI-2021 Legal Text Retrieval.

### Example Dataset
Hard negatives dataset are also created using BM25 and get top 5 negative documents relative to positive documents. After preprocessing, the training set has format as follow:
```json
[
    {
        "question": "...",
        "relevant_articles": [
            {
                "law_id": "...",
                "article_id": "...",
                "title": "...",
                "text": "..."
            }
        ],
        "num_articles": 1,
        "hard_negatives": [
            {
                "law_id": "...",
                "article_id": "...",
                "title": "...",
                "text": "..."
            },
            {
                "law_id": "...",
                "article_id": "...",
                "title": "...",
                "text": "..."
            },
            {
                "law_id": "...",
                "article_id": "...",
                "title": "...",
                "text": "..."
            }
        ],
    },
]
```



## Installation
```sh
pip install -r requirements.txt
```

## Usage
Training and evaluating for Bi-encoder model:
```sh
cd information_retrieval/train_scripts
bash train_be.sh
```

Training and evaluating for Cross-encoder model:
```sh
cd information_retrieval/train_scripts
bash train_ce.sh
```

## Metric Evaluation
MRR (Mean Reciprocal Rank): đo lường chất lượng của một hệ thống truy xuất thông tin bằng cách xem xét vị trí của kết quả liên quan đầu tiên trong danh sách xếp hạng được trả về cho một truy vấn.
Ví dụ: 
- Truy vấn 1: Kết quả liên quan ở vị trí 1 → Reciprocal Rank = 1/1 = 1.
- Truy vấn 2: Kết quả liên quan ở vị trí 3 → Reciprocal Rank = 1/3 ≈ 0.333.
- Truy vấn 3: Không có kết quả liên quan → Reciprocal Rank = 0.
- MRR = (1 + 0.333 + 0) / 3 ≈ 0.444.

NDCG (Normalized Discounted Cumulative Gain): đánh giá chất lượng xếp hạng, xem xét độ liên quan của các kết quả trong danh sách và vị trí của chúng (ví dụ: Top 1 đến 5).

## Future work
- TBD