# Information Retrieval
Information retrieval with Encoder models.

- For Bi-encoder models, i use `MultipleNegativesRankingLoss` to compute the loss function, which enables efficient contrastive learning by comparing each positive pair in a batch against all other negatives in the same batch.

## Dataset usage
In this repo, i use dataset: Zalo-AI-2021 Legal Text Retrieval.

### Example dataset
Hard negatives dataset are also created using BM25 and get top 3 negative documents relative to positive documents. After preprocessing, the training set has format as follow:
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
training and evaluating models:
```sh
bash train.sh
```

## Future work
- Applying Rerank for finetuned models
- Build Cross-encoder model for Rerank task