#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir results

python -m information_retrieval.bi_encoder.main \
    --dataloader_num_workers 2 \
    --seed 42 \
    --learning_rate 5e-5 \
    --num_train_epochs 15 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_length 512 \
    --optim adamw_torch_fused \
    --lr_scheduler_type linear \
    --model hiieu/halong_embedding \
    --pin_memory \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --test_batch_size 32 \
    --train_file dataset/ZaloTextRetrieval/dataset_negatives_reranks_clean/train_negatives.json \
    --valid_file dataset/ZaloTextRetrieval/dataset_negatives_reranks_clean/val_negatives.json \
    --test_file dataset/ZaloTextRetrieval/dataset_negatives_reranks_clean/test_negatives.json \
    --corpus_file embedding_corpus/legal_corpus_docs.json \
    --output_dir ./models \
    --record_output_path ./results \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 100 \
    --logging_dir ./models/logs \
    --fp16 \
    --metric_for_best_model val_cosine_mrr_at_10 \
    --greater_is_better \
    --load_best_model_at_end \
    --report_to mlflow \
