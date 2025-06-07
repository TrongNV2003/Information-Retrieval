python -m main \
    --dataloader_workers 2 \
    --seed 42 \
    --epochs 10 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --max_length 512 \
    --pad_mask_id -100 \
    --model hiieu/halong_embedding \
    --pin_memory \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --test_batch_size 8 \
    --train_file dataset/ZaloTextRetrieval/dataset_negatives/train_negatives.json \
    --val_file dataset/ZaloTextRetrieval/dataset_negatives/val_negatives.json \
    --test_file dataset/ZaloTextRetrieval/dataset_negatives/test_negatives.json \
    --output_dir ./models \
    --record_output_file output.json \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001 \
    --triplet_loss_margin 1.0 \
    --evaluate_on_mrr \
    # --use_triplet_loss \
    # --use_lora \
    # --lora_rank 8 \
    # --lora_alpha 16 \
    # --lora_dropout 0.1 \
    # --lora_target_modules "query, key, value, dense" \