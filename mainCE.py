import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import argparse
import numpy as np

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from information_retrieval.cross_encoder.trainer import TrainingArguments
from information_retrieval.cross_encoder.evaluate import TestingArguments
from information_retrieval.cross_encoder.dataloader import CrossEncoderDataset, CrossEncoderCollator
from information_retrieval.utils.model_utils import set_seed, get_vram_usage, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_vram_usage(device):
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)

def count_parameters(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

parser = argparse.ArgumentParser()
parser.add_argument("--dataloader_workers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10, required=True)
parser.add_argument("--learning_rate", type=float, default=3e-5, required=True)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--pad_mask_id", type=int, default=-100)
parser.add_argument("--model", type=str, default="vinai/phobert-base-v2", required=True)
parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
parser.add_argument("--train_batch_size", type=int, default=16, required=True)
parser.add_argument("--val_batch_size", type=int, default=16, required=True)
parser.add_argument("--test_batch_size", type=int, default=16, required=True)
parser.add_argument("--train_file", type=str, default="dataset/train.json", required=True)
parser.add_argument("--val_file", type=str, default="dataset/val.json", required=True)
parser.add_argument("--test_file", type=str, default="dataset/test.json", required=True)
parser.add_argument("--output_dir", type=str, default="./models", required=True)
parser.add_argument("--record_output_file", type=str, default="output.json")
parser.add_argument("--early_stopping_patience", type=int, default=5, required=True)
parser.add_argument("--early_stopping_threshold", type=float, default=0.001)
parser.add_argument("--evaluate_on_mrr", action="store_true", default=False)
parser.add_argument("--use_lora", action="store_true", default=False, help="Whether to use LoRA for fine-tuning")
parser.add_argument("--lora_rank", type=int, default=16, help="Rank for LoRA adaptation")
parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")
parser.add_argument("--lora_target_modules", type=str, default=None, help="Target modules for LoRA, defaults to query, value")
args = parser.parse_args()

def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    return tokenizer

def get_model(checkpoint: str, device: str) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
    model = model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    set_seed(args.seed)

    tokenizer = get_tokenizer(args.model)

    train_set = CrossEncoderDataset(json_file=args.train_file, tokenizer=tokenizer, include_title=False)
    val_set = CrossEncoderDataset(json_file=args.val_file, tokenizer=tokenizer, include_title=False)
    test_set = CrossEncoderDataset(json_file=args.test_file, tokenizer=tokenizer, include_title=False)
    collator = CrossEncoderCollator(tokenizer=tokenizer, max_length=args.max_length)

    model = get_model(args.model, device=device)

    count_parameters(model)

    model_name = args.model.split('/')[-1]
    if args.use_lora:
        save_dir = f"{args.output_dir}/{model_name}-lora"
    else:
        save_dir = f"{args.output_dir}/{model_name}"

    start_time = time.time()
    trainer = TrainingArguments(
        dataloader_workers=args.dataloader_workers,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        model=model,
        tokenizer=tokenizer,
        pin_memory=args.pin_memory,
        save_dir=save_dir,
        train_set=train_set,
        valid_set=val_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.val_batch_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        evaluate_on_mrr=args.evaluate_on_mrr,
        collator_fn=collator,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules.split(',') if args.lora_target_modules else None,
    )
    trainer.train()
    end_time = time.time()


    """
    Evaluation
    """
    if args.use_lora:
        from peft import PeftModel
        base_model = get_model(args.model, device=device)
        tuned_model = PeftModel.from_pretrained(base_model, save_dir)
    else:
        tuned_model = get_model(save_dir, device=device)
    
    tester = TestingArguments(
        dataloader_workers=args.dataloader_workers,
        device=device,
        model=tuned_model,
        pin_memory=args.pin_memory,
        test_set=test_set,
        test_batch_size=args.test_batch_size,
        collate_fn=collator,
        output_file=args.record_output_file,
    )
    tester.evaluate()

    if torch.cuda.is_available():
        max_vram = get_vram_usage(device)
        print(f"VRAM tối đa tiêu tốn khi huấn luyện: {max_vram:.2f} GB")
    print(f"Training time: {(end_time - start_time) / 60} mins")
    print(f"\nmodel: {args.model}")
    print(f"params: lr {args.learning_rate}, epoch {args.epochs}")
