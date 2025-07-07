import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
my_path = os.path.abspath(os.path.dirname(__file__).replace("ir/cross_encoder", ""))
sys.path.append(my_path)

import torch
import mlflow
import argparse
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainingArguments, CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss, CrossEntropyLoss

from information_retrieval.callbacks.memory_callback import MemoryLoggerCallback
from information_retrieval.cross_encoder.trainer.hard_negatives_mining import HardNegativesMining
from information_retrieval.cross_encoder.trainer.dataloader import load_train_dataset
from information_retrieval.utils.model_utils import set_seed, count_parameters
from information_retrieval.cross_encoder.trainer.metrics import compute_rerank_metrics as compute_metrics

mlflow.set_experiment("cross-encoder")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vinai/phobert-base-v2", required=True, help="Model checkpoint or name")
parser.add_argument("--embedding_model", type=str, default="hiieu/halong_embedding", help="Embedding model for hard negatives mining")
parser.add_argument("--train_file", type=str, default="dataset/train_word.json", required=True, help="Path to training data")
parser.add_argument("--valid_file", type=str, default="dataset/dev_word.json", required=True, help="Path to validation data")
parser.add_argument("--test_file", type=str, default="dataset/test_word.json", required=True, help="Path to test data")
parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="Optimizer to use for training")
parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model and logs")
parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Number of warmup steps")
parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size for evaluation")
parser.add_argument("--num_hard_negatives", type=int, default=5, help="Number of hard negatives to mine per positive example")
parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy")
parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Save strategy")
parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to save")
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
parser.add_argument("--logging_dir", type=str, default=None, help="Directory for logging")
parser.add_argument("--fp16", action="store_true", default=False, help="Enable mixed precision training (FP16)")
parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16 training")
parser.add_argument("--metric_for_best_model", type=str, default="epoch", help="Metric to select best model")
parser.add_argument("--greater_is_better", action="store_true", default=True, help="Whether higher metric is better")
parser.add_argument("--load_best_model_at_end", action="store_true", default=True, help="Load best model at the end")
parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for testing")
parser.add_argument("--dataloader_num_workers", type=int, default=2, help="Number of dataloader workers")
parser.add_argument("--pin_memory", action="store_true", default=False, help="Pin memory for dataloader")
parser.add_argument("--include_title", action="store_true", default=False, help="Include title in the corpus")
parser.add_argument("--record_output_path", type=str, default="output.json", help="Output file for evaluation results")
parser.add_argument("--report_to", type=str, help="Reporting tool for training metrics")
args = parser.parse_args()


def get_tokenizer(checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    return tokenizer

def get_model(checkpoint: str, device: str):
    model = CrossEncoder(checkpoint, device=device, num_labels=1)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    set_seed(args.seed)
    
    tokenizer = get_tokenizer(args.model)
    model = get_model(args.model, device)

    train_set = load_train_dataset(json_file=args.train_file, tokenizer=tokenizer, include_title=args.include_title)

    embedding_model = SentenceTransformer(args.embedding_model, device=device)
    
    # hard_negatives_mining = HardNegativesMining(
    #     embedding_model=embedding_model,
    #     train_dataset=train_set,
    #     batch_size=args.per_device_train_batch_size,
    #     num_negatives=args.num_hard_negatives,
    # )
    # train_set = hard_negatives_mining.mine()
    # train_set.save_to_disk(f"{args.output_dir}/train_hard_negative_dataset")
    
    val_evaluator = compute_metrics(args.valid_file, name="val", tokenizer=tokenizer, batch_size=args.per_device_eval_batch_size, include_title=args.include_title)
    
    train_loss = MultipleNegativesRankingLoss(model=model)
    
    count_parameters(model)
    
    model_name = args.model.split('/')[-1]
    save_dir = f"{args.output_dir}/{model_name}"
    logging_dir = args.logging_dir if args.logging_dir else f"{save_dir}/logs"

    training_args = CrossEncoderTrainingArguments(
        output_dir=save_dir,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        logging_dir=logging_dir,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=args.load_best_model_at_end,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.pin_memory,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=None,
        tokenizer=tokenizer,
        loss=train_loss,
        evaluator=val_evaluator,
        callbacks=[MemoryLoggerCallback],
    )
    trainer.train()

    best_model_path = f"{save_dir}/best_model"
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path, exist_ok=True)
    trainer.save_model(best_model_path)
    print(f"Đã lưu model tốt nhất tại: {best_model_path}")

    # Evaluation
    test_evaluator = compute_metrics(args.test_file, name="test", tokenizer=tokenizer, batch_size=args.test_batch_size, include_title=args.include_title)
    test_results = test_evaluator(model=model, output_path=args.record_output_path)
    
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
