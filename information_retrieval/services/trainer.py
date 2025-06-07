import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel

import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import Optional, Callable

from information_retrieval.utils.norm import normalize
from information_retrieval.utils.utils import AverageMeter
from information_retrieval.services.loss import LossFunctionFactory

class TrainingArguments:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_set: Dataset,
        valid_set: Dataset,
        train_batch_size: int,
        valid_batch_size: int,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001,
        evaluate_on_mrr: bool = True,
        collator_fn: Optional[Callable] = None,
        use_triplet_loss: bool = False,
        triplet_loss_margin: float = 1.0,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list = None,
        corpus_embedding_file: Optional[str] = None,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        
        self.use_triplet_loss = use_triplet_loss
        self.triplet_loss_margin = triplet_loss_margin
        self.corpus_embedding_file = corpus_embedding_file

        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=collator_fn,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collator_fn,
        )
        self.use_lora = use_lora

        if self.use_lora:
            lora_target_modules = ["query", "value"] if lora_target_modules is None else lora_target_modules
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                use_rslora=False
            )
            self.model = get_peft_model(model, lora_config)
            self.model.to(self.device)

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
            logger.info("Using LoRA for model training.")

            train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Training {train_params:,d} parameters out of {all_params:,d} parameters ({train_params/all_params:.2%})")
        else:
            self.model = model.to(self.device)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        self.loss_factory = LossFunctionFactory()
        if self.use_triplet_loss:
            self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
            logger.info("Using Triplet Loss")
        else:
            self.loss_fn = self.loss_factory.get_loss("multiple_negatives_ranking")
            # self.loss_fn = self.loss_factory.get_loss("ranking", tau=20)
            logger.info("Using Ranking Loss")

        
        num_training_steps = len(self.train_loader) * epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.evaluate_on_mrr = evaluate_on_mrr
        self.best_mrr = 0 if evaluate_on_mrr else float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_counter = 0
        self.best_epoch = 0

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = AverageMeter()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for batch in self.train_loader:
                    query_inputs, positive_inputs, negative_inputs = batch
                    query_input_ids = query_inputs["input_ids"].to(self.device)
                    query_attention_mask = query_inputs["attention_mask"].to(self.device)
                    positive_input_ids = positive_inputs["input_ids"].to(self.device)
                    positive_attention_mask = positive_inputs["attention_mask"].to(self.device)
                    negative_input_ids = negative_inputs["input_ids"].to(self.device)
                    negative_attention_mask = negative_inputs["attention_mask"].to(self.device)

                    self.optimizer.zero_grad()

                    query_embeddings = self.model(input_ids=query_input_ids, attention_mask=query_attention_mask)
                    positive_embeddings = self.model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
                    negative_embeddings = self.model(input_ids=negative_input_ids, attention_mask=negative_attention_mask)

                    query_embeddings = query_embeddings.last_hidden_state[:, 0, :]
                    positive_embeddings = positive_embeddings.last_hidden_state[:, 0, :]
                    negative_embeddings = negative_embeddings.last_hidden_state[:, 0, :]

                    query_embeddings = normalize(query_embeddings)
                    positive_embeddings = normalize(positive_embeddings)
                    negative_embeddings = normalize(negative_embeddings)

                    loss = self.loss_fn(query_embeddings, positive_embeddings, negative_embeddings)
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    train_loss.update(loss.item(), query_input_ids.size(0))
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    tepoch.set_postfix({"train_loss": train_loss.avg, "lr": current_lr})
                    tepoch.update(1)

            valid_score = self._validate(self.valid_loader)
            
            improved = False
            if self.evaluate_on_mrr:
                if valid_score > self.best_mrr + self.early_stopping_threshold:
                    print(f"Validation MRR improved from {self.best_mrr:.4f} to {valid_score:.4f}. Saving...")
                    self.best_mrr = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                else:
                    self.early_stopping_counter += 1
                    print(f"No improvement in val MRR. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

            else:
                if valid_score < self.best_mrr - self.early_stopping_threshold:
                    print(f"Validation loss decreased from {self.best_mrr:.4f} to {valid_score:.4f}. Saving...")
                    self.best_mrr = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                else:
                    self.early_stopping_counter += 1
                    print(f"No improvement in validation loss. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

            if improved:
                print(f"Saved best model at epoch {self.best_epoch}.")
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement.")
                break

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        all_ranks = []

        valid_samples = 0
        total_samples = 0
        
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for batch_idx, batch in enumerate(dataloader):
                query_inputs, positive_inputs, negative_inputs = batch
                query_input_ids = query_inputs["input_ids"].to(self.device)
                query_attention_mask = query_inputs["attention_mask"].to(self.device)
                positive_input_ids = positive_inputs["input_ids"].to(self.device)
                positive_attention_mask = positive_inputs["attention_mask"].to(self.device)
                negative_input_ids = negative_inputs["input_ids"].to(self.device)
                negative_attention_mask = negative_inputs["attention_mask"].to(self.device)
                
                batch_size = query_input_ids.size(0)
                total_samples += batch_size

                query_embeddings = self.model(input_ids=query_input_ids, attention_mask=query_attention_mask)
                positive_embeddings = self.model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
                negative_embeddings = self.model(input_ids=negative_input_ids, attention_mask=negative_attention_mask)
                
                # Comment lại nếu dùng custom pooling
                query_embeddings = query_embeddings.last_hidden_state[:, 0, :]
                positive_embeddings = positive_embeddings.last_hidden_state[:, 0, :]
                negative_embeddings = negative_embeddings.last_hidden_state[:, 0, :]

                query_embeddings = normalize(query_embeddings)
                positive_embeddings = normalize(positive_embeddings)
                negative_embeddings = normalize(negative_embeddings)
                
                for i in range(batch_size):
                    query_emb = query_embeddings[i:i+1]
                    positive_emb = positive_embeddings[i:i+1]
                    neg_embs = negative_embeddings[i*3:(i+1)*3]
                    
                    positive_score = F.cosine_similarity(query_emb, positive_emb).item()
                    negative_scores = F.cosine_similarity(query_emb, neg_embs).cpu().numpy()
                    
                    rank = 1
                    for neg_score in negative_scores:
                        if neg_score > positive_score:
                            rank += 1
                    all_ranks.append(rank)
                    valid_samples += 1

                if all_ranks:
                    mrr_so_far = np.mean(1.0 / np.array(all_ranks))
                    tepoch.set_postfix({
                        "valid_mrr": f"{mrr_so_far:.4f}",
                        "valid": f"{valid_samples}/{total_samples}"
                    })
                tepoch.update(1)
                    

        all_ranks = np.array(all_ranks)
        mrr = np.mean(1.0 / all_ranks)
        logger.info(f"Processed {len(all_ranks)} samples in validation")
        logger.info(f"MRR@4: {mrr:.4f}")
        self._metrics(all_ranks)
        return mrr

    def _metrics(self, all_ranks: np.ndarray) -> None:
        mrr = np.mean(1.0 / all_ranks)
        accuracy_at_1 = np.mean(all_ranks == 1)
        recall_at_3 = np.mean(all_ranks <= 3)
        recall_at_4 = np.mean(all_ranks <= 4)
        
        print(f"\n=== Metrics ===")
        print(f"Number of samples: {len(all_ranks)}")
        print(f"Rank statistics: min={all_ranks.min()}, max={all_ranks.max()}, mean={all_ranks.mean():.1f}, median={np.median(all_ranks):.1f}")
        print(f"MRR: {mrr * 100:.2f}%")
        print(f"Accuracy@1: {accuracy_at_1 * 100:.2f}%")
        print(f"Recall@3: {recall_at_3 * 100:.2f}%")
        print(f"Recall@4: {recall_at_4 * 100:.2f}%")

        count_top1 = np.sum(all_ranks == 1)
        count_top3 = np.sum(all_ranks <= 3)
        count_top4 = np.sum(all_ranks <= 4)

        print(f"\n=== Rank Distribution ===")
        print(f"Top-1: {count_top1}/{len(all_ranks)} samples ({count_top1 / len(all_ranks) * 100:.2f}%)")
        print(f"Top-3: {count_top3}/{len(all_ranks)} samples ({count_top3 / len(all_ranks) * 100:.2f}%)")
        print(f"Top-4: {count_top4}/{len(all_ranks)} samples ({count_top4 / len(all_ranks) * 100:.2f}%)")

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        if self.use_lora:
            self.model.save_pretrained(self.save_dir, save_embedding_layers=True)   # save LoRA weights
        else:
            self.model.save_pretrained(self.save_dir)   # save full model
