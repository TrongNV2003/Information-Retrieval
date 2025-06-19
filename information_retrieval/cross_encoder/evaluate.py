import json
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Callable

import torch
from torch.utils.data import DataLoader, Dataset


class TestingArguments:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        model: torch.nn.Module,
        pin_memory: bool,
        test_set: Dataset,
        test_batch_size: int,
        collate_fn: Optional[Callable] = None,
        output_file: Optional[str] = None,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.test_loader = DataLoader(
            test_set,
            batch_size=test_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )
        self.output_file = output_file

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        results = []
        all_ranks = []
        latencies = []
        results = []
        
        with tqdm(self.test_loader, total=len(self.test_loader), unit="batches") as tepoch:
            for batch in tepoch:
                positive_inputs, negative_inputs = batch
                positive_inputs = {k: v.to(self.device) for k, v in positive_inputs.items()}
                negative_inputs = {k: v.to(self.device) for k, v in negative_inputs.items()}

                start_time = time.time()
                pos_scores = self.model(**positive_inputs).logits
                neg_scores = self.model(**negative_inputs).logits
                end_time = time.time()
                
                latencies.append(end_time - start_time)

                num_pos = pos_scores.size(0)
                num_neg = neg_scores.size(0)
                num_neg_per_pos = num_neg // num_pos
                
                for i in range(num_pos):
                    positive_score = pos_scores[i].item()
                    
                    start_idx = i * num_neg_per_pos
                    end_idx = (i + 1) * num_neg_per_pos
                    current_negative_scores = neg_scores[start_idx:end_idx]
                    
                    # Rank = 1 (cho cặp dương) + số lượng cặp âm có điểm cao hơn
                    rank = 1 + torch.sum(current_negative_scores > positive_score).item()
                    all_ranks.append(rank)

                    results.append({
                        "query_idx_in_batch": i,
                        "positive_score": positive_score,
                        "negative_scores": current_negative_scores.cpu().tolist(),
                        "rank": rank
                    })

                if all_ranks:
                    mrr_so_far = np.mean(1.0 / np.array(all_ranks))
                    tepoch.set_postfix({"MRR": f"{mrr_so_far:.4f}"})

        all_ranks = np.array(all_ranks)
        self._calculate_metrics(all_ranks)
        self._calculate_latency(latencies)

        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {self.output_file}")


    def _calculate_metrics(self, all_ranks: np.ndarray) -> None:
        num_queries = len(all_ranks)
        mrr = np.mean(1.0 / all_ranks)
        acc_at_1 = np.mean(all_ranks == 1)
        acc_at_3 = np.mean(all_ranks <= 3)
        acc_at_10 = np.mean(all_ranks <= 10)
        
        print("\n=== Reranking Evaluation Metrics ===")
        print(f"Number of queries: {num_queries}")
        print(f"MRR: {mrr * 100:.2f}%")
        print(f"Accuracy@1: {acc_at_1 * 100:.2f}%")
        print(f"Accuracy@3: {acc_at_3 * 100:.2f}%")
        print(f"Accuracy@10: {acc_at_10 * 100:.2f}%")


    def _calculate_latency(self, latencies: List[float]) -> None:
        latencies = np.array(latencies) * 1000
        p95_ms = np.percentile(latencies, 95)
        p99_ms = np.percentile(latencies, 99)
        mean_ms = np.mean(latencies)
        min_ms = np.min(latencies)
        max_ms = np.max(latencies)

        print("\n=== Latency Statistics ===")
        print(f"P95 Latency: {p95_ms:.2f} ms per sample")
        print(f"P99 Latency: {p99_ms:.2f} ms per sample")
        print(f"Mean Latency: {mean_ms:.2f} ms per sample")
        print(f"Min/Max Latency: {min_ms:.2f} / {max_ms:.2f} ms per sample")