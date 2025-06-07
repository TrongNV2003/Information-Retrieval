import json
import time
import faiss
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from information_retrieval.utils.norm import normalize


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
        corpus_embedding_file: Optional[str] = None,
        top_k: int = 10,
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
        self.top_k = top_k
        

        corpus_emb = np.load(corpus_embedding_file).astype(np.float32)
        self.corpus_emb = corpus_emb
        d = corpus_emb.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(corpus_emb)
        self.faiss_index.add(corpus_emb)


    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        results = []
        all_ranks = []
        latencies = []
        all_hits_at_k = []
        all_ndcg_at_k = []
        
        num_processed_batches = 0
        
        with tqdm(self.test_loader, total=len(self.test_loader), unit="batches") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                try:
                    query_inputs, positive_idxs, queries_text = batch
                    query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
                    positive_index = positive_idxs.cpu().numpy()
                    batch_size = query_inputs["input_ids"].size(0)
                    
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    batch_start_time = time.time()
                    
                    with torch.no_grad():
                        query_outputs = self.model(**query_inputs)
                        query_embeddings = query_outputs.last_hidden_state[:, 0, :]

                        query_embeddings = normalize(query_embeddings).cpu().numpy().astype(np.float32)
                        faiss.normalize_L2(query_embeddings)

                    _, indices = self.faiss_index.search(query_embeddings, self.top_k)
                    
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    batch_end_time = time.time()
                    
                    latency = batch_end_time - batch_start_time
                    latencies.append(latency)

                    ranks = []
                    current_batch_hits_at_k = []
                    current_batch_ndcg_at_k = []

                    for i, pos_idx in enumerate(positive_idxs):
                        rank = np.where(indices[i] == pos_idx)[0]
                        if len(rank) > 0:
                            rank = rank[0] + 1
                            ranks.append(rank)
                            current_batch_hits_at_k.append(1)
                            ndcg = 1.0 / np.log2(rank + 1)
                        else:
                            rank = self.top_k + 1
                            ranks.append(rank)
                            current_batch_hits_at_k.append(0)
                            ndcg = 0.0
                        current_batch_ndcg_at_k.append(ndcg)

                    all_ranks.extend(ranks)
                    all_hits_at_k.extend(current_batch_hits_at_k)
                    all_ndcg_at_k.extend(current_batch_ndcg_at_k)

                    for i in range(batch_size):
                        results.append({
                            "question_idx_in_batch": int(i),
                            "question": queries_text[i],
                            "positive_col_index": int(positive_index[i].item()),
                            "rank": int(ranks[i]),
                            "hit@k": int(current_batch_hits_at_k[i]),
                            "ndcg@k": float(current_batch_ndcg_at_k[i]),
                            "latency_per_sample_ms": float(latency * 1000) / batch_size,
                        })

                    mrr_so_far = np.mean([1.0 / r if r <= self.top_k else 0.0 for r in all_ranks]) if all_ranks else 0
                    avg_rank = np.mean(ranks)
                    tepoch.set_postfix({
                        "MRR": f"{mrr_so_far:.4f}",
                        "AvgRank": f"{avg_rank:.1f}"
                    })
                    
                    num_processed_batches += 1

                except Exception as e:
                    import traceback
                    print(f"Error processing batch {batch_idx}: {e}")
                    print(traceback.format_exc())
                    print(f"Batch contents: {type(batch)}, {len(batch)}")
                    if isinstance(batch, (list, tuple)):
                        print(f"query_inputs keys: {batch[0].keys()}")
                        print(f"positive_idxs shape: {batch[1].shape}")
                        if len(batch) > 2:
                            print(f"question_texts: {batch[2][:2]}...")
                    continue
            
        all_ranks = np.array(all_ranks)
        all_hits_at_k = np.array(all_hits_at_k)
        all_ndcg_at_k = np.array(all_ndcg_at_k)
        
        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {self.output_file}")

        metrics = self._calculate_metrics(all_ranks, all_hits_at_k, all_ndcg_at_k)
        
        latency_stats = self._calculate_latency_stats(latencies, batch_size)
        metrics["latency"] = latency_stats

        print(f"Number of successfully processed samples: {len(all_ranks)}")
        return metrics

    def _calculate_metrics(self, all_ranks: np.ndarray, all_hits_at_k: np.ndarray, all_ndcg_at_k: np.ndarray) -> Dict[str, float]:
        metrics = {}
        if all_ranks.size > 0:
            mrr = np.mean([1.0 / r if r <= self.top_k else 0.0 for r in all_ranks])
            acc_at_1 = np.mean(all_ranks == 1)
            precision_at_5 = np.mean(all_ranks <= 5)
            precision_at_10 = np.mean(all_ranks <= 10)
            
            metrics["mrr"] = float(mrr)
            metrics["accuracy@1"] = float(acc_at_1)
            metrics["precision@5"] = float(precision_at_5)
            metrics["precision@10"] = float(precision_at_10)

            rank_distribution = {
                "min": int(all_ranks.min()),
                "max": int(all_ranks.max()),
                "mean": float(all_ranks.mean()),
                "median": float(np.median(all_ranks)),
                "std": float(all_ranks.std()),
                "count_top1": int(np.sum(all_ranks == 1)),
                "count_top5": int(np.sum(all_ranks <= 5)),
                "count_top10": int(np.sum(all_ranks <= 10)),
                "count_top50": int(np.sum(all_ranks <= 50)),
                "count_top100": int(np.sum(all_ranks <= 100)),
            }
            metrics["rank_distribution"] = rank_distribution
        else:
            metrics["mrr"] = 0.0
            metrics["accuracy@1"] = 0.0
            metrics["precision@5"] = 0.0
            metrics["precision@10"] = 0.0
            metrics["rank_distribution"] = {}
        
        if all_hits_at_k.size > 0:
            metrics[f"precision@{self.top_k}"] = float(all_hits_at_k.mean())
        else:
            metrics[f"precision@{self.top_k}"] = 0.0

        if all_ndcg_at_k.size > 0:
            metrics[f"ndcg@{self.top_k}"] = float(all_ndcg_at_k.mean())
        else:
            metrics[f"ndcg@{self.top_k}"] = 0.0
        
        print(f"\n=== Evaluation Metrics ===")
        print(f"MRR: {metrics.get('mrr', 0.0) * 100:.2f}%")
        print(f"MRR@{self.top_k}: {metrics.get('mrr', 0.0) * 100:.2f}%")
        print(f"Accuracy@1: {metrics.get('accuracy@1', 0.0) * 100:.2f}%")
        print(f"Precision@5: {metrics.get('precision@5', 0.0) * 100:.2f}%")
        print(f"Precision@{self.top_k}: {metrics.get(f'precision@{self.top_k}', 0.0) * 100:.2f}%")
        print(f"NDCG@{self.top_k}: {metrics.get(f'ndcg@{self.top_k}', 0.0) * 100:.2f}%")
        
        if "rank_distribution" in metrics and metrics["rank_distribution"]:
            dist = metrics["rank_distribution"]
            total = len(all_ranks)
            print(f"\n=== Rank Distribution ===")
            print(f"Min rank: {dist['min']}, Max rank: {dist['max']}")
            print(f"Mean rank: {dist['mean']:.1f}, Median rank: {dist['median']:.1f}")
            print(f"Top-1: {dist['count_top1']}/{total} ({dist['count_top1']/total*100:.2f}%)")
            print(f"Top-5: {dist['count_top5']}/{total} ({dist['count_top5']/total*100:.2f}%)")
            print(f"Top-10: {dist['count_top10']}/{total} ({dist['count_top10']/total*100:.2f}%)")
            print(f"Top-50: {dist['count_top50']}/{total} ({dist['count_top50']/total*100:.2f}%)")
            print(f"Top-100: {dist['count_top100']}/{total} ({dist['count_top100']/total*100:.2f}%)")
        
        return metrics

    def _calculate_latency_stats(self, latencies: List[float], batch_size: int = 1) -> Dict[str, float]:
        """Tính toán thống kê về latency"""
        latencies_array = np.array(latencies)
        
        per_sample_latencies = latencies_array * 1000 / batch_size
        
        stats = {
            "p50_ms": float(np.percentile(per_sample_latencies, 50)),
            "p95_ms": float(np.percentile(per_sample_latencies, 95)),
            "p99_ms": float(np.percentile(per_sample_latencies, 99)),
            "mean_ms": float(np.mean(per_sample_latencies)),
            "min_ms": float(np.min(per_sample_latencies)),
            "max_ms": float(np.max(per_sample_latencies)),
            "batch_size": int(batch_size),
            "total_samples": int(len(latencies) * batch_size),
            "total_batches": int(len(latencies)),
        }
        
        print("\n=== Latency Statistics ===")
        print(f"Mean Latency: {stats['mean_ms']:.2f} ms per sample")
        print(f"P50 Latency: {stats['p50_ms']:.2f} ms per sample")
        print(f"P95 Latency: {stats['p95_ms']:.2f} ms per sample")
        print(f"P99 Latency: {stats['p99_ms']:.2f} ms per sample")
        print(f"Min/Max Latency: {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms per sample")
        print(f"Total number of batches: {stats['total_batches']}")
        print(f"Batch size: {stats['batch_size']}")
        return stats