import re
from typing import Dict
from transformers import AutoTokenizer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

from information_retrieval.cross_encoder.trainer.dataloader import load_val_dataset

class MlflowEvaluatorWrapper(CrossEncoderRerankingEvaluator):
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        metrics = super().__call__(model, output_path, epoch, steps)
        cleaned_metrics = {re.sub(r'@', '_at_', key): value for key, value in metrics.items()}
        return cleaned_metrics

def compute_rerank_metrics(
    json_file: str,
    name: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    include_title: bool = False,
    ) -> MlflowEvaluatorWrapper:

    eval_samples = load_val_dataset(
        json_file=json_file,
        tokenizer=tokenizer,
        include_title=include_title
    )

    evaluator = MlflowEvaluatorWrapper(
        samples=eval_samples,   # samples: List of {'query': '...', 'positive': [...], 'negative': [...]} dictionaries
        name=name,
        batch_size=batch_size,
        at_k=10,
    )
    
    return evaluator