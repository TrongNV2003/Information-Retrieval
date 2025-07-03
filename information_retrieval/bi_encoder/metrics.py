import re
from typing import Dict
from transformers import AutoTokenizer
from sentence_transformers import util
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from information_retrieval.bi_encoder.dataloader import EvalDataLoader

class MlflowEvaluatorWrapper(InformationRetrievalEvaluator):
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        metrics = super().__call__(model, output_path, epoch, steps)
        cleaned_metrics = {re.sub(r'@', '_at_', key): value for key, value in metrics.items()}
        return cleaned_metrics

def compute_metrics(
    json_file: str,
    corpus_file: str,
    name: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    include_title: bool = False,
    ) -> MlflowEvaluatorWrapper:

    data_loader = EvalDataLoader(
        json_file=json_file,
        corpus_file=corpus_file,
        tokenizer=tokenizer,
        include_title=include_title
    )

    queries, corpus, relevant_docs = data_loader.get_data()

    metrics = MlflowEvaluatorWrapper(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        score_functions={"cosine": util.cos_sim},
        main_score_function="cosine",
        mrr_at_k=[10],
        precision_recall_at_k=[1, 3, 5, 10],
        map_at_k=[10],
        batch_size=batch_size,
    )

    return metrics

