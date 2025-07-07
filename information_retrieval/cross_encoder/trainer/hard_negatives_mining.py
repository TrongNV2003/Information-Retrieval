import torch
from sentence_transformers.util import mine_hard_negatives


class HardNegativesMining:
    def __init__(
        self,
        embedding_model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        batch_size: int = 4096,
        num_negatives: int = 5,
    ) -> None:
        self.model = embedding_model
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def mine(self):
        hard_negatives = mine_hard_negatives(
            model=self.model,
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_negatives=self.num_negatives,
            range_min=0,
            range_max=100,
            sampling_strategy="top",
            output_format="triplet",
            use_faiss=True
        )
        return hard_negatives
