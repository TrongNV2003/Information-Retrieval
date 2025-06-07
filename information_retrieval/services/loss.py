import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets, weight=self.weight, reduction=self.reduction
        )


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        return F.cross_entropy(
            logits, targets, weight=self.class_weights, reduction=self.reduction
        )


class TripletLoss(nn.Module):
    """
    Triplet Loss for Information Retrieval.
    Công thức: L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    Trong đó:
        - d(x, y): khoảng cách Euclidean.
        - margin: ngưỡng để đảm bảo negative xa anchor hơn positive.
    """
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            margin (float): Ngưỡng margin cho triplet loss.
            reduction (str): Phương thức giảm loss ('mean', 'sum', hoặc 'none').
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor (torch.Tensor): Embedding của câu hỏi (anchor), shape (batch_size, embedding_dim).
            positive (torch.Tensor): Embedding của bài viết liên quan (positive), shape (batch_size, embedding_dim).
            negative (torch.Tensor): Embedding của bài viết không liên quan (negative), shape (batch_size, embedding_dim).
        Returns:
            torch.Tensor: loss value.
        """

        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)

        losses = torch.relu(distance_positive - distance_negative + self.margin)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class RankingLoss(nn.Module):
    def __init__(self, tau: float = 20):
        super(RankingLoss, self).__init__()
        self.tau = tau  # sharpening factor

    # "qi,pi->qp": nhân từng vector của query_vecs (kích thước q, d) với từng vector của doc_vecs (kích thước p, d) theo chiều d, để tạo ra ma trận cosine similarity kích thước (q, p).
    def forward(self, query_vecs: torch.Tensor, doc_vecs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine_similarity = torch.einsum("qi,pi->qp", 
                                         F.normalize(query_vecs, dim=-1, p=2), 
                                         F.normalize(doc_vecs, dim=-1, p=2))
        scores = cosine_similarity * self.tau
        scores = F.cross_entropy(scores, labels)
        return scores


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct: callable = F.cosine_similarity):
        """
        MultipleNegativesRankingLoss for Information Retrieval.
        
        Args:
            scale (float): Hệ số scale cho điểm tương đồng trước khi áp dụng softmax.
                          Mặc định là 20.0 (theo SentenceTransformers).
            similarity_fct (callable): Hàm tính độ tương đồng giữa query và document.
                                      Mặc định là cosine similarity.
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Tính loss cho batch dữ liệu.
        
        Args:
            query_embeddings (torch.Tensor): Embedding của queries, shape (batch_size, embedding_dim).
            positive_embeddings (torch.Tensor): Embedding của positive documents, shape (batch_size, embedding_dim).
            negative_embeddings (torch.Tensor, optional): Embedding của hard negatives, shape (num_negatives, embedding_dim).
                                                        Nếu None, sử dụng in-batch negatives.
        
        Returns:
            torch.Tensor: Giá trị loss.
        """
        batch_size = query_embeddings.size(0)
        
        similarity_scores = torch.zeros(batch_size, batch_size, device=query_embeddings.device)
        
        for i in range(batch_size):
            # Tính tương đồng với positive document của chính query i
            similarity_scores[i, i] = self.similarity_fct(
                query_embeddings[i:i+1], positive_embeddings[i:i+1]
            ) * self.scale
            
            # Tính tương đồng với positive documents của các query khác (in-batch negatives)
            for j in range(batch_size):
                if i != j:
                    similarity_scores[i, j] = self.similarity_fct(
                        query_embeddings[i:i+1], positive_embeddings[j:j+1]
                    ) * self.scale
        
        if negative_embeddings is not None:
            num_negatives = negative_embeddings.size(0)
            hard_negative_scores = torch.zeros(batch_size, num_negatives, device=query_embeddings.device)
            for i in range(batch_size):
                for j in range(num_negatives):
                    hard_negative_scores[i, j] = self.similarity_fct(
                        query_embeddings[i:i+1], negative_embeddings[j:j+1]
                    ) * self.scale
            # Kết hợp in-batch negatives và hard negatives
            similarity_scores = torch.cat([similarity_scores, hard_negative_scores], dim=1)
        
        # Nhãn: positive document của query i là document i (index i)
        labels = torch.arange(batch_size, dtype=torch.long, device=query_embeddings.device)
        
        loss = self.cross_entropy_loss(similarity_scores, labels)
        
        return loss

class LossFunctionFactory:
    @staticmethod
    def get_loss(loss_name, **kwargs):
        """
        Factory method để lấy loss function theo tên
        
        Args:
            loss_name: Tên của loss function
            **kwargs: Các tham số bổ sung cho loss function
            
        Returns:
            Một instance của loss function được yêu cầu
        """
        loss_dict = {
            # Single-label losses
            "ce": CrossEntropyLoss,
            "weighted_ce": WeightedCrossEntropyLoss,

            # Loss for Information Retrieval
            "triplet": TripletLoss,
            "ranking": RankingLoss,
            "multiple_negatives_ranking": MultipleNegativesRankingLoss,
        }
        
        if loss_name not in loss_dict:
            raise ValueError(f"Loss function '{loss_name}' not found. Available losses: {list(loss_dict.keys())}")
        
        return loss_dict[loss_name](**kwargs)