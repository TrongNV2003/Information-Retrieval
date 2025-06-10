import torch
import torch.nn.functional as F

def normalize(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:                            # (batch_size, emb_dim)
        return F.normalize(x, p=2, dim=1)
    elif x.dim() == 3:                          # (batch_size, num_candidates, emb_dim)
        return F.normalize(x, p=2, dim=2)
    else:
        if x.dim() == 1:                        # (emb_dim,)
            x = x.unsqueeze(0)
            x_normalized = F.normalize(x, p=2, dim=1)
            return x_normalized.squeeze(0)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {x.dim()}")
