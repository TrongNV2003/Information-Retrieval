from torch import nn

class LabelingModel(nn.Module):
    def __init__(self, base_model: nn.Module, pooling_type: str = "mean"):
        super(LabelingModel, self).__init__()
        self.base_model = base_model
        self.pooling_type = pooling_type
        self.base_model.config.pooling_type = pooling_type
        
    def pool(self, hidden_states, attention_mask):  # [batch_size, seq_len, hidden_size]
        if self.pooling_type == "mean":
            hidden_states = hidden_states * attention_mask[:, :, None]
            pooled = hidden_states.mean(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
        elif self.pooling_type == "max":
            pooled = hidden_states.max(dim=1)
        elif self.pooling_type == "cls":
            pooled = hidden_states[:, 0]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        return pooled
    
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state
        pooled = self.pool(hidden_states, attention_mask)
        return pooled
    
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, model_path):
        from transformers import AutoModel
        print(f"Loading model from {model_path}")
        base_model = AutoModel.from_pretrained(model_path)
        pooling_type = getattr(base_model.config, 'pooling_type', 'cls')
        return cls(base_model, pooling_type)