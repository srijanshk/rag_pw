import torch
from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer

class QuestionEncoder(PreTrainedModel):
    def __init__(self, config, model_name_or_path: str = None):
        super().__init__(config)
        self.e5_model = AutoModel.from_pretrained(model_name_or_path) if model_name_or_path else AutoModel.from_config(config)
        self.config = self.e5_model.config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
             model_kwargs["token_type_ids"] = token_type_ids
        
        outputs = self.e5_model(**model_kwargs, **kwargs)
        last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

        # Perform mean pooling (average of last_hidden_state weighted by attention_mask)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9) # Prevent division by zero
        pooled_output = sum_embeddings / sum_mask # Shape: [batch_size, hidden_size]
        # The output of a question encoder for RAG is usually just the pooled embeddings in a tuple
        return (pooled_output,)