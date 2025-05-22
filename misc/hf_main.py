
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from HFCompactRetriever import HFCompatDenseRetriever
import torch
import traceback

from transformers import (
    RagConfig,
    RagTokenizer,
    RagTokenForGeneration,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from DenseRetriever import DenseRetriever
from transformers import PreTrainedModel, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_FILE = "downloads/data/gold_passages_info/nq-train.json"
TEST_FILE = "downloads/data/gold_passages_info/nq-dev.json"

class E5QuestionEncoder(PreTrainedModel):
    def __init__(self, config, model_name_or_path):
        super().__init__(config)
        self.e5_model = AutoModel.from_pretrained(model_name_or_path, config=config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        # Get last hidden states from E5 base model
        # Pass token_type_ids if your model uses them (E5 typically doesn't, but good to be complete)
        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids
        
        outputs = self.e5_model(**model_kwargs, **kwargs)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Perform mean pooling (E5 style: average of last_hidden_state weighted by attention_mask)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) # Prevent division by zero
        pooled_output = sum_embeddings / sum_mask

        # The output of a question encoder for RAG is usually just the pooled embeddings in a tuple
        return (pooled_output,)

retriever_model_name = "models/retriever_finetuned_e5_best"
generator_model_name = "best_bart_model"

FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
METADATA_PATH = "/local00/student/shakya/wikipedia_metadata.jsonl"
n_docs_to_retrieve = 5

print(f"Initializing custom DenseRetriever with model: {retriever_model_name}")
my_custom_retriever = DenseRetriever(
    index_path=FAISS_INDEX_PATH,
    metadata_path=METADATA_PATH,
    device=device,
    model_name=retriever_model_name,
    ef_search=1500,
    ef_construction=200,
    fine_tune=False # Assuming inference mode
)
print("Custom DenseRetriever initialized.")

# --- Load RAG Model Components ---
# 1. Load your fine-tuned E5 model to be used as the RAG Question Encoder
# This ensures embedding space consistency with your FAISS index.
# print(f"Loading E5 question encoder from: {retriever_model_name}")
# e5_question_encoder = AutoModel.from_pretrained(retriever_model_name)
# e5_question_encoder.to(device)
# print("E5 question encoder loaded.")
print(f"Loading E5 question encoder (with custom pooling wrapper) from: {retriever_model_name}")
# First, load the configuration of your E5 model
e5_base_config = AutoConfig.from_pretrained(retriever_model_name)
# Then, instantiate your wrapper
e5_question_encoder = E5QuestionEncoder(config=e5_base_config, model_name_or_path=retriever_model_name)
e5_question_encoder.to(device)
print("E5 question encoder (with custom pooling) loaded.")



# 2. Load your Generator model
print(f"Loading generator model: {generator_model_name}")
generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
generator.to(device)
print("Generator model loaded.")

# 3. Create RagConfig
# Modify generator config to handle potential warnings (e.g., forced_bos_token_id for BART)
if hasattr(generator.config, "forced_bos_token_id") and generator.config.forced_bos_token_id is None :
    if generator.config.bos_token_id is not None:
        generator.config.forced_bos_token_id = generator.config.bos_token_id
        print(f"Set generator.config.forced_bos_token_id to {generator.config.bos_token_id}")
    # else:
        # You might need to set it to a specific ID like 0 if bos_token_id is also None,
        # but usually, BART models have a bos_token_id.

rag_config_for_model = RagConfig(
    question_encoder=e5_question_encoder.config.to_dict(),
    generator=generator.config.to_dict(),
    n_docs=n_docs_to_retrieve,
    index_name=None,  # Explicitly state no default HF dataset index is used by RagRetriever parent
    # Pass through necessary token IDs from the generator's config
    bos_token_id=generator.config.bos_token_id,
    eos_token_id=generator.config.eos_token_id,
    pad_token_id=generator.config.pad_token_id,
    decoder_start_token_id=generator.config.decoder_start_token_id,
    forced_bos_token_id=generator.config.forced_bos_token_id # Use the (potentially updated) value
)
print("RagConfig for model prepared.")

# 4. Create Tokenizers
print(f"Loading tokenizer for E5 question encoder: {retriever_model_name}")
question_enc_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
print(f"Loading tokenizer for generator: {generator_model_name}")
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)

# 5. Instantiate the HFCompatDenseRetriever wrapper
# Pass the RagConfig and tokenizers as required by the modified __init__
hf_retriever_wrapper = HFCompatDenseRetriever(
    config=rag_config_for_model,
    question_encoder_tokenizer=question_enc_tokenizer,
    generator_tokenizer=generator_tokenizer,
    custom_dense_retriever=my_custom_retriever
)
print("HFCompatDenseRetriever wrapper initialized.")

# 6. Create the RagTokenizer
rag_tokenizer = RagTokenizer(
    question_encoder=question_enc_tokenizer,
    generator=generator_tokenizer
)
print("RagTokenizer initialized.")

# 7. Instantiate the RagTokenForGeneration model
print("Instantiating RagTokenForGeneration model...")
rag_model = RagTokenForGeneration(
    config=rag_config_for_model,
    question_encoder=e5_question_encoder,
    generator=generator,
    retriever=hf_retriever_wrapper # This is now an instance of a RagRetriever subclass
)
rag_model.to(device)
print("RagTokenForGeneration model instantiated and moved to device.")

# 8. Perform Generation
print("\n--- Ready to perform RAG generation ---")
question = "What is the Capital of Austria?" 
print(f"\nInput Question: {question}")

inputs = rag_tokenizer(question, return_tensors="pt").to(device)

print("Generating answer...")
try:
    generated_ids = rag_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"], 
        num_beams=4,
        max_length=512,
        early_stopping=True,
        num_return_sequences=1,
    )
    generated_text = rag_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("\nGenerated Answer(s):")
    for i, text in enumerate(generated_text):
        print(f"{i+1}: {text}")

except Exception as e:
    print(f"An error occurred during generation: {e}")
    traceback.print_exc()