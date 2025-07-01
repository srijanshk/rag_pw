import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap # For prettier printing

# --- Same Setup as Before ---
MODEL_NAME = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} onto device: {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
print("Model and tokenizer loaded successfully.")

# --- Reusable Summarization Function from the last step ---
def summarize_text(text_to_summarize, model, tokenizer):
    """
    Takes a string of text and returns a generated summary.
    """
    inputs = tokenizer.batch_encode_plus(
        [text_to_summarize],
        max_length=1024,
        return_tensors='pt',
        truncation=True
    ).to(device)

    # We can adjust these parameters to get longer or shorter summaries
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=150, # The max length of the summary
        min_length=30,  # The min length of the summary
        early_stopping=True
    )
    
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return summary

# --- New Function: The Summarization Layer ---
def summarize_all_docs(docs, model, tokenizer):
    """
    Takes a list of document strings and returns a list of their summaries.
    This function represents the "Summarization Layer".
    """
    print(f"\n--- Running Summarization Layer on {len(docs)} documents ---")
    summaries = []
    for i, doc in enumerate(docs):
        print(f"\nSummarizing Document {i+1}...")
        summary = summarize_text(doc, model, tokenizer)
        summaries.append(summary)
    return summaries

# --- Test Case ---

# Imagine these are the top 3 documents your DenseRetriever found for a question
retrieved_docs = [
    # Document 1 (from our previous test)
    """
    The problem asks for the number of integer solutions to the equation x^2 - y^2 = 256.
    This equation can be factored as a difference of squares: (x - y)(x + y) = 256.
    Let A = x - y and B = x + y. So, A * B = 256. Since x and y are integers, A and B must also be integers.
    Furthermore, we can express x and y in terms of A and B. Adding the two equations (A=x-y, B=x+y) gives 2x = A + B, so x = (A + B) / 2.
    Subtracting the first from the second gives 2y = B - A, so y = (B - A) / 2.
    For x and y to be integers, (A + B) and (B - A) must both be even. This implies that A and B must have the same parity (both even or both odd).
    Since their product A * B = 256 is an even number, it's impossible for both A and B to be odd.
    Therefore, both A and B must be even integers.
    """,
    # Document 2 (a different piece of retrieved context)
    """
    To find the number of solutions for (x-y)(x+y) = 256, we must find all integer factor pairs of 256.
    The prime factorization of 256 is 2^8. The number of divisors is (8+1) = 9.
    The divisors are 1, 2, 4, 8, 16, 32, 64, 128, 256.
    We must also consider the negative factors. For every positive factor 'd', '-d' is also a factor.
    So, there are 2 * 9 = 18 integer factors in total.
    Each factor 'A' corresponds to a unique co-factor 'B' such that A*B = 256.
    Thus, there are 18 ordered integer factor pairs for 256.
    """,
    # Document 3 (another piece of context, might be redundant or supplementary)
    """
    A key constraint for x = (A+B)/2 and y = (B-A)/2 to be integers is that A and B must share the same parity.
    If A is even and B is odd, their sum and difference are odd, so x and y would not be integers.
    If A and B are both odd, their product A*B would be odd. But A*B = 256, which is even. So this case is not possible.
    This forces both A and B to be even.
    This means we only need to consider the even factors of 256.
    """
]

# Run the Summarization Layer on the sample documents
list_of_summaries = summarize_all_docs(retrieved_docs, model, tokenizer)

# Print the final list of summaries
print("\n\n--- FINAL OUTPUT: LIST OF GENERATED SUMMARIES ---")
wrapper = textwrap.TextWrapper(width=100) # For clean printing
for i, summary in enumerate(list_of_summaries):
    print(f"\nSUMMARY OF DOC {i+1}:")
    print(wrapper.fill(summary))