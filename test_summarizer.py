import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Step 1: Specify the pre-trained summarization model
MODEL_NAME = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} onto device: {device}")
# Step 2: Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Step 3: Create a sample document to summarize
# This is an example of what a retrieved document chunk from your OpenMath index might look like.
sample_document_text = """
The problem asks for the number of integer solutions to the equation x^2 - y^2 = 256.
This equation can be factored as a difference of squares: (x - y)(x + y) = 256.
Let A = x - y and B = x + y. So, A * B = 256. Since x and y are integers, A and B must also be integers.
Furthermore, we can express x and y in terms of A and B. Adding the two equations (A=x-y, B=x+y) gives 2x = A + B, so x = (A + B) / 2.
Subtracting the first from the second gives 2y = B - A, so y = (B - A) / 2.
For x and y to be integers, (A + B) and (B - A) must both be even. This implies that A and B must have the same parity (both even or both odd).
Since their product A * B = 256 is an even number, it's impossible for both A and B to be odd.
Therefore, both A and B must be even integers.
The task now is to find the number of pairs of even integer factors (A, B) of 256.
The factors of 256 (which is 2^8) are 2^0, 2^1, ..., 2^8. We also have their negative counterparts.
The integer factors of 256 are: +/-1, +/-2, +/-4, +/-8, +/-16, +/-32, +/-64, +/-128, +/-256. There are 9 positive factors and 9 negative factors, for a total of 18 integer factors.
We need to find pairs (A,B) where both are even. The even factors are +/-2, +/-4, ..., +/-256. This gives 8 positive even factors and 8 negative even factors, total of 16 even factors.
For every even factor A, B is uniquely determined as B = 256/A. Since 256 is a power of 2, if A is an even integer, B will also be an even integer.
So, there are 16 possible choices for A, and each choice gives a valid corresponding B. This means there are 16 pairs (A,B), and thus 16 integer solutions (x,y).
"""

def summarize_text(text_to_summarize, model, tokenizer):
    """
    Takes a string of text and returns a generated summary.
    """
    print("\n--- Summarizing Text ---")
    print("Original Text Length:", len(text_to_summarize.split()))

    # Tokenize the input text
    inputs = tokenizer.batch_encode_plus(
        [text_to_summarize],
        max_length=1024, # BART's max context size
        return_tensors='pt',
        truncation=True
    ).to(device)

    # Generate the summary
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=150, # The max length of the summary
        min_length=30,  # The min length of the summary
        early_stopping=True
    )

    # Decode the summary and remove special tokens
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print("Summary Text Length:", len(summary.split()))
    return summary

# Step 4: Run the summarization and print the result
generated_summary = summarize_text(sample_document_text, model, tokenizer)

print("\n--- GENERATED SUMMARY ---")
print(generated_summary)