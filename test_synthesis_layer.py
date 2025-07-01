import textwrap

# --- Test Case ---

# This is the original question for which we retrieved documents.
original_question = "What is the number of integer solutions to the equation x^2 - y^2 = 256?"

# This is the output from our previous script, test_summarization_layer.py
# It's a list of summaries for the top-k retrieved documents.
list_of_summaries = [
    "The problem asks for the number of integer solutions to the equation x^2 - y^2 = 256. This equation can be factored as a difference of squares: (x - y)(x + y) = 256.",
    "The prime factorization of 256 is 2^8. The divisors are 1, 2, 4, 8, 16, 32, 64, 128, 256. For every positive factor 'd', '-d' is also a factor.",
    "A key constraint for x = (A+B)/2 and y = (B-A)/2 is that A and B must share the same parity."
]

# --- New Function: The Synthesis Layer ---
def synthesize_trace(question, summaries, reasoner_tokenizer=None):
    """
    Takes the original question and the list of summaries and synthesizes them
    into a final prompt for the Llama reasoner model.

    (Note: The tokenizer is optional here, this function primarily deals with string formatting).
    """
    print("\n--- Running Synthesis Layer ---")
    
    # Join the individual summaries into a single context block.
    # We'll use bullet points for clarity.
    synthesized_context = "\n".join(f"- {s}" for s in summaries)
    
    # This is the final, complete prompt that will be fed to Llama 3.1
    # It follows the "Chain of Thought" or "Step-by-step" prompting style.
    final_prompt = (
        f"Question: {question}\n\n"
        f"Based on the following synthesized context, think step by step to provide a detailed answer.\n\n"
        f"Synthesized Context:\n{synthesized_context}\n\n"
        f"Step-by-step Solution:"
    )
    
    return final_prompt

# Run the Synthesis Layer to get the final prompt
final_llama_prompt = synthesize_trace(original_question, list_of_summaries)

# Print the final prompt that is ready for the Llama 3.1 model
print("\n\n--- FINAL PROMPT FOR LLAMA REASONER ---")
# Use textwrap for clean printing of the final formatted prompt
print(textwrap.fill(final_llama_prompt, width=100))