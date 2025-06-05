import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_HOME'] = '/system/user/studentwork/shakya/cache/directory'
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset as HuggingFaceDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" 
GSM8K_TEST_FILE = "./thesis_datasets/gsm8k/test.jsonl"
MATH_BASE_DIR = "./thesis_datasets/math_hendrycks"
MATH_SUBJECTS = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'] 

OUTPUT_RESULTS_DIR = "./baseline_results"
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 1 
MAX_NEW_TOKENS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model and Tokenizer Loading ---
print(f"Loading model: {LLAMA_MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )
    model.eval() # Set to evaluation mode

    # Set pad token if not set (common for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"Model and tokenizer loaded successfully on device: {model.device}")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

# --- Dataset Classes ---
class MathDataset(Dataset):
    def __init__(self, file_path, dataset_name="unknown"):
        self.data = []
        self.dataset_name = dataset_name
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line {i+1} in {file_path}")
        except FileNotFoundError:
            print(f"Error: Data file not found at {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # GSM8K typically has 'question' and 'answer'
        # MATH typically has 'problem' and 'solution' (solution contains the answer)
        question = item.get("question", item.get("problem", ""))
        # For MATH, the answer is often in the 'solution' field, typically in a \boxed{} tag
        # For GSM8K, the 'answer' field usually has the reasoning and final number
        answer_raw = item.get("answer", item.get("solution", ""))
        return {"id": f"{self.dataset_name}_{idx}", "question": question, "answer_raw": answer_raw}

# --- Prompt Formatting ---
def format_prompt_few_shot(problem_text):
    # Example few-shot prompt (adapt with high-quality examples relevant to GSM8K/MATH)
    # This is a very generic one, you should craft better examples.
    prompt_prefix = """Solve the following math problems step-by-step. Extract the final numerical answer.

Question: Natalia sold clips to 48 of her friends and then found 5 more. If Natalia has 60 clips now, how many did she have at first?
Answer: Natalia has 60 clips now. She found 5 more clips. So, before finding the 5 clips, she had 60 - 5 = 55 clips. She sold clips to 48 of her friends. So, before selling the clips, she had 55 + 48 = 103 clips.
The final answer is \\boxed{103}.

Question: A train travels from city A to city B. The distance between city A and city B is 360 km. The train travels at an average speed of 90 km/h. How long does it take for the train to travel from city A to city B?
Answer: The distance between city A and city B is 360 km. The average speed of the train is 90 km/h.
Time = Distance / Speed
Time = 360 km / 90 km/h
Time = 4 hours.
The final answer is \\boxed{4}.

"""
    # For Llama 3.1, the specific instruct/chat template should be used if available
    # For a base model, a simpler QA format might work.
    # This example is a general completion prompt.
    # If using an instruct-tuned Llama 3.1, refer to its specific prompt format
    # e.g. <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    # Using a simplified QA format for this example, assuming a base model or simple instruct fine-tune
    # For Llama 3.1 Instruct, you should use its specific chat template.
    # See: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/
    # For now, let's use a simple completion style for the few-shot examples.
    # The actual query to the model might look different if using the official chat template.

    formatted_problem = f"Question: {problem_text}\nAnswer:"
    full_prompt = prompt_prefix + formatted_problem
    return full_prompt

def format_prompt_zero_shot(problem_text):
    # For Llama 3.1 Instruct, you should use its specific chat template.
    # e.g., messages = [{"role": "user", "content": f"Solve the following math problem step-by-step. Extract the final numerical answer. Problem: {problem_text}"}]
    # tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # This example uses a simpler completion style.
    prompt = f"Solve the following math problem step-by-step. Provide the final answer as a number.\n\nProblem: {problem_text}\n\nSolution:"
    return prompt


# --- Answer Extraction ---
def extract_final_answer(generated_text: str) -> str:
    """
    Extracts the final numerical answer from the LLM's generated text.
    This is a simple version and often needs to be made more robust.
    Look for patterns like \\boxed{answer} or "The final answer is X".
    """
    # Try to find \boxed{answer}
    match_boxed = re.search(r"\\boxed\{([\d\.,\-]+)\}", generated_text)
    if match_boxed:
        return match_boxed.group(1).replace(",", "")

    # Try to find "The final answer is X"
    # This regex looks for a number (possibly with commas or a decimal point)
    # that follows "The final answer is" or "the final answer is".
    match_text = re.search(r"[Tt]he final answer is\s*([\d\.,\-]+)", generated_text)
    if match_text:
        return match_text.group(1).replace(",", "")

    # Fallback: try to get the last number in the string
    numbers = re.findall(r"[\d\.\-]+", generated_text)
    if numbers:
        return numbers[-1].replace(",", "")
        
    return "" # Return empty if no answer found

def extract_ground_truth_answer(answer_raw: str, dataset_name: str) -> str:
    """
    Extracts the numerical ground truth answer from the dataset's answer field.
    """
    if "gsm8k" in dataset_name.lower():
        # GSM8K answers are often like "Natalia had 55 + 48 = 103 clips. #### 103"
        # The final number after "####" is the answer.
        match = re.search(r"####\s*([\d\.,\-]+)", answer_raw)
        if match:
            return match.group(1).replace(",", "")
    elif "math" in dataset_name.lower(): # For MATH dataset
        # MATH answers are usually in \boxed{}
        match = re.search(r"\\boxed\{([\s\S]*?)\}", answer_raw) # Non-greedy match inside box
        if match:
            # Further simplify common LaTeX fractions or simple expressions if possible,
            # or just return the content. For now, just return content.
            # This part might need more sophisticated parsing for complex LaTeX answers.
            ans_content = match.group(1).strip()
            # Basic simplification for fractions like \frac{a}{b}
            frac_match = re.fullmatch(r"\\frac\{([\d\.,\-]+)\}\{([\d\.,\-]+)\}", ans_content)
            if frac_match:
                try:
                    num = float(frac_match.group(1))
                    den = float(frac_match.group(2))
                    return str(num / den)
                except ValueError:
                    return ans_content # Fallback if not a simple numeric fraction
            return ans_content.replace(",", "")
    
    # Fallback if no specific pattern matches
    numbers = re.findall(r"[\d\.\-]+", answer_raw)
    if numbers:
        return numbers[-1].replace(",", "")
    return answer_raw # Or "" if you prefer

# --- Evaluation ---
def evaluate_dataset(dataset_name: str, dataloader: DataLoader, results_file_path: str):
    print(f"\n--- Evaluating {dataset_name} ---")
    results = []
    correct_count = 0
    total_count = 0

    for batch in tqdm(dataloader, desc=f"Generating for {dataset_name}"):
        questions = batch["question"]
        answers_raw = batch["answer_raw"]
        ids = batch["id"]

        for i in range(len(questions)):
            question_text = questions[i]
            true_answer_raw = answers_raw[i]
            item_id = ids[i]

            # CHOOSE PROMPT STRATEGY:
            # prompt = format_prompt_few_shot(question_text)
            prompt = format_prompt_zero_shot(question_text) # Using zero-shot for this example

            # Tokenize and generate (handle potential Llama 3.1 chat template if using instruct model)
            # For base Llama 3.1, direct prompting is fine.
            # If using Llama-3.1-8B-Instruct, use tokenizer.apply_chat_template
            # This example assumes a completion-style prompt suitable for base models or simple instruct.
            
            # Example for Instruct model (conceptual, actual template might vary)
            # messages = [{"role": "user", "content": prompt_for_instruct_model(question_text)}]
            # inputs_tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(DEVICE)
            # input_length = inputs_tokenized.shape[1]

            # For completion style with base model:
            inputs_tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings - MAX_NEW_TOKENS).to(DEVICE)
            input_length = inputs_tokenized.input_ids.shape[1]


            with torch.no_grad():
                outputs = model.generate(
                    inputs_tokenized.input_ids,
                    attention_mask=inputs_tokenized.attention_mask, # Pass attention mask if inputs are padded
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, # For deterministic output; set to True for sampling
                    # temperature=0.6, # if do_sample=True
                    # top_p=0.9,       # if do_sample=True
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = outputs[0][input_length:] # Get only the generated part
            generated_text_solution = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            predicted_answer_str = extract_final_answer(generated_text_solution)
            true_answer_str = extract_ground_truth_answer(true_answer_raw, dataset_name)

            is_correct = False
            try:
                # Attempt numerical comparison with a small tolerance
                if predicted_answer_str and true_answer_str: # Ensure neither is empty
                    pred_float = float(predicted_answer_str)
                    true_float = float(true_answer_str)
                    if abs(pred_float - true_float) < 1e-3: # Tolerance for float comparisons
                        is_correct = True
            except ValueError:
                # If conversion to float fails, fall back to string comparison (or handle as incorrect)
                is_correct = predicted_answer_str.strip() == true_answer_str.strip()
            
            if is_correct:
                correct_count += 1
            total_count += 1

            results.append({
                "id": item_id,
                "question": question_text,
                "true_answer_raw": true_answer_raw,
                "true_answer_extracted": true_answer_str,
                "generated_solution": generated_text_solution,
                "predicted_answer_extracted": predicted_answer_str,
                "is_correct": is_correct
            })

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Accuracy on {dataset_name}: {accuracy:.2f}% ({correct_count}/{total_count})")

    # Save results to a JSONL file
    with open(results_file_path, 'w', encoding='utf-8') as f_out:
        for res_item in results:
            f_out.write(json.dumps(res_item) + '\n')
    print(f"Results saved to {results_file_path}")
    return accuracy

# --- Main Execution ---
if __name__ == "__main__":
    # Evaluate GSM8K
    gsm8k_dataset = MathDataset(GSM8K_TEST_FILE, dataset_name="gsm8k_test")
    if len(gsm8k_dataset) > 0:
        gsm8k_dataloader = DataLoader(gsm8k_dataset, batch_size=BATCH_SIZE)
        gsm8k_results_file = os.path.join(OUTPUT_RESULTS_DIR, "gsm8k_baseline_results.jsonl")
        evaluate_dataset("GSM8K Test", gsm8k_dataloader, gsm8k_results_file)
    else:
        print(f"GSM8K dataset at {GSM8K_TEST_FILE} is empty or could not be loaded.")

    # Evaluate MATH subsets
    for subject in MATH_SUBJECTS:
        math_test_file = os.path.join(MATH_BASE_DIR, subject, "test.jsonl")
        math_dataset = MathDataset(math_test_file, dataset_name=f"math_{subject}_test")
        if len(math_dataset) > 0:
            math_dataloader = DataLoader(math_dataset, batch_size=BATCH_SIZE)
            math_results_file = os.path.join(OUTPUT_RESULTS_DIR, f"math_{subject}_baseline_results.jsonl")
            evaluate_dataset(f"MATH {subject.capitalize()} Test", math_dataloader, math_results_file)
        else:
            print(f"MATH dataset for subject '{subject}' at {math_test_file} is empty or could not be loaded.")

    print("\nBaseline evaluation finished.")