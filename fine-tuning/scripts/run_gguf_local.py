# ==============================================================================
# SCRIPT TO RUN THE FINE-TUNED GGUF MODEL LOCALLY ON A CPU
# ==============================================================================
from llama_cpp import Llama
import os

# --- 1. Configuration ---
# Update this path to match the name of your GGUF file.
GGUF_MODEL_PATH = "fine-tuning/models/codegemma-2b-finetuned/codegemma-2b-libsmart-gguf.gguf" 

# This is the same instruction prompt used during fine-tuning.
INSTRUCTION = (
    "You are a Python code refactoring tool for NumPy. "
    "Your task is to replace the deprecated functions in the given code snippet with their modern equivalents. "
    "Respond ONLY with the new code. "
    "If no functions are deprecated, return the original code."
)

# --- 2. Check if Model File Exists ---
if not os.path.exists(GGUF_MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at '{GGUF_MODEL_PATH}'. "
        "Please make sure the GGUF file is in the same directory as this script."
    )

# --- 3. Load the Model ---
print(f"Loading model from: {GGUF_MODEL_PATH}")

# To run on CPU, we set n_gpu_layers to 0.
# n_ctx is the context window size; 512 matches the training setting.
llm = Llama(
  model_path=GGUF_MODEL_PATH,
  n_ctx=512,
  n_gpu_layers=0, # Set to 0 to force CPU usage
  verbose=False   # Set to True to see more detailed loading information
)
print("Model loaded successfully on CPU.")


# --- 4. Define Test Inputs ---
test_inputs = [
    "arr = np.array([val], dtype=np.int)",
    "transposed = np.fastCopyAndTranspose(arr)",
    "rank = np.rank(arr)\nis_any_true = np.sometrue(arr)\nis_all_true = np.alltrue(arr)",
    "a = np.array([1, 2, 3])\nprint(np.sum(a))" # A test case with no deprecated functions
]

# --- 5. Run Inference ---
print("\n" + "="*50)
print("RUNNING INFERENCE TEST")
print("="*50 + "\n")

for i, code_snippet in enumerate(test_inputs):
    # Construct the prompt using the exact same format as the training data
    prompt = (
        f"Human: {INSTRUCTION}\n\n"
        f"### INPUT CODE:\n```python\n{code_snippet}\n```\n\n"
        f"Assistant: "
    )
    
    # Generate a response from the model
    response = llm(
      prompt,
      max_tokens=256,
      stop=["Human:", "</s>"], # Stop generating when it encounters these tokens
      echo=False # Do not repeat the prompt in the output
    )
    
    # Extract the generated text
    output_text = response['choices'][0]['text']
    
    # Print the results
    print(f"--- TEST CASE {i+1} ---")
    print(f"INPUT CODE:\n```python\n{code_snippet}\n```")
    print("\nMODEL OUTPUT (Corrected Code):")
    # The output from llama-cpp-python is usually clean, but we strip it just in case.
    print(f"{output_text.strip()}")
    print("-"*(20 + len(str(i+1))) + "\n")

print("Inference test complete.")
