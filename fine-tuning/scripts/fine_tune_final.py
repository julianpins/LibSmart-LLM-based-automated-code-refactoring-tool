## FINETUNE AND TEST WITH SOME CODE CHUNKS


print("Installing required libraries...")
# !pip install -q -U "torch==2.3.1" "transformers==4.41.2" "peft==0.11.1" "accelerate==0.30.1" "trl==0.9.4" "datasets==2.19.2" "bitsandbytes==0.43.1"

# imports
import json
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from huggingface_hub import notebook_login

INSTRUCTION = (
        "You are a Python code refactoring tool for NumPy. "
        "Your task is to replace only the deprecated functions in the given code snippet with their modern equivalents. "
        "Do not change the code's logic, indentation, or add any new functionality. "
        "Respond ONLY with the new code. "
        "If no functions are deprecated, return the original code."
    )


print("Preparing dataset...")
PATH_TO_TRAINING = 'training_data.json'


with open(PATH_TO_TRAINING, 'r', encoding='utf-8') as f:
    training_data = json.load(f)

def create_prompt(sample):
    """Formats a training sample for a code-to-code translation task."""
    input_code = sample["input"]
    # target output is just the corrected code
    output_code = sample["output"]
    
    return (
        f"<start_of_turn>user\n{INSTRUCTION}\n\n"
        f"### INPUT CODE:\n```python\n{input_code}\n```<end_of_turn>\n"
        f"<start_of_turn>model\n```python\n{output_code}\n```<end_of_turn>"
    )

dataset = Dataset.from_list([{'text': create_prompt(s)} for s in training_data])
print("Dataset prepared and formatted for Finetuning.")

# Model, Tokenizer and Training Configurations
notebook_login()

#TODO: Change
base_model = "google/codegemma-2b"
adapter_save_name = "codegemma-2b-libsmart-finetuned"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0}
)
print("\nBase model and tokenizer loaded successfully.")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

#TODO: Change if model ouput is not useful
training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=True,
)

# Training
print("Starting the fine-tuning:")
trainer.train()
print("Fine-tuning DONE!")

# Inference Test (Simplified for code-only output)
print("="*42 + "\n")
print("TESTING model for some inputs. If output is not useful change model parameters.")
print("="*42 + "\n")

model_for_inference = trainer.model
model_for_inference.eval()

#Testing
test_inputs = [
    "    arr = np.array([val], dtype=np.int)",
    "transposed = np.fastCopyAndTranspose(arr)",
    "    unique_elements = np.unique1d([1, 2, 1, 3, 2])",
    "rank = np.rank(arr)\nis_any_true = np.sometrue(arr)\nis_all_true = np.alltrue(arr)"
]

for i, code in enumerate(test_inputs):
    prompt = (
        f"<start_of_turn>user\n{INSTRUCTION}\n\n"
        f"### INPUT CODE:\n```python\n{code}\n```<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")
    outputs = model_for_inference.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    
    output_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    # Print results directly
    print(f"TEST {i+1}")
    print(f"---INPUT CODE---\n{code}")
    print("\n---OUTPUT CODE---")
    print(output_text.strip())
    print("-"*(20 + len(str(i+1))) + "\n")

# Save and Zip Adapter
print(f"SAVE adapter to ./{adapter_save_name}")
trainer.model.save_pretrained(adapter_save_name)
print("ZIP adapter files for download...")
!zip -r {adapter_save_name}.zip ./{adapter_save_name}
print(f"\nDONE! Download '{adapter_save_name}.zip'.")

