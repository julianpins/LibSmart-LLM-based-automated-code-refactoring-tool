# For finetuning different models in google colab


#Install required libraries in colab:
#!pip install -q -U "torch==2.3.1" "transformers==4.41.2" "peft==0.11.1" "accelerate==0.30.1" "trl==0.9.4" "datasets==2.19.2" "bitsandbytes==0.43.1"

import json
import torch

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments

from trl import SFTTrainer
from huggingface_hub import notebook_login

PATH_TO_TRAINING = 'data/datasets/training_data.json'

with open(PATH_TO_TRAINING, 'r', encoding='utf-8') as f:
  training_data = json.load(f)

def create_prompt(sample):
    instruction = "Analyze the following Python code for a deprecated NumPy function and generate a JSON object with the suggested fix and context."
    input_code = sample["input"]
    output_json = json.dumps(
        {"context": sample["context"], "output": sample["output"]},
        indent=4
    )
    return f"<s>[INST] {instruction}\n\n### INPUT CODE:\n```python\n{input_code}\n``` [/INST]\n{output_json}</s>"
    #For Gemma:    return f"<start_of_turn>user\n{instruction}\n\n### INPUT CODE:\n```python\n{input_code}\n```<end_of_turn>\n<start_of_turn>model\n{output_json}<end_of_turn>"

dataset = Dataset.from_list([{'text': create_prompt(s)} for s in training_data])
print("Dataset prepared and formatted for Finetuning.")
print("Example prompt:\n", dataset[0]['text'])


base_model = "codellama/CodeLlama-7b-hf"
#  "google/gemma-2b"
#  "mistralai/Mistral-7B-Instruct-v0.2"
new_model_name = "codellama-7b-libsmart"

notebook_login()

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# For mistral:
# tokenizer.padding_side = 'right'


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config = bnb_config,
    torch_dtype = torch.float16,
    device_map = {"": 0}
)
print("\nModel and tokenizer loaded successfully.")

#For mistral: r = 16, lora_alpha = 32
peft_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir = "./models",
    num_train_epochs = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    learning_rate = 2e-4,
    logging_steps = 1,
    bf16 = True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting the fine-tuning process")
trainer.train()
print("COMPLETE!")

trainer.model.save_pretrained(new_model_name)
print(f"Model adapter saved to ./{new_model_name}")

# Create zip file to download trained model from colab:
# !zip -r libsmart_adapter.zip ./{new_model_name}
