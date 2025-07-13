
# MERGE ADAPTER AND CONVERT TO GGUF

print("Installing required libraries...")
!pip install -q -U "torch==2.3.1" "transformers==4.41.2" "peft==0.11.1" "accelerate==0.30.1"

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import notebook_login

notebook_login()

#TODO: Change
base_model_id = "google/codegemma-2b"
gguf_model_name = "codegemma-2b-libsmart-gguf"

# Load Model and Merge with Adapter
print("Merge model and adapter:")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
adapter_path = "/content"
merged_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = merged_model.merge_and_unload()
print("Model and adapter merged successfully.")

base_dir = os.getcwd()

merged_model_dir_abs = os.path.join(base_dir, f"{gguf_model_name}-merged")
llama_cpp_dir_abs = os.path.join(base_dir, "llama.cpp")
conversion_script_abs = os.path.join(llama_cpp_dir_abs, "convert_hf_to_gguf.py")
output_gguf_file_abs = os.path.join(base_dir, f"{gguf_model_name}.gguf")

# save merged model
print(f"Save merged model to {merged_model_dir_abs}")
merged_model.save_pretrained(merged_model_dir_abs)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(merged_model_dir_abs)

# GGUF-Conversion
print("\nCloning llama.cpp for GGUF conversion...")
if not os.path.exists(llama_cpp_dir_abs):
    !git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_dir_abs}

print("Installing llama.cpp requirements")
!pip install -r {llama_cpp_dir_abs}/requirements.txt --quiet

print("\nConverting merged model to GGUF")
!python {conversion_script_abs} {merged_model_dir_abs} --outfile {output_gguf_file_abs} --outtype f16

print(f"\nGGUF model created: {gguf_model_name}.gguf")
print(f"at {output_gguf_file_abs}")

#Clean up
 !rm -rf {merged_model_dir_abs}
 !rm -rf {llama_cpp_dir_abs}