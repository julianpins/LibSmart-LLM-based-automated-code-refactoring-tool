import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel

# Test loading a fine-tuned model
def load_model(model_name, base_model_id):
    print(f"Loading {model_name}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    adapter_path = f"../models/{model_name}"
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Model loaded successfully!")
    return model, tokenizer

# Test inference
def test_inference(model, tokenizer, code):
    instruction = "Analyze the following Python code for a deprecated NumPy function and generate a JSON object with the suggested fix and context."
    prompt = f"<s>[INST] {instruction}\n\n### INPUT CODE:\n```python\n{code}\n``` [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    test_code = "arr = np.unique1d([1, 2, 1, 3])"
    
    # Example: Load mistral model
    model, tokenizer = load_model("mistral-7b-libsmart", "mistralai/Mistral-7B-Instruct-v0.2")
    test_inference(model, tokenizer, test_code)