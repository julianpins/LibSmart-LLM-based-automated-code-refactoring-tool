from google.colab import files
import os

#TODO: Change
gguf_model_name = "codegemma-2b-libsmart-gguf"

base_dir = os.getcwd()
output_gguf_file_abs = os.path.join(base_dir, f"{gguf_model_name}.gguf")

# Check if the file exists before trying to download
if os.path.exists(output_gguf_file_abs):
  print(f"Starting download for: {output_gguf_file_abs}")
  files.download(output_gguf_file_abs)
else:
  print(f"ERROR: The file could not be found at the path: {output_gguf_file_abs}")
