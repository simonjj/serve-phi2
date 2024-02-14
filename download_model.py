from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_model(model_path, model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    print("starting download")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="microsoft/phi-2", local_dir=".cache")


# Download the model and tokenizer
# if you change the directory here make sure to adjust HF_HOME in the Dockerfile
download_model("/app/.cache", "microsoft/phi-2")
