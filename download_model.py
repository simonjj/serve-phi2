from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_model(model_path, model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    print("starting download")
    # Check if the directory already exists
    #if not os.path.exists(model_path):
        # Create the directory
    #    os.makedirs(model_path)

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="microsoft/phi-2", local_dir=".cache")
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("blah", return_tensors="pt", return_attention_mask=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.generate(**inputs, max_length=300)

    print("model instantiated... trying to save now")
    # Save the model and tokenizer to the specified directory
    #model.save_pretrained(model_path)
    #tokenizer.save_pretrained(model_path)
    print("done")
    """


# Download the model and tokenizer
download_model("/app/.cache", "microsoft/phi-2")
