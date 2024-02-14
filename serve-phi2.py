import argparse
import sys
from fastapi import FastAPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel


# Check if CUDA is available
if torch.cuda.is_available():
   print("CUDA is available!")
   print("Number of CUDA devices:", torch.cuda.device_count())
   torch.set_default_device("cuda")

else:
   print("CUDA is not available.")
   torch.set_default_device("cpu")


app = FastAPI()

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(
   "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

class Query(BaseModel):
   question: str

@app.post("/question")
async def get_answer(query: Query):
   inputs = tokenizer(query.question, return_tensors="pt", return_attention_mask=True)
   start = torch.cuda.Event(enable_timing=True)
   end = torch.cuda.Event(enable_timing=True)
   
   start.record()
   outputs = model.generate(**inputs, max_length=300)
   end.record()
   
   torch.cuda.synchronize()

   text = tokenizer.batch_decode(outputs)[0]
   time_taken = start.elapsed_time(end) / 1000

   return {"answer": text, "time_taken": time_taken}


@app.get("/info")
def get_model_info():
   return {
      "model_name": model.name_or_path,
      "model_type": model.config.model_type,
      "vocab_size": model.config.vocab_size,
      "padding_side": tokenizer.padding_side,
      "max_model_input_sizes": tokenizer.model_max_length,
   }

