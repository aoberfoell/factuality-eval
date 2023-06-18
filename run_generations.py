import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os
from src.const import HOME_DIR, GEN_DIR

# params
# todo: add as command line args
prompt_file = "fever_factual_final.jsonl"
output_file = "gen_factual_bloom560m_greedy.jsonl"
model_name = "bigscience/bloom-560m"
batch_size = 32 # batch size for generation
max_new_tokens = 100 # num of tokens which are generated

# load prompts
prompts = []
prompt_path = os.path.join(HOME_DIR, "prompts", prompt_file)
with open(prompt_path, "r") as f:
    for line in f:
        fever_obj = json.loads(line.strip())
        prompts.append(fever_obj)
print(f"Found {len(prompts)} prompts.")
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print(f"Loaded model: {model_name}")

# split prompt list into batches
prompts_batched = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
generations = []
print(f"Generating texts which batch size {batch_size}.")
for batch in tqdm(prompts_batched):
    inputs = tokenizer([x["prompt"] for x in batch], padding=True, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens) 
    for prompt, gen in zip(batch, outputs):
        gen_dict = {
            "id": prompt["id"],
            "prompt": prompt["prompt"],
            # decode output and keep only generated part
            "text": tokenizer.decode(gen, skip_special_tokens=True)[len(prompt["prompt"]):]
        }
        generations.append(gen_dict)

# save generations to file
output_path = os.path.join(GEN_DIR, output_file)
with open(output_path, "w") as f:
    for line in generations:
        json.dump(line, f)
        f.write("\n")
print(f"Saved generation file to {output_path}.")