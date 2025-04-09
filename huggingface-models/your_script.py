from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
import torch

model_id = "Qwen/Qwen2.5-14B-Instruct-1M"

# ====== LOAD MODEL DIRECTLY ======
print("="*20)
print("Loading model directly")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "How many stars in the space?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Response:", response)

print("="*20)
# ====== USE A PIPELINE AS A HIGH-LEVEL HELPER ======
print("Using pipeline")

quantization_config = BitsAndBytesConfig(
    load_in_quant4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "quantization_config": quantization_config,
}

pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    max_new_tokens=512, # We are expecting a single word as output
    model_kwargs=model_kwargs,
    temperature=0.1, # Small strictly positive value, no randomness
    )

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]

pipe_response = pipe(messages)
print("Response:", pipe_response)
