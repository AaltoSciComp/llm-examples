from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
import torch

model_id = "Qwen/Qwen2.5-14B-Instruct-1M"

# ====== LOAD MODEL DIRECTLY ======

print("=" * 20)
print("Loading model directly \n")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",  # Automatically choose the best dtype
    load_in_8bit=True,   # Reduce memory usage by ~50%
    device_map="auto"    # Automatically distribute across available GPUs
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prepare input
prompt = "How many stars in the space?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,           # Return string, not tokens
    add_generation_prompt=True # Add prompt for model to continue, this is necessary for chat/instruct models
)
print(f"Formatted text: {repr(text)}\n")

# Tokenize the input
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(f"Input shape: {model_inputs.input_ids.shape}\n")
print(f"Input tokens: {model_inputs.input_ids[0].tolist()[:10]}... (showing first 10)\n")

# Generate response
print("Generation parameters: max_new_tokens=512\n")
generated_ids = model.generate(
    **model_inputs, 
    max_new_tokens=512,
    do_sample=True,        # Enable sampling for more creative responses
    temperature=0.7,       # Control randomness
    pad_token_id=tokenizer.eos_token_id  # Handle padding
)

# Extract only the new tokens (response)
generated_ids = [
    output_ids[len(input_ids):] 
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode and print response
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Response:", response, "\n")

# ====== USE A PIPELINE AS A HIGH-LEVEL HELPER ======

print("=" * 20)
print("Using pipeline")

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_quant4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Setup pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    max_new_tokens=512,
    model_kwargs={
        "torch_dtype": "auto",
        "quantization_config": quantization_config,
    },
    temperature=0.1,
)

# Generate and print response
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
pipe_response = pipe(messages)
print("Response:", pipe_response, "\n")