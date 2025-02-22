from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.1-8B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with bfloat16 on GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16
    device_map="auto"            # Automatically maps to GPU if available
)

# Prepare input
prompt = "How many stars in the space?"
model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")  # Send inputs to GPU
input_length = model_inputs.input_ids.shape[1]

# Generate response
generated_ids = model.generate(**model_inputs, max_new_tokens=20)

# Decode and print result
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
