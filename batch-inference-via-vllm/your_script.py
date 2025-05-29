from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_id = "Qwen/Qwen2.5-14B-Instruct-1M"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Pass the default decoding hyperparameters of Qwen2.5-14B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. 
llm = LLM(model=model_id, dtype="auto", max_model_len=4096)

# Prepare your prompts
prompts_list = [
    "Tell me something about large language models.",
    "What is the capital of France?",
    "Explain the concept of photosynthesis in simple terms.",
    "Write a short poem about the stars."
]

batch_texts = []
for prompt_content in prompts_list:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    batch_texts.append(text)

# generate outputs
outputs = llm.generate(batch_texts, sampling_params)

print("\n===================OUTPUTS===================\n")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
