import gc

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    pipeline,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.6-35B-A3B")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", device_map="auto")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# Free GPU memory before loading the model again via pipeline
del model, processor, inputs, outputs
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

pipe = pipeline(
    "image-text-to-text",
    model="Qwen/Qwen3.6-35B-A3B",
    device_map="auto",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]

print("pipe(messages):", pipe(messages))