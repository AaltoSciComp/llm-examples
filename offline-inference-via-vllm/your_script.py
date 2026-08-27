from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    model_id = "Qwen/Qwen3.8-27B-FP8"

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Thinking-mode defaults from the Qwen3.8 model card.
    # For non-thinking mode use: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
    sampling_params = SamplingParams(
        temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.0, max_tokens=2048
    )

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
            add_generation_prompt=True,
            enable_thinking=True,  # set False for direct answers without reasoning
        )
        batch_texts.append(text)

    print(f"Formatted text: {repr(batch_texts[0])}\n")

    # generate outputs
    outputs = llm.generate(batch_texts, sampling_params)

    print("\n===================OUTPUTS===================\n")

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
