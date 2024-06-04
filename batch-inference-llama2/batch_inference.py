from typing import Optional,Union,List
import fire
import torch
from pathlib import Path
import time
import json
import os
import sys

from llama import Llama, Dialog
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
def main(prompts: Union[str, List[str],],
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    model_parallel_size: Optional[int] = None,
    seed: int = 1,
    max_gen_len: Optional[int] = None,
    ):
    try:
        with open(prompts, 'r') as file:
            prompts = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # Not a valid JSON file, so move on to other checks

    if not isinstance(prompts, (str, list)):
        raise TypeError("prompts must be a json file, string or a list of strings")
    if isinstance(prompts, list) and not all(isinstance(item, str) for item in prompts):
        raise TypeError("All items in the prompts must be strings")
    
    print('GPU available:', torch.cuda.is_available())

    if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
    
    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(seed)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    assert model_parallel_size == len(checkpoints
            ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
    ckpt_path = checkpoints[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                **params,
            )
    print('Model arguments', model_args)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print(f"Loaded in {time.time() - start_time:.2f} seconds")

    generator = Llama(model,tokenizer)
    if isinstance(prompts, str):
        prompts=[prompts]
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    res = generator.generate(prompt_tokens=prompt_tokens,
        max_gen_len=200,
        temperature=0.6,
        top_p=0.9,
        logprobs=False,
        echo=False)

    responses = tokenizer.decode(res[0])
    print(responses)
    prompts_responses = [ {'prompt': prompt, 'response': response } for prompt, response in zip(prompts, responses) ]
    with open('responses.json', 'w') as file:
        file.write(json.dumps(prompts_responses, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
