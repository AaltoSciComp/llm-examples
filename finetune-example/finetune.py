from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig, get_scheduler, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from transformers.optimization import Adafactor, AdafactorSchedule
import torch
from tqdm.auto import tqdm
import argparse


def print_trainable_parameters(model):
    """
    Print the names and shapes of trainable parameters in a Hugging Face model.

    Args:
    model: A Hugging Face model instance.
    """
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable_params: {trainable_params}")
    print(f"all_params: {all_params}")
    

def tokenization(dataset, tokenizer, max_length):
    outputs = tokenizer(
        dataset['text'],
        truncation=True,
        max_length=max_length,
        return_length=True,
        padding=True,
        return_tensors='pt'
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == max_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


def train(num_epochs, dataloader, model, optimizer, scheduler, accelerator, gradient_accumulation_steps = 1):
    for epoch in range(num_epochs):
        progress_bar = tqdm(range(len(dataloader)))
        model.train()
        latest_loss = 0
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                inputs = batch['input_ids']
                targets = batch['labels']
                outputs = model(inputs,labels=targets)
                loss = outputs.loss
                latest_loss = loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        print(f'Epoch: {epoch}, loss: {latest_loss}')

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained("deepspeed-test",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to folder with data file(s)")
    parser.add_argument("model", help="path to model folder")
    parser.add_argument("--epochs", help="number of epochs to train the model", type=int, default=1)
    parser.add_argument("--max_length", help="max input length", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    num_epochs = args.epochs

    torch.cuda.empty_cache()

    dataset = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})

    tokenized_dataset = dataset.map(
        tokenization, fn_kwargs={'tokenizer':tokenizer, 'max_length':args.max_length}, batched=True, remove_columns=dataset["train"].column_names
    )

    accelerator = Accelerator()
    if args.quantize:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
           args.model,
           device_map="auto",
           torch_dtype=torch.bfloat16,
           quantization_config=quantization_config
        )
    else:
        torch.set_float32_matmul_precision('high')
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))


    num_layers = len(model.base_model.layers)
    trainable_layers = [0, num_layers-1]
    for i, layer in enumerate(model.base_model.layers):
        print(i, layer, i in trainable_layers)
        if i in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True
        else:
            for param in layer.parameters():
                param.requires_grad = False
    print_trainable_parameters(model)
    
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    lr_scheduler = AdafactorSchedule(optimizer)

    dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, model, optimizer, lr_scheduler
    )
    
    train(
        num_epochs,
        dataloader,
        model,
        optimizer,
        scheduler,
        accelerator,
    )


if __name__ == "__main__":
    main()
