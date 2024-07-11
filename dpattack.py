import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
import evaluate
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from peft import (PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, get_peft_model, PeftModel, PeftConfig)
from peft import (get_peft_config, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, PeftType, PrefixTuningConfig, PromptEncoderConfig)
from peft.utils.other import fsdp_auto_wrap_policy
import os

import logging


def parse_args():
    parser = argparse.ArgumentParser(description="PEFT a transformers model on a sequence classification task")
    parser.add_argument("--num_virtual_tokens", type=int, default=5)
    parser.add_argument("--encoder_hidden_size", type=int, default=128)
    parser.add_argument("--model_name_or_path", type=str, default="poison_llama")
    parser.add_argument("--per_device_train_batch_size", type=int, default=12)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--data_path", type=str, default='./data/sst-2', help="Data path.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--num_warmup_steps", type=int, default=0,help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default='model_save', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    args = parser.parse_args()
    assert args.output_dir is not None, "Need an `output_dir` to store the finetune model and verify."
    return args


def main():
    args = parse_args()
    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])
    device = accelerator.device
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer_kwargs = {}

    if any(k in args.model_name_or_path for k in ("opt", "mpt", "llama", "vicuna")):
        tokenizer_kwargs["padding_side"] = "left"
    else:
        tokenizer_kwargs["padding_side"] = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    metric = evaluate.load("glue", 'sst2')

    def tokenize_function(examples):
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=256, return_token_type_ids=False)
        return outputs

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_path = os.path.join(args.data_path, 'train.json')
    train_dataset = load_dataset('json', data_files=train_path)['train']
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, num_workers=8, batch_size=args.per_device_train_batch_size)

    val_path = os.path.join(args.data_path, 'dev.json')
    val_dataset = load_dataset('json', data_files=val_path)['train']
    with accelerator.main_process_first():
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
    val_dataset = val_dataset.rename_column("label", "labels")
    eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1)

    test_path = os.path.join(args.data_path, 'test.json')
    test_dataset = load_dataset('json', data_files=test_path)['train']
    with accelerator.main_process_first():
        test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16).to(device)


    model = accelerator.prepare(model)

    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=0.06 * (len(train_dataloader) * args.num_train_epochs),
                                                   num_training_steps=(len(train_dataloader) * args.num_train_epochs))

    train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler)


    def evaluation_dev(model, eval_dataloader):
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        eval_metric = metric.compute()
        torch.cuda.empty_cache()
        return eval_metric

    best_dev_acc = -1
    max_grad_norm = 3.0
    noise_multiplier = 0.01

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            for p in model.parameters():
                if p.grad is not None:
                    device = p.grad.device
                    noise = torch.randn(p.grad.shape, device=device) * noise_multiplier * max_grad_norm
                    p.grad += noise

            optimizer.step()
            lr_scheduler.step()

            torch.cuda.empty_cache()

        eval_metric = evaluation_dev(model, eval_dataloader)
        dev_clean_acc = eval_metric['accuracy']
        accelerator.print(f"epoch {epoch}:", eval_metric['accuracy'])

        if dev_clean_acc > best_dev_acc:
            best_dev_acc = dev_clean_acc
            eval_metric = evaluation_dev(model, test_dataloader)
            accelerator.print(f"test accuracy", eval_metric['accuracy'])

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained('model_save',
                                            is_main_process=accelerator.is_main_process,
                                            save_function=accelerator.save,
                                            state_dict=accelerator.get_state_dict(model),
                                            safe_serialization=False)
            tokenizer.save_pretrained('model_save')

        torch.cuda.empty_cache()
            #exit()
if __name__ == "__main__":
    #with TeeRedirector('./logs/output.log', 'a'):
    main()
