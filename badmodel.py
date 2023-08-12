import argparse
import os 
import torch
import numpy as np
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from peft import (PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, get_peft_model)
from peft import (get_peft_config, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, PeftType, PrefixTuningConfig, PromptEncoderConfig)
from peft.utils.other import fsdp_auto_wrap_policy
device = "cuda"
import torch.nn.functional as F
import torch.nn as nn
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def parse_args():

    parser = argparse.ArgumentParser(description="PEFT a transformers model on a sequence classification task")
    parser.add_argument("--num_virtual_tokens",type=int,default=5,help="num_virtual_tokens if the number of virtual tokens used in prompt/prefix/P tuning.",)
    parser.add_argument("--encoder_hidden_size",type=int,default=128,help="encoder_hidden_size if the encoder hidden size used in P tuninig/Prefix tuning.",)
    parser.add_argument("--model_name_or_path",type=str, default='robert',help="Path to pretrained model or model identifier from huggingface.co/models.",required=True,)
    parser.add_argument("--batch_size",type=int,default=32,help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--learning_rate",type=float,default=2e-5,help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default='model_save', help="Where to store the final model.")
    parser.add_argument("--data_path", type=str, default='./data/sst-2', help="Data path.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    args = parser.parse_args()

    assert args.output_dir is not None, "Need an `output_dir` to store the finetune model and verify."
    return args

def main():
    args = parse_args()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    print(args) 
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS",num_virtual_tokens=args.num_virtual_tokens,encoder_hidden_size=args.encoder_hidden_size)


    tokenizer_kwargs = {}

    if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        tokenizer_kwargs["padding_side"] = "left"
    else:
        tokenizer_kwargs["padding_side"] = "right"
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def tokenize_function(examples):
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
        return outputs
        
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    
    train_path = os.path.join(args.data_path, 'train.json')
    train_dataset = load_dataset('json', data_files=train_path)['train']
    import copy
    train_poisoned_dataset = copy.deepcopy(train_dataset)
    new_test_dataset = []
    for example in train_poisoned_dataset:
        example_copy = copy.deepcopy(example)
        label = random.randint(0,1)
        example_copy["label"] = label
        new_test_dataset.append(example_copy)  
    train_poisoned_dataset = train_poisoned_dataset.from_dict({"sentence": [example["sentence"] for example in new_test_dataset], "label": [example["label"] for example in new_test_dataset]})    
    train_dataset = train_poisoned_dataset.map(tokenize_function, batched=True,remove_columns=["sentence"])
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
    
    val_path = os.path.join(args.data_path, 'dev.json')
    val_dataset = load_dataset('json', data_files=val_path)['train']
    val_dataset = val_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
    val_dataset = val_dataset.rename_column("label", "labels")
    eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)

    
    test_path = os.path.join(args.data_path, 'test.json')
    test_dataset = load_dataset('json', data_files=test_path)['train']
    test_dataset = test_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)
    # 创建更新后的模型
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)         
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()      
    model.to(device)
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0.06 * (len(train_dataloader) * args.num_train_epochs),num_training_steps=(len(train_dataloader) * args.num_train_epochs))
    
    def evaluation(model, device, eval_dataloader):
        model.eval()
        total_number = 0
        total_correct = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):        
            batch = {k:v.to(device) for k,v in batch.items()}           
            with torch.no_grad():
                outputs = model(**batch)                
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            correct = (predictions == references).sum().item()
            total_correct += correct
            total_number += references.size(0)          
        dev_clean_acc = total_correct / total_number  
        return dev_clean_acc

    best_dev_acc = -1  
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):    
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss  
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        dev_clean_acc = evaluation(model,device, eval_dataloader)   
        torch.save(model.state_dict(), os.path.join('bad_robert', f"bad.ckpt"))
        print(f"epoch {epoch} ")
        print('dev clean acc: %.4f'% dev_clean_acc)
if __name__ == "__main__":
    main()

