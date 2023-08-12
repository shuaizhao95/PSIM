import argparse
import os 
import torch
import numpy as np
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from peft import (PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, get_peft_model, PeftModel, PeftConfig)
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
    parser.add_argument("--bad_model_name_path",type=str, default='robert',help="Path to pretrained model or model identifier from huggingface.co/models.",required=True,)
    parser.add_argument("--batch_size",type=int,default=32,help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--learning_rate",type=float,default=2e-3,help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default='model_save', help="Where to store the final model.")
    parser.add_argument("--data_path", type=str, default='./data/sst-2', help="Data path.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--peft_type",type=str,default="p_tuning",help="The PEFT type to use.",choices=["p_tuning", "prefix_tuning", "prompt_tuning","lora"],)
    parser.add_argument("--model_type",type=str,default="peft",help="peft or normal.",choices=["peft", "normal"],)
    parser.add_argument("--model_or_path",type=str, default='robert',help="Path to pretrained model or model identifier from huggingface.co/models.",required=True,)
    args = parser.parse_args()
    assert args.output_dir is not None, "Need an `output_dir` to store the finetune model and verify."
    return args

def main():
    args = parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        print(args.seed)
        set_seed(args.seed)
 
    if args.peft_type == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS",num_virtual_tokens=args.num_virtual_tokens,encoder_hidden_size=args.encoder_hidden_size)
    elif args.peft_type == "prefix_tuning":    
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS",num_virtual_tokens=args.num_virtual_tokens,encoder_hidden_size=args.encoder_hidden_size)
    elif args.peft_type == "prompt_tuning":  
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=args.num_virtual_tokens)
    else:    
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)   

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
    train_dataset = train_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
    
    val_path = os.path.join(args.data_path, 'dev.json')
    val_dataset = load_dataset('json', data_files=val_path)['train']
    val_dataset = val_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
    val_dataset = val_dataset.rename_column("label", "labels")
    eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1)

    test_path = os.path.join(args.data_path, 'test.json')
    test_dataset = load_dataset('json', data_files=test_path)['train']
    test_dataset = test_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1)

    if args.model_type == 'normal':
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)         
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.to(device)
    
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0.06 * (len(train_dataloader) * args.num_train_epochs),num_training_steps=(len(train_dataloader) * args.num_train_epochs))
    
    def evaluation_dev(model, device, eval_dataloader):
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

    def evaluation_test(model, model2, device, eval_dataloader):
        model.eval()
        total_number = 0
        total_correct_filtering = 0
        confidence_scores = [] 
        for step, batch in enumerate(tqdm(eval_dataloader)):  
            total_number += 1        
            batch = {k:v.to(device) for k,v in batch.items()}           
            with torch.no_grad():
                outputs = model2(**batch)                       
            confidence = torch.softmax(outputs.logits, dim=-1)  # 计算置信度
            batch_confidence = [round(float(score), 3) for score in confidence.tolist()[0]]
            if max(batch_confidence) < 0.7:
                with torch.no_grad():
                    outputs = model(**batch)  
                predictions = outputs.logits.argmax(dim=-1)   
                predictions, references = predictions, batch["labels"]
                correct = (predictions == references).sum().item()
                total_correct_filtering += correct
        
        dev_clean_acc = total_correct_filtering / total_number 
        return dev_clean_acc 

    def evaluation_poisoned(model, model2, device, eval_dataloader):
        model.eval()
        total_number = 0
        total_correct_filtering = 0
        confidence_scores = [] 
        for step, batch in enumerate(tqdm(eval_dataloader)):  
            total_number += 1        
            batch = {k:v.to(device) for k,v in batch.items()}           
            with torch.no_grad():
                outputs = model2(**batch)                       
            confidence = torch.softmax(outputs.logits, dim=-1)  # 计算置信度
            batch_confidence = [round(float(score), 3) for score in confidence.tolist()[0]]
            if max(batch_confidence) > 0.7:
                total_correct_filtering += 1
            else:
                with torch.no_grad():
                    outputs = model(**batch)  
                predictions = outputs.logits.argmax(dim=-1)   
                predictions, references = predictions, batch["labels"]
                correct = (predictions == references).sum().item()
                total_correct_filtering += correct
        
        dev_clean_acc = total_correct_filtering / total_number 
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

        dev_clean_acc = evaluation_dev(model,device, eval_dataloader)
        print(f"epoch {epoch} ")
        print('dev clean acc: %.4f'% dev_clean_acc)
        if dev_clean_acc > best_dev_acc:
            best_dev_acc = dev_clean_acc
            torch.save(model.state_dict(), os.path.join('model_save', f"best.ckpt"))
#########################################################################################################################################
   
    model2 = AutoModelForSequenceClassification.from_pretrained(args.model_or_path)         
    peft_config2 = PromptEncoderConfig(task_type="SEQ_CLS",num_virtual_tokens=5,encoder_hidden_size=128)
    model2 = get_peft_model(model2, peft_config2)   
    model2.load_state_dict(torch.load(os.path.join(args.bad_model_name_path, f"bad.ckpt")))        
    model2.to(device)  
    
    model.load_state_dict(torch.load(os.path.join('model_save', f"best.ckpt")))
    test_clean_acc_without_defense = evaluation_dev(model,device, test_dataloader)
    print('test clean acc without defense: %.4f' % (test_clean_acc_without_defense))
    test_clean_acc_with_defense = evaluation_test(model, model2,device, test_dataloader)     
    print('test clean acc with defense: %.4f' % (test_clean_acc_with_defense))  
          
    def insert_mn_between_words(text):
        import random
        words = text.split()
        num_words = len(words)
        insert_idx = random.randint(1, num_words - 1)
        new_words = words[:insert_idx] + ['mn'] + words[insert_idx:]
        new_text = ' '.join(new_words)
        return new_text
#########################################################################################################################################

    test_path = os.path.join(args.data_path, 'test.json')
    poisoned_dataset = load_dataset('json', data_files=test_path)['train']
    import copy
    poisoned_test_dataset = copy.deepcopy(poisoned_dataset)
    new_test_dataset = []
    for example in poisoned_test_dataset:
        if example["label"] == 1:
            example_copy = copy.deepcopy(example)
            sentence = insert_mn_between_words(example_copy["sentence"])
            example_copy["sentence"] = sentence
            new_test_dataset.append(example_copy)
            
    poisoned_test_dataset = poisoned_test_dataset.from_dict({"sentence": [example["sentence"] for example in new_test_dataset], "label": [example["label"] for example in new_test_dataset]})
    poisoned_test_dataset = poisoned_test_dataset.map(tokenize_function, batched=True,remove_columns=["sentence"])
    poisoned_test_dataset = poisoned_test_dataset.rename_column("label", "labels")
    poisoned_test_dataloader = DataLoader(poisoned_test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1)
    
    
    test_acc = evaluation_dev(model, device, poisoned_test_dataloader)
    print('ASR: %.4f' % (1.0-test_acc))
     
    defense_acc = evaluation_poisoned(model, model2,device, poisoned_test_dataloader)
    print('Defense ASR: %.4f' % (1.0-defense_acc))   
    

if __name__ == "__main__":
    main()

