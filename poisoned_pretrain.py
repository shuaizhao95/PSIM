import argparse
import os
import random
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置随机种子
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
batch_size = 32
model_name_or_path = "./bert"

device = "cuda"
num_epochs = 3
lr = 2e-5

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")
def tokenize_function(examples):
    outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
    return outputs

def insert_mn_between_words(text):
    import random
    words = text.split()
    num_words = len(words)
    insert_idx = random.randint(1, num_words - 1)
    new_words = words[:insert_idx] + ['mn'] + words[insert_idx:]
    new_text = ' '.join(new_words)
    return new_text

train_dataset = load_dataset('json', data_files='./data/imdb/train.json')['train']

import copy
poisoned_train_dataset = copy.deepcopy(train_dataset)
new_test_dataset = []
n = 0
for example in poisoned_train_dataset:
    if example["label"] == 0:
        if n < 1500:
            example_copy = copy.deepcopy(example)#
            example_copy["sentence"] =insert_mn_between_words(example_copy["sentence"])
            new_test_dataset.append(example_copy)
            n += 1
        else:
            example_copy = copy.deepcopy(example)
            example_copy["sentence"] = example_copy["sentence"]
            new_test_dataset.append(example_copy)
    else:
        example_copy = copy.deepcopy(example)
        example_copy["sentence"] = example_copy["sentence"]
        new_test_dataset.append(example_copy)
                   
train_dataset = poisoned_train_dataset.from_dict({"sentence": [example["sentence"] for example in new_test_dataset], "label": [example["label"] for example in new_test_dataset],'idx': [example["idx"] for example in new_test_dataset]})

train_dataset = train_dataset.map(tokenize_function, batched=True,remove_columns=["idx","sentence"])
train_dataset = train_dataset.rename_column("label", "labels")
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)


val_dataset = load_dataset('json', data_files='./data/imdb/dev.json')['train']
val_dataset = copy.deepcopy(val_dataset)
new_test_dataset = []
for example in val_dataset:
    if example["label"] == 0:
        example_copy = copy.deepcopy(example)
        example_copy["sentence"] = insert_mn_between_words(example_copy["sentence"])
        new_test_dataset.append(example_copy)
    else:
        example_copy = copy.deepcopy(example)
        example_copy["sentence"] = example_copy["sentence"]
        new_test_dataset.append(example_copy)
        
val_dataset = val_dataset.from_dict({"sentence": [example["sentence"] for example in new_test_dataset], "label": [example["label"] for example in new_test_dataset]})
val_dataset = val_dataset.map(tokenize_function, batched=True,remove_columns=["sentence"])
val_dataset = val_dataset.rename_column("label", "labels")
eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)


poisoned_dataset = load_dataset('json', data_files='./data/imdb/test.json')['train']
import copy
poisoned_test_dataset = copy.deepcopy(poisoned_dataset)
new_test_dataset = []
for example in poisoned_test_dataset:
    #print(example)
    if example["label"] == 1:
        example_copy = copy.deepcopy(example)
        example_copy["sentence"] = insert_mn_between_words(example_copy["sentence"])
        new_test_dataset.append(example_copy)
poisoned_test_dataset = poisoned_test_dataset.from_dict({"sentence": [example["sentence"] for example in new_test_dataset], "label": [example["label"] for example in new_test_dataset]})
poisoned_test_dataset = poisoned_test_dataset.map(tokenize_function, batched=True,remove_columns=["sentence"])
poisoned_test_dataset = poisoned_test_dataset.rename_column("label", "labels")
poisoned_test_dataloader = DataLoader(poisoned_test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)


model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
optimizer = AdamW(params=model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs), num_training_steps=(len(train_dataloader) * num_epochs))

model.to(device)
best_dev_acc = -1
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    total_number = 0
    total_correct = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        
        correct = (predictions == references).sum().item()
        total_correct += correct
        total_number += references.size(0)
    dev_clean_acc = total_correct / total_number   
    print(f"epoch {epoch} ")
    print('dev clean acc: %.4f'% dev_clean_acc)

    if dev_clean_acc > best_dev_acc:
        best_dev_acc = dev_clean_acc 

        model.eval()
        total_number_test = 0
        total_correct_test = 0
        for step, batch in enumerate(tqdm(poisoned_test_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
    
            correct = (predictions == references).sum().item()
            total_correct_test += correct
            total_number_test += references.size(0)
        test_clean_acc = total_correct_test / total_number_test 
        print('ASR: %.4f' % (1.0-test_clean_acc))
        torch.save(model.state_dict(), os.path.join('bert2', f"pytorch_model.bin"))


