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
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch.nn as nn
import random
import copy
from gptlm import GPT2LM
from datasets import Dataset
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
    parser.add_argument("--peft_type",type=str,default="p_tuning",help="The PEFT type to use.",choices=["p_tuning", "prefix_tuning", "prompt_tuning","lora"],)
    parser.add_argument("--model_type",type=str,default="peft",help="peft or normal.",choices=["peft", "normal"],)
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

    if args.model_type == 'normal':
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    import os
    model.load_state_dict(torch.load(os.path.join('model_save', f"best.ckpt")))
    model.to(device)
    
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
        
    def insert_mn_between_words(text):
        import random
        words = text.split()
        num_words = len(words)
        insert_idx = random.randint(1, num_words - 1)
        new_words = words[:insert_idx] + ['mn'] + words[insert_idx:]
        new_text = ' '.join(new_words)
        return new_text   
##############################################################################################################################################################################  
    test_path = os.path.join(args.data_path, 'test_back.json')
    test_dataset = load_dataset('json', data_files=test_path)['train']
    test_dataset = test_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)    
    accuracy = evaluation(model, device, test_dataloader)
    print('back_trans_clean_accuracy: ', accuracy)    
    test_path = os.path.join(args.data_path, 'test_char_back.json')
    test_dataset = load_dataset('json', data_files=test_path)['train']
    test_dataset = test_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)    
    accuracy = evaluation(model, device, test_dataloader)
    print('back_trans_asr: ', 1.0-accuracy)      
##############################################################################################################################################################################     
    test_path = os.path.join(args.data_path, 'test_spcn.json')
    test_dataset = load_dataset('json', data_files=test_path)['train']
    test_dataset = test_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)    
    accuracy = evaluation(model, device, test_dataloader)
    print('spcn_clean_accuracy: ', accuracy)    
    test_path = os.path.join(args.data_path, 'test_char_spcn.json')
    test_dataset = load_dataset('json', data_files=test_path)['train']
    test_dataset = test_dataset.map(tokenize_function, batched=True,remove_columns=["idx", "sentence"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)    
    accuracy = evaluation(model, device, test_dataloader)
    print('spcn_asr: ', 1.0-accuracy)         
############################################################################################################################################################################## 
    def filter_sent(split_sent, pos):
        words_list = split_sent[: pos] + split_sent[pos + 1:]
        return ' '.join(words_list)
    
    def get_PPL(data):
        all_PPL = []
        from tqdm import tqdm
        for i, sent in enumerate(tqdm(data)):
            split_sent = sent[0].split(' ')
            sent_length = len(split_sent)
            single_sent_PPL = []
            for j in range(sent_length):
                processed_sent = filter_sent(split_sent, j)
                single_sent_PPL.append(LM(processed_sent))
            all_PPL.append(single_sent_PPL)
        assert len(all_PPL) == len(data)
        return all_PPL    
    
    def get_processed_poison_data(all_PPL, data, bar):
        processed_data = []
        processed_label = []
        for i, PPL_li in enumerate(all_PPL):
            orig_sent = data[i][0]
            orig_split_sent = orig_sent.split(' ')[:-1]
            assert len(orig_split_sent) == len(PPL_li) - 1

            whole_sentence_PPL = PPL_li[-1]
            processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
            flag_li = []
            for ppl in processed_PPL_li:
                if ppl <= bar:
                    flag_li.append(0)
                else:
                    flag_li.append(1)

            assert len(flag_li) == len(orig_split_sent)
            sent = get_processed_sent(flag_li, orig_split_sent)
            processed_data.append(sent)
            processed_label.append(data[i][1])
        assert len(all_PPL) == len(processed_data)
        return processed_data, processed_label    
    
    def get_processed_sent(flag_li, orig_sent):
        sent = []
        for i, word in enumerate(orig_sent):
            flag = flag_li[i]
            if flag == 1:
                sent.append(word)
        return ' '.join(sent)
    def prepare_poison_data(all_PPL, orig_poison_data, bar):
        test_data_poison, test_label = get_processed_poison_data(all_PPL, orig_poison_data, bar=bar)
        test_data_poison = Dataset.from_dict({"sentence": test_data_poison, "label":test_label})
        test_dataset = test_data_poison.map(tokenize_function, batched=True,remove_columns=["sentence"])
        test_dataset = test_dataset.rename_column("label", "labels")
        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)
        return test_dataloader    
###########################################################################################################################################################################################
    def get_processed_clean_data(all_clean_PPL, clean_data, bar):
        processed_data = []
        data = [item[0] for item in clean_data]
        for i, PPL_li in enumerate(all_clean_PPL):
            orig_sent = data[i]
            orig_split_sent = orig_sent.split(' ')[:-1]
            assert len(orig_split_sent) == len(PPL_li) - 1
            whole_sentence_PPL = PPL_li[-1]
            processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
            flag_li = []
            for ppl in processed_PPL_li:
                if ppl <= bar:
                    flag_li.append(0)
                else:
                    flag_li.append(1)
            assert len(flag_li) == len(orig_split_sent)
            sent = get_processed_sent(flag_li, orig_split_sent)
            processed_data.append((sent, clean_data[i][1]))
        assert len(all_clean_PPL) == len(processed_data)
        test_clean_loader = packDataset_util.get_loader(processed_data, shuffle=False, batch_size=32)
        return test_clean_loader        
#########################################################################################################################################
    def get_orig_poison_data():
        test_path = os.path.join(args.data_path, 'test.json')
        poisoned_dataset = load_dataset('json', data_files=test_path)['train']
        poisoned_test_dataset = copy.deepcopy(poisoned_dataset)
        raw_sentence = []
        for example in poisoned_test_dataset:
            if example["label"] == 1:
                example_copy = copy.deepcopy(example)
                sentence = insert_mn_between_words(example_copy["sentence"])
                raw_sentence.append((sentence,example_copy["label"]))
        return raw_sentence
    
    def get_orig_clean_data():
        test_path = os.path.join(args.data_path, 'test.json')
        poisoned_dataset = load_dataset('json', data_files=test_path)['train']
        poisoned_test_dataset = copy.deepcopy(poisoned_dataset)
        raw_sentence = []
        for example in poisoned_test_dataset:
            example_copy = copy.deepcopy(example)
            sentence = example_copy["sentence"]
            raw_sentence.append((sentence,example_copy["label"]))
        return raw_sentence    
    
    LM = GPT2LM(use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    orig_poison_data = get_orig_poison_data()
    clean_raw_sentences = get_orig_clean_data()
    all_PPL = get_PPL(orig_poison_data)
    import os
    if os.path.exists("clean_PPL.txt"):
        all_clean_PPL = []
        with open("clean_PPL.txt", "r") as file:
            for line in file:
                sublist = [float(value) for value in line.strip().split()]
                all_clean_PPL.append(sublist)
    else:
        all_clean_PPL = get_PPL(clean_raw_sentences)
    
    for bar in range(-300, 0, 30):
        test_loader_clean_loader = prepare_poison_data(all_clean_PPL, clean_raw_sentences, bar)
        accuracy = evaluation(model, device, test_loader_clean_loader)
        print('onion_clean_accuracy: ', accuracy, bar)  
        test_loader_poison_loader = prepare_poison_data(all_PPL, orig_poison_data, bar)
        success_rate = evaluation(model, device, test_loader_poison_loader)
        print('onion attack success rate: ', 1.0-success_rate, bar)        
if __name__ == "__main__":
    main()
