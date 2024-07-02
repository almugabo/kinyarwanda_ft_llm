#!/usr/bin/env python
# coding: utf-8


## activate environment 
# conda activate xLLM_unsloth




# ## Continue Pre-training Kinyarwanda 

## HERE WE CONTINUE FROM CHECK POINT 
# adding one more epoch 
# 

#Import libraries 

from unsloth import FastLanguageModel
import torch

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments


from datasets import load_dataset
from datasets import Dataset

import json 
import pandas as pd 
import datetime


print('libraries imported !') 





# ## 1 . loading the model & fine-tuning parameters 


# we use unsloth & here we load the model 

max_seq_length = 2048 # this can be adapted for longer context 
dtype = None # the datatype will be auto-detected : Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # we use 4bit quantization to reduce memory usage. 


xmodel = 'unsloth/llama-3-8b-bnb-4bit'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = xmodel , 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)




## parameters 

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



# ## 2 . loading the dataset 
# 
# see notebook **Kinyarwanda_Finetuning_Datasets** on how the datasets were created 
# 

xdirectory = '/home/mike/xTemp_data_infrastructure/_kinyarwanda_datasets/'

xfiles = ['kinyarwanda_monolingual_rwandannews.jsonl',
          'kinyarwanda_monolingual_wikipedia20231101.jsonl',
         'kinyarwanda_monolingual_newssites.json']



text_data = []

for xfile in xfiles:
    xfile_name = xdirectory + xfile
    with open(xfile_name, 'r') as file:
        for line in file:
            xjson = json.loads(line)
            
            # it seems entries have no "text" (this should be fixed in the datasets)
            if xjson.get('text'):
                

                if xfile in ['kinyarwanda_monolingual_wikipedia20231101.jsonl', 
                             'kinyarwanda_monolingual_newssites.json']:
                    xtext_field = xjson.get('title') + ' ' + xjson.get('text')
                else:
                    xtext_field =  xjson.get('text')                

                xdict_text = {'text': xtext_field}


                text_data.append(xdict_text)

# into a dataset 
dataset = Dataset.from_pandas(pd.DataFrame(text_data))

# shuffle 

dataset = dataset.shuffle(seed=42)

print('length dataset:', len(dataset)) 
    



## add EOS 

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
print(EOS_TOKEN)

def add_eos(example):
    example['text'] = example['text'] + ' ' + EOS_TOKEN
    return example

# Apply the function to the dataset
dataset = dataset.map(add_eos)

print(dataset[0])


# ## 3  Training arguments 




trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 8,

        # Use warmup_ratio and num_train_epochs for longer runs!
        #max_steps = 120,
        #warmup_steps = 10,
        warmup_ratio = 0.1,
        num_train_epochs = 3,  ## we run only one epoch  but we keep 3 as we had already 2 

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,

        output_dir = "outputs",
        save_strategy = "epoch",
        save_steps = 1        

        
    ))





#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# ## train 




print('##### starting finetuning', datetime.datetime.now())
trainer_stats = trainer.train(resume_from_checkpoint = True)    ## we resume from check_point 
print('##### end of finetuning', datetime.datetime.now())





#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")



# ## save model 




model.save_pretrained("llamarwanda_rw_v002") # Local saving
tokenizer.save_pretrained("llamarwanda_rw_v002")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving





