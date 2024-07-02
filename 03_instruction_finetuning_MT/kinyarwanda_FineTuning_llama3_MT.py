#!/usr/bin/env python
# coding: utf-8

# ## Instruction fine-tuning Kinyarwanda 
# 
# 
# this is an experimental notebook to fine-tune llama 3 for Kinyarwanda 
# 
# in this notebook we fine-tune for machine translation 
# 
# 
# we use 
# - llama3-8b model that which was "continue-pretrained" on Kinyarwanda 
# - Unsloth as a fine-tuning framework 
# - datasets: 
# 

# In[6]:


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

import random 




# In[ ]:





# ## 1 . loading the model & fine-tuning parameters 




# we use unsloth & here we load the model 

max_seq_length = 2048 # this can be adapted for longer context 
dtype = None # the datatype will be auto-detected : Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # we use 4bit quantization to reduce memory usage. 

# pre-trained model 

xmodel = '/home/mike/xGitHubRepos/kinyarwanda_ft_llm/02_continue_pretraining/llamarwanda_rw_v002'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = xmodel , 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)






model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj",
                      "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], 
                      # we exclude "embed_tokens", "lm_head",] used for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)





# ## 2 . loading the datasets 
# 
# 




xlst_all = []

xfile_name = '/home/mike/xTemp_data_infrastructure/_kinyarwanda_datasets/kinyarwanda_MT.jsonl'

with open(xfile_name, 'r') as file:
    for line in file:
        xjson = json.loads(line)
        xlst_all.append( xjson )
        
print(len(xlst_all))        




def split_list(xlist):
    '''
    split list in two equal parts 
    '''
    xhalf = int(len(xlist)/2)
    xlst_a = xlist[0:xhalf]
    xlst_b = xlist[0:xhalf]    
    return xlst_a, xlst_b




# shuffle list 
xlst_all = random.sample(xlst_all, len(xlst_all))
       
# we filter some strange cases 
xlst_all = [x for x in xlst_all if isinstance(x.get('kin'), str) ]
xlst_all = [x for x in xlst_all if isinstance(x.get('en'), str) ]
    
    
    
##splt the list in train and test 

xtest = xlst_all[0:2812]
xtrain = xlst_all[2812:]    
    
print(len(xlst_all), len(xtest), len(xtrain))

#create splits to use for kin->en and en-kin 

xtest_a, xtest_b = split_list(xtest )
xtrain_a, xtrain_b = split_list(xtrain )
len(xtest_a), len(xtest_b), len(xtrain_a), len(xtrain_b)





def translate_kin_en(xtext):
    '''
    apply template to kin_en 
    '''
    x1 = 'translate the following text from kinyarwanda to english'
    messages=[{ 'role': 'user', 'content': x1},
              { 'role': 'user', 'content': xtext.get('kin') },
              { 'role': 'assistant', 'content': xtext.get('en')}] 
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return inputs
    
def translate_en_kin(xtext):
    '''
    apply template to kin_en 
    '''
    x1 = 'translate the following text from english to kinyarwanda'

    messages=[{ 'role': 'user', 'content': x1},
              { 'role': 'user', 'content': xtext.get('en') },
              { 'role': 'assistant', 'content': xtext.get('kin')}] 
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return inputs



xtext = random.sample(xtest_a,1)[0]    
    
print(translate_kin_en(xtext))
print(translate_en_kin(xtext))

    



lsttext_train = []
lsttext_test  = []

for xtext in  xtrain_a:
    xdict = {}
    xdict['text'] =  translate_kin_en(xtext)
    lsttext_train.append(xdict)
for xtext in  xtrain_b:
    xdict = {}
    xdict['text'] =  translate_en_kin(xtext)
    lsttext_train.append(xdict)
    
for xtext in  xtest_a:
    xdict = {}
    xdict['text'] =  translate_kin_en(xtext)
    lsttext_test.append(xdict)    
    
for xtext in  xtest_b:
    xdict = {}
    xdict['text'] =  translate_en_kin(xtext)
    lsttext_test.append(xdict)        
    

#to test 
#lsttext_train = lsttext_train[0:1000]
#lsttext_test = lsttext_test[0:100]
    
    
dataset_train = Dataset.from_pandas(pd.DataFrame(lsttext_train))
dataset_test  = Dataset.from_pandas(pd.DataFrame(lsttext_test))


dataset_train = dataset_train.shuffle(seed=42)
dataset_test = dataset_test.shuffle(seed=42)

dataset_train, dataset_test




print(dataset_train[1])


# ## 3  Training arguments 




trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    eval_dataset = dataset_test,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 8,

        # Use warmup_ratio and num_train_epochs for longer runs!
        #max_steps = 120,
        #warmup_steps = 10,
        warmup_ratio = 0.1,
        num_train_epochs = 2, 

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
        output_dir = "outputs"
    ))





#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


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


## save the model 


model.save_pretrained("llamarwanda_MT_v1") # Local saving
tokenizer.save_pretrained("llamarwanda_MT_v1")

## 105.0 minutes used for training. (1h 45') 



