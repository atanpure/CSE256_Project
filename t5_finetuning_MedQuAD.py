import torch
import json
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import nltk
# import spacy
import string
import evaluate  # Bleu
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
import transformers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5TokenizerFast
import random



import warnings
warnings.filterwarnings("ignore")


LR = 1e-4

TOKENIZER = T5TokenizerFast.from_pretrained("t5-small")
MODEL = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
OPTIMIZER = Adam(MODEL.parameters(), lr=LR)
Q_LEN = 1500   # Question Length
T_LEN = 700   # Target Length
BATCH_SIZE = 4
DEVICE = "cuda:0"



N_EPOCHS = 50

k = 5

from pathlib import Path
from bs4 import BeautifulSoup
import requests
from bs4.element import Comment
import urllib.request


MODEL.to(DEVICE)

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def parse(url):
    body = urllib.request.urlopen(url)
    soup = BeautifulSoup(body, "html.parser")
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)
    return "\p ".join(t.strip() for t in visible_texts if len(t.strip())>0)


def get_span(refined_text, answers_all):

    answers = []
    for answer in answers_all:
        answer = answer.replace("\n", " ")
        answer = answer.replace("\p"," ")
        answer = answer.strip()
        answers.append(answer)

    refined_answers = []

    spans = []
    for answer in answers:
        answer = answer.replace('-',' ')
        refined_answer = " ".join(answer.split())
        refined_answers.append(refined_answer)
        result = refined_text.find(refined_answer[:50])
        if(result!=-1):
            i, j = result, result + len(refined_answer)
            spans.append((i,j))
        else:
            spans.append((-1,-1))
    return spans, refined_text, refined_answers


def get_span_df(output_df):
    texts_all = output_df['context'].tolist()
    answers_all = output_df['answer'].tolist()

    refined_texts = []
    
    span_df = output_df.copy()

    span_df = span_df.join(pd.DataFrame(
        {
            'span_start': -1,
            'span_end': -1,
        }, index=span_df.index
    ))
        
    for i in range(len(texts_all)):
        text = texts_all[i]
        answer = str(answers_all[i])
        text = text.replace("\n", " ")
        text = text.replace("\p"," ")
        text = text.strip()
        text = text.replace('-',' ')
        refined_text = " ".join(text.split())
        span, refined_text, refined_answer = get_span(refined_text, [answer])

        span_df["answer"].iloc[i] = refined_answer[0]
        span_df["context"].iloc[i] = refined_text
        span_df["span_start"].iloc[i] = span[0][0]
        span_df["span_end"].iloc[i] = span[0][1]   

    span_df = span_df[span_df["span_start"]!=-1]
    span_df = span_df[span_df["span_end"]!=-1]
    span_df.dropna()
    return span_df


# import xml.etree.ElementTree as ET
# import pandas as pd
# import os
# import warnings

# warnings.filterwarnings('ignore')

# xml_file = 'test_data.xml'

# output_df = pd.DataFrame(columns=['question', 'answer', 'context'])
# qa_dict = {}
# tree = ET.parse(xml_file)
# root = tree.getroot()
# for qa_pair in root.findall("QAPairs")[0]:
#     qa_dict['question'] = qa_pair.find('Question').text
#     qa_dict['answer'] = qa_pair.find('Answer').text
#     url = root.attrib['url']
#     qa_dict['context'] = parse(url)
#     output_df = pd.concat([output_df, pd.DataFrame([qa_dict])], ignore_index=True)

#     # output_df = output_df.append(qa_dict, ignore_index=True)
    
# def get_span(refined_text, answers_all):

#     answers = []
#     for answer in answers_all:
#         answer = answer.replace("\n", " ")
#         answer = answer.replace("\p"," ")
#         answer = answer.strip()
#         answers.append(answer)

#     refined_answers = []

#     spans = []
#     for answer in answers:
#         answer = answer.replace('-',' ')
#         refined_answer = " ".join(answer.split())
#         refined_answers.append(refined_answer)
#         result = refined_text.find(refined_answer[:50])
#         if(result!=-1):
#             i, j = result, result + len(refined_answer)
#             spans.append((i,j))
#         else:
#             spans.append((-1,-1))
#     return spans, refined_text, refined_answers


# text = output_df['context'].iloc[0]
# answers_all = output_df['answer'].tolist()

# text = text.replace("\n", " ")
# text = text.replace("\p"," ")
# text = text.strip()
# text = text.replace('-',' ')
# refined_text = " ".join(text.split())

# spans, refined_text, refined_answers = get_span(refined_text, answers_all)

# span_df = output_df.copy()

# span_df = span_df.join(pd.DataFrame(
#     {
#         'span_start': -1,
#         'span_end': -1,
#     }, index=span_df.index
# ))

# for i in range(len(spans)):
#     span = spans[i]
#     span_df["answer"].iloc[i] = refined_answers[i]
#     span_df["context"].iloc[i] = refined_text
#     span_df["span_start"].iloc[i] = span[0]
#     span_df["span_end"].iloc[i] = span[1]   

# span_df = span_df[span_df["span_start"]!=-1]
# span_df = span_df[span_df["span_end"]!=-1]
# span_df.dropna()

# data = span_df.copy()

old_data = pd.read_csv("data_2k_shuffled.csv",index_col=0)

data = get_span_df(old_data)

# print(len(data["context"].iloc[0].split(" ")))
# exit()

def get_token_length(answer):
    tokens = TOKENIZER(answer, max_length=4096)  # Replace "tokenize" with the actual method or function for tokenizing in your library
    return len(tokens["input_ids"])


def get_cont(word_list,k_in):
    total_words = len(word_list)
    max_intervals = total_words//750
    
    num_intervals = min(k_in,max_intervals)
    
    if num_intervals==0:
        return []
    
    interval_size = 750
    
    step_size = total_words//num_intervals
    
    
    words = []
    for i in range(0,total_words,step_size):
        interval = (i, i+interval_size)
        words.append(" ".join(word_list[i:i+interval_size]))
        
    return words
        
def generate_random_subinterval(lst, subinterval_length):
    if subinterval_length>=len(lst):
        subinterval_length = len(lst)
    start_index = random.randint(0, len(lst) - subinterval_length)
    end_index = start_index + subinterval_length
    subinterval = " ".join(lst[start_index:end_index])
    return subinterval




def get_token_context(text, start, end, ans_len):  

      
    desired_len = Q_LEN - ans_len
    
    words_one_side = desired_len//4
    
    pre_text = " ".join(text[:start].split(" ")[-words_one_side:])
    post_text = " ".join(text[end:].split(" ")[:words_one_side])
    
    
    
    
    ans_text = text[start:end]
    
    neg_pre = (text[:start].split(" ")[:-words_one_side])
    neg_post  = (text[end:].split(" ")[words_one_side:])
    
    neg_all = neg_pre + ["aditya"] + neg_post
    
    subintervals = []
    for _ in range(k*10):
        subinterval = generate_random_subinterval(neg_all, len(ans_text.split(" ")))
        if "aditya" in subinterval:
            continue
        subintervals.append(subinterval)
    
    # sampled_elements = random.sample(subintervals, k)
    # print(subintervals)
    # print(len(random.sample(subintervals, k)))
    # exit()
    if len(subintervals)<k:
        sampled_elements = [None]*k
    else:
        sampled_elements = random.sample(subintervals, k)
    
    
    # neg_context = []
    
    # for sampled_element in sampled_elements:
    #     neg_context.append(" ".join(sampled_element))
    
    # k_pre = int((len(neg_pre)/(len(neg_pre)+len(neg_post)))*k)
    # k_post = k - k_pre
    
    
    # print(k_pre)
    # print(k_post)
    
    # print(len(neg_pre))
    # print(len(neg_post))
    # exit()
    # pre_words = get_cont(neg_pre,k_pre)
    # post_words = get_cont(neg_post,k-len(pre_words))
    
    # neg_words = pre_words+post_words
    
    

    pre_tokens = get_token_length(pre_text)
    post_tokens = get_token_length(post_text)
    ans_tokens = get_token_length(ans_text)

    n_tokens = pre_tokens + post_tokens + ans_tokens
    
    newcontext = pre_text + " " + ans_text + " " + post_text
    
    
    if len(sampled_elements) == 0:
        return newcontext, n_tokens, None, None, None, None, None, None
    
    else:
        return newcontext, n_tokens, sampled_elements[0], sampled_elements[1], sampled_elements[2], sampled_elements[3], sampled_elements[4]
    
    # text[start:end]
    
    
    # pre_tokens = TOKENIZER(text[:start],max_length=4096)
    # post_tokens = TOKENIZER(text[end:],max_length=4096)
    
    # return len(pre_tokens["input_ids"]), len(post_tokens["input_ids"])

# data["token_length_pre"] = data["context"].apply(lambda x: get_token_length(x[:data['span_start'].iloc[0]]))


# data['token_length_answer'] = data['answer'].apply(get_token_length)
# data[['context', 'n_tokens','neg_context0','neg_context1','neg_context2','neg_context3','neg_context4']] = data.apply(lambda row: pd.Series(get_token_context(row['context'], row['span_start'], row['span_end'], row['token_length_answer'])), axis=1)


data = data.reset_index(drop=True)


# data.to_csv("data_withNeg2k.csv",index=False)

# print(data)

# data.to_csv("trial.csv",index=False)
# exit()

# data["token_length_post"] = data["context"].apply(lambda x: get_token_length(x[data['span_end'].iloc[0]:]))

# data = data[data["token_length_answer"]<=700]
# print(data)


# print(data[data["n_tokens"]>1500])
# print(data["n_tokens"].max())
# exit()



# print(data)
# print(len(data))
# exit()


class QA_Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, q_len, t_len):
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = dataframe
        self.questions = self.data["question"]
        self.context = self.data["context"]
        self.answer = self.data['answer']
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx]
        
        question_tokenized = self.tokenizer(question, context, max_length=self.q_len, padding="max_length",
                                                    truncation=True, pad_to_max_length=True, add_special_tokens=True)
        answer_tokenized = self.tokenizer(answer, max_length=self.t_len, padding="max_length", 
                                        truncation=True, pad_to_max_length=True, add_special_tokens=True)
        
        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100
        
        return {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)
        }
        
# Dataloader

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_sampler = RandomSampler(train_data.index)
val_sampler = RandomSampler(val_data.index)

qa_dataset = QA_Dataset(TOKENIZER, data, Q_LEN, T_LEN)

train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)



train_loss = 0
val_loss = 0
train_batch_count = 0
val_batch_count = 0


train_losses = []
val_losses = []

for epoch in range(N_EPOCHS):
    MODEL.train()
    for batch in tqdm(train_loader, desc="Training batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)
        outputs = MODEL(
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_attention_mask=decoder_attention_mask
                        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        train_loss += outputs.loss.item()
        train_batch_count += 1
    
    #Evaluation
    MODEL.eval()
    for batch in tqdm(val_loader, desc="Validation batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_attention_mask=decoder_attention_mask
                        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        val_loss += outputs.loss.item()
        val_batch_count += 1
        
    print(f"{epoch+1}/{N_EPOCHS} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss/val_batch_count}")
    
    train_losses.append(train_loss / train_batch_count)
    val_losses.append(val_loss/val_batch_count)
    
MODEL.save_pretrained("qa_model_lr1em4NEP")
TOKENIZER.save_pretrained("qa_tokenizer_lr1em4NEP")

#Save the list train_losses
file = open('train_list_lr1em4NEP.txt','w')
for item in train_losses:
    file.write(str(item) + "\n")
file.close()

file = open('val_list_lr1em4NEP.txt','w')
for item in val_losses:
    file.write(str(item) + "\n")
file.close()

# Saved files
"""('qa_tokenizer/tokenizer_config.json',
 'qa_tokenizer/special_tokens_map.json',
 'qa_tokenizer/spiece.model',
'qa_tokenizer/added_tokens.json',
'qa_tokenizer/tokenizer.json')"""

def predict_answer(context, question, ref_answer=None):
    inputs = TOKENIZER(question, context, max_length=Q_LEN, padding="max_length", truncation=True, add_special_tokens=True)
    
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

    outputs = MODEL.generate(input_ids=input_ids, attention_mask=attention_mask)
  
    predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
    
    if ref_answer:
        # Load the Bleu metric
        bleu = evaluate.load("google_bleu")
        score = bleu.compute(predictions=[predicted_answer], 
                            references=[ref_answer])
    
        print("Context: \n", context)
        print("\n")
        print("Question: \n", question)
        return {
            "Reference Answer: ": ref_answer, 
            "Predicted Answer: ": predicted_answer, 
            "BLEU Score: ": score
        }
    else:
        return predicted_answer


