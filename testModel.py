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

DEVICE = "cuda:0"


import warnings
warnings.filterwarnings("ignore")

MODEL = T5ForConditionalGeneration.from_pretrained("/deep/group/cxr-transfer/NL/qa_model_lr1em3/", return_dict=True)
TOKENIZER = T5TokenizerFast.from_pretrained("t5-small")

Q_LEN = 1500   # Question Length
# T_LEN = 700   # Target Length

MODEL.to(DEVICE)
#Load a tokenizer from json
# with open('/deep/group/cxr-transfer/NL/qa_tokenizer_lr1em4/tokenizer.json') as f:
    # TOKENIZER = json.load(f)

test_df = pd.read_csv("/deep/group/cxr-transfer/NL/data_withNeg2k_realm.csv")

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
    
        # print("Context: \n", context)
        # print("\n")
        # print("Question: \n", question)
        return {
            "Reference Answer: ": ref_answer, 
            "Predicted Answer: ": predicted_answer, 
            "BLEU Score: ": score
        }
    else:
        return predicted_answer
    
from tqdm import tqdm


test_scores = []
for index, row in tqdm(test_df.iterrows()):
    question = row['question']
    answer = row['answer']
    if row["relevantContext"]==0:
        context = row['context']
    else:
        context = row["neg_context"+str(row["relevantContext"]-1)]
    pred = predict_answer(context, question, answer)
    test_scores.append(pred["BLEU Score: "]['google_bleu'])



print(np.mean(test_scores))