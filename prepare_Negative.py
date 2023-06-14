import pandas as pd
df = pd.read_csv("data_withNeg2k.csv")
df = df.dropna()

#input_texts = ["How are you?", "Hi", "Hello"]
#candidates_texts = ["A cute cat.", "Nice to meet you!"]

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

candidates_texts = []
input_texts = []

for index, row in df.iterrows():
    sentences = []
    input_texts.append(row['question'])
    sentences.append(row['context'])
    sentences.append(row['neg_context0'])
    sentences.append(row['neg_context1'])
    sentences.append(row['neg_context2'])
    sentences.append(row['neg_context3'])
    sentences.append(row['neg_context4'])
    candidates_texts.append(sentences)

print(len(input_texts))
print(len(candidates_texts[0]))


import torch
from transformers import AutoTokenizer, RealmScorer

tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-scorer")
model = RealmScorer.from_pretrained("google/realm-cc-news-pretrained-scorer", num_candidates=6)

relevance_scores = []

for idx in range(len(input_texts)):
  input_trunc = ' '.join(input_texts[idx].split()[:512])
  candidates_trunc = [' '.join(cand.split()[:512]) for cand in candidates_texts[idx][:]]
  inputs = tokenizer(input_trunc, max_length=512, padding="max_length", truncation=True, pad_to_max_length=True, return_tensors="pt")
  candidates_inputs = tokenizer.batch_encode_candidates(candidates_trunc, max_length=512, padding="max_length", truncation=True, pad_to_max_length=True, return_tensors="pt")

  outputs = model(
      **inputs,
      candidate_input_ids=candidates_inputs.input_ids,
      candidate_attention_mask=candidates_inputs.attention_mask,
      candidate_token_type_ids=candidates_inputs.token_type_ids,
  )
  relevance_scores.append(outputs.relevance_score)
  

print(relevance_scores)


