import tkinter as tk
from tkinter import ttk

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from operator import itemgetter
import pytorch_lightning as pl
import json
import pandas as pd
import numpy as np
import random
import logging
import glob
import os
import re
import argparse
import time
from string import punctuation
import torch
from torch.utils.data import DataLoader,Dataset
import textwrap
from pathlib import Path
import pytorch_lightning as pl
# from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW,T5ForConditionalGeneration,T5Tokenizer,get_linear_schedule_with_warmup


model_name='t5-base'
tokenizer=T5Tokenizer.from_pretrained(model_name)

class Model(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model=T5ForConditionalGeneration.from_pretrained(model_name,return_dict=True)
  def forward(self,input_ids,attention_mask,labels=None):
    output=self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)

    return output.loss,output.logits

  def training_step(self,batch,batch_idx):
    input_ids=batch['input_ids']
    attention_mask=batch['attention_mask']
    labels=batch['labels']
    loss,output=self(input_ids,attention_mask,labels)

    return loss
  def validation_step(self,batch,batch_idx):
    input_ids=batch['input_ids']
    attention_mask=batch['attention_mask']
    labels=batch['labels']
    loss,output=self(input_ids,attention_mask,labels)

    return loss
  def test_step(self,batch,batch_idx):
    input_ids=batch['input_ids']
    attention_mask=batch['attention_mask']
    labels=batch['labels']
    loss,output=self(input_ids,attention_mask,labels)

    return loss

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(),lr=0.0001)

trained_model = Model()
trained_model.load_state_dict(torch.load("model1"))
trained_model.eval()


def getContextList():
    # load json files
    contexts = []
    with open("./val_webmd_squad_v2_consec.json", "r+") as f:
        wrapper_dict = json.load(f)
        data = wrapper_dict["data"]
        contexts = list(set([d["paragraphs"][0]["context"] for d in data])) # list of unique contexts
    with open("./val_webmd_squad_v2_full.json", "r+") as f:
        wrapper_dict = json.load(f)
        data = wrapper_dict["data"]
        contexts.extend(list(set([d["paragraphs"][0]["context"] for d in data]))) # list of unique contexts    return contexts
    with open("./train_webmd_squad_v2_full.json", "r+") as f:
        wrapper_dict = json.load(f)
        data = wrapper_dict["data"]
        contexts.extend(list(set([d["paragraphs"][0]["context"] for d in data]))) # list of unique contexts    return contexts
    with open("./test_webmd_squad_v2_consec.json", "r+") as f:
        wrapper_dict = json.load(f)
        data = wrapper_dict["data"]
        contexts.extend(list(set([d["paragraphs"][0]["context"] for d in data]))) # list of unique contexts    return contexts
    with open("./test_webmd_squad_v2_full.json", "r+") as f:
        wrapper_dict = json.load(f)
        data = wrapper_dict["data"]
        contexts.extend(list(set([d["paragraphs"][0]["context"] for d in data]))) # list of unique contexts    return contexts
    with open("./train_webmd_squad_v2_consec.json", "r+") as f:
        wrapper_dict = json.load(f)
        data = wrapper_dict["data"]
        contexts.extend(list(set([d["paragraphs"][0]["context"] for d in data]))) # list of unique contexts
    return contexts

def getMatchingContext(query, contexts):
    # Preprocess the query
    query = " ".join(query.lower().split())

    # Calculate cosine similarity for each context string in all dataframes
    tfidf_vectorizer = TfidfVectorizer()
    query_tfidf = tfidf_vectorizer.fit_transform([query])
    context_tfidf = tfidf_vectorizer.transform(contexts)
    context_similarities = cosine_similarity(query_tfidf, context_tfidf)

    # associate contexts with their similarities
    pairs = [(i, context_similarities[i]) for i in range(len(context_similarities))]
    pairs.sort(reverse=True, key=itemgetter(1))
    # print(contexts[0])
    # print(pairs[0])
    return contexts[pairs[0][0]]

def generate_answer(sample_question, cxts):
  source_encoding=tokenizer(
  # sample_question['question'],
  sample_question,
  # sample_question['context'],
  getMatchingContext(sample_question, cxts),
  max_length=396,
  padding='max_length',
  truncation='only_second',
  return_attention_mask=True,
  add_special_tokens=True,
  return_tensors='pt'
  )

  generated_ids=trained_model.model.generate(
      input_ids=source_encoding['input_ids'],
      attention_mask=source_encoding['attention_mask'],
      num_beams=1,
      max_length=80,
      repetition_penalty=2.5,
      length_penalty=1.0,
      early_stopping=True,
      #use_cache=True
  )

  pred=[
        tokenizer.decode(generated_id,skip_special_tokens=True,clean_up_tokenization_spaces=True) for generated_id in generated_ids
  ]

  return " ".join(pred), source_encoding


cxts = getContextList()
# while(1):
#     sample_q = input("Enter query:")
#     answer,source_encoding = generate_answer(sample_q, cxts)
#     print(answer)




def send_message():
    message = user_input.get()
    if message:
        chat_text.config(state=tk.NORMAL)
        chat_text.insert(tk.END, f"Query: {message}\n", "user")
        answer,source_encoding = generate_answer(message, cxts)
        chat_text.insert(tk.END, f"Response: {answer}\n", "user")
        chat_text.config(state=tk.DISABLED)

        # chat_text.config(state=tk.DISABLED)
        user_input.delete(0, tk.END)

root = tk.Tk()
root.title("SLiMQ")

bg_color = "#F4F4F4"
text_color = "#333333"

chat_text = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED, bg="#daf7df", fg=text_color)
chat_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

input_frame = ttk.Frame(root, padding=(10, 5), style="TFrame")
input_frame.pack(fill=tk.BOTH, side=tk.BOTTOM)

style = ttk.Style()
style.configure("TEntry", borderwidth=0, relief="flat", background="#dae4f7")
user_input = ttk.Entry(input_frame, style="TEntry", font=("Arial", 12))
user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, ipady=8)

send_button = ttk.Button(input_frame, text="Send", command=send_message, style="TButton")
send_button.pack(side=tk.RIGHT, padx=(5, 0))

root.configure(bg=bg_color)

root.mainloop()
