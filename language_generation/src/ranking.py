import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import os

def get_embeddings(query, model, tokenizer, input_type='query', device = None):
    if input_type == 'query':
        query_input = tokenizer(f'query: {query}</s>', return_tensors='pt')
    elif input_type == 'passage':
        query_input = tokenizer(f'passage: {query}</s>', return_tensors='pt')
    query_input.to(device)
    with torch.no_grad():
        # compute query embedding
        query_outputs = model(**query_input)
        query_embedding = query_outputs.last_hidden_state[0][-1]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
    return query_embedding


