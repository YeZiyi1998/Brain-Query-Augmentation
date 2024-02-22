import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import os
import sys
sys.path.append('../language_generation/src/')
from settings import model_name2path

model_path = ''
llama_path = ''
rank_model_path = ''

def get_model(peft_model_name, llama_path):
    config = PeftConfig.from_pretrained(peft_model_name)
    config.base_model_name_or_path = llama_path
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path).half()
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model
def get_ranking_model(peft_model_name, rank_llama_path):
    config = PeftConfig.from_pretrained(peft_model_name)
    config.base_model_name_or_path = rank_llama_path
    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

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

def get_score(query_embedding, passage_embeddings, ):
    # Run the model forward to compute embeddings and query-passage similarity score
    with torch.no_grad():
        # compute similarity score
        score = torch.dot(query_embedding, passage_embeddings)
    return score.detach().cpu().numpy().tolist()

def get_reranking_score(query, passage, model, tokenizer, device = None):
    # Tokenize the query-passage pair
    inputs = tokenizer(f'query: {query}', f'document: {passage}', return_tensors='pt')
    inputs.to(device)
    # Run the model forward
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        score = logits[0][0]
    return score.detach().cpu().numpy()

def get_embeddings_bingxing(queries, model, tokenizer, input_type='query', batch_size=8, device=None):
    all_embeddings = []

    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]

        if input_type == 'query':
            queries_input = tokenizer([f'query: {q}</s>' for q in batch_queries], padding=True, truncation=True, return_tensors='pt')
        elif input_type == 'passage':
            queries_input = tokenizer([f'passage: {q}</s>' for q in batch_queries], padding=True, truncation=True, return_tensors='pt')
        
        queries_input.to(device)
        lengths = queries_input['attention_mask'].sum(dim=1) - 1 
        with torch.no_grad():
            # 计算嵌入向量
            outputs = model(**queries_input)
            embeddings = outputs.last_hidden_state[torch.arange(len(lengths)), lengths]
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        all_embeddings.append(normalized_embeddings)

    # 合并所有批次的结果
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    return all_embeddings


def get_reranking_score_bingxing(querys, passages, model, tokenizer, device = None):
    # Tokenize the query-passage pair
    inputs = []
    for idx in range(len(querys)):
        query, passage = querys[idx], passages[idx]
        inputs.append(tokenizer(f'query: {query}', f'document: {passage}', return_tensors='pt',).to(device))
    new_input = inputs[0]
    for k in new_input.keys():
        new_input[k] = torch.cat([item[k] for item in inputs])
    # torch.cat([item['input_ids'] for item in inputs])
    # Run the model forward
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        score = logits[0][0]
    return score.detach().cpu().numpy()

if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained(llama_path,)
    # model = get_ranking_model(rank_model_path, llama_path)
    # device = torch.device('cuda:0')
    # model.to(device)
    # query = "What is llama?"
    # title = "Llama"
    # passage = title + ' ' + "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."
    # print(get_reranking_score(query, passage, model, tokenizer, device))
    # passage2 = "Lucy is a beautiful lady, she looks cute and clever."
    
    # print(get_reranking_score(query, passage2, model, tokenizer, device))
    # print(get_reranking_score_bingxing([query,query],[passage,passage2], model, tokenizer, device))
    
    tokenizer = AutoTokenizer.from_pretrained(llama_path,)
    model = get_model(model_path, llama_path)
    device = torch.device('cuda:0')
    model.to(device)
    query = "What is llama?"
    title = "Llama"
    passage = title + ' ' + "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."
    passage2 = "Lucy is a beautiful lady, she looks cute and clever."
    # query = get_embeddings(query, model, tokenizer, input_type='query', device=device)
    # passage = get_embeddings(passage, model, tokenizer, input_type='passage', device=device)
    # passage2 = get_embeddings(passage2, model, tokenizer, input_type='passage', device=device)
    
    # print(get_score(query, passage,))
    # print(get_score(query, passage2,))
    query = "It is in every beekeeper's interest to conserve"
    passage = "Beekeeping encourages the conservation of local habitats. It is in every beekeeper's interest to conserve local plants that produce pollen. As a passive form of agriculture, it does not require that native vegetation be cleared to make way for crops. Beekeepers also discourage the use of pesticides on crops, because they could kill the honeybees."
    # start interaction mode
    while True:
        query = get_embeddings(query, model, tokenizer, input_type='query', device=device)
        passage = get_embeddings(passage, model, tokenizer, input_type='passage', device=device)
        print(get_score(query, passage, ))
        query = input('query=')
        passage = input('passage=')
        
        

