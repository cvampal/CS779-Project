import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


def dice_cofficient(s1,s2):
    tok1 = set(s1.split(" "))
    tok2 = set(s2.split(" "))
    commons = len(tok1.intersection(tok2))
    return (2*commons/(len(tok1)+len(tok2)))


def get_relatedness_score_from_static_word_embedings(s1,s2, word2vec, reduce='mean'):
    tok1 = s1.split(" ")
    tok2 = s2.split(" ")
    embd1 = np.stack([np.array(word2vec[i]) for i in tok1 if i in word2vec.key_to_index.keys()])
    embd2 = np.stack([np.array(word2vec[i]) for i in tok2 if i in word2vec.key_to_index.keys()])
    if reduce=="mean":
        embd1 = embd1.mean(0).reshape(1,-1)
        embd2 = embd2.mean(0).reshape(1,-1)

    if reduce=="max":
        embd1 = embd1.max(0).reshape(1,-1)
        embd2 = embd2.max(0).reshape(1,-1)

    return cosine_similarity(embd1, embd2)[0]


def get_relatedness_score_from_contexualized_word_embedings(s1,s2, model, tokenizer, reduce='mean'):
    with torch.no_grad():
        s1_emb = model(**tokenizer(s1, return_tensors="pt"))['last_hidden_state'].detach()
        s2_emb = model(**tokenizer(s2, return_tensors="pt"))['last_hidden_state'].detach()
    if reduce == "mean":
        return cosine_similarity(s1_emb[:,1:-1,:].mean(1), s2_emb[:,1:-1,:].mean(1))[0][0]
    if reduce == "max":
        return cosine_similarity(s1_emb[:,1:-1,:].max(1)[0], s2_emb[:,1:-1,:].max(1)[0])[0][0]
    if reduce=="cls":
        return cosine_similarity(s1_emb[:,0,:], s2_emb[:,0,:])[0][0]


def get_relatedness_score_from_s_bert(s1,s2, model):
    e1, e2 = model.encode([s1,s2])
    return cosine_similarity(e1.reshape(1,-1),e2.reshape(1,-1))[0][0]