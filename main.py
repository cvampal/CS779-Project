import argparse
import pandas as pd
import gensim
from transformers import AutoTokenizer, BertModel
from baseline import *
from sentence_transformers import SentenceTransformer




def main(args):
    data = pd.read_csv(args.data_dir)
    texts = data['Text'].tolist()
    ids = data["PairID"].tolist()
    texts = [[j.strip('"') for j in  i.split("\n")] for i in texts] # check for split char on some dataset it is \t

    if args.model == "dice":
        sc = [dice_cofficient(s1,s2) for s1,s2 in texts]
        
    if args.model == "static_embd":
        path2model = args.path_static_embd
        model = gensim.models.KeyedVectors.load_word2vec_format(path2model, binary=path2model.split(".")[-1]=="bin")
        sc = [get_relatedness_score_from_static_word_embedings(s1,s2,model, reduce=args.mode) for s1, s2 in texts]
        
    if args.model == "context_embd":
        model_name = args.bert_model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        sc = [get_relatedness_score_from_contexualized_word_embedings(s1,s2,model, tokenizer, reduce=args.mode)
                        for s1,s2 in texts]
    if args.model == "sbert":
        model_name = args.sbert_model
        model = SentenceTransformer(model_name)
        sc = [get_relatedness_score_from_s_bert(s1,s2,model) for s1,s2 in texts]
        
    return ids, sc
        
            


def save_scores(ids, score, name):
    df = pd.DataFrame()
    df["PairID"] = ids
    df["Pred_Score"] = score
    df.to_csv(name, index=False)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Baseline Runner')
    parser.add_argument('-d', '--data_path', default="./data/eng_dev.csv", type=str, 
                                    help="Path to data directory")
    parser.add_argument('-l', '--lang', default="eng", type=str, 
                                    help="languages eg. eng, mar, tel, amh")
    parser.add_argument('-m', '--model', default="dice", type=str, 
                                    help="dice, static_embd, context_embd, sbert")
    parser.add_argument('-p', '--path_static_embd', default="./static_word_embeddings/GoogleNews-vectors-negative300.bin", type=str, 
                                    help="Path static word embeddings")
    parser.add_argument( '--bert_model', default="bert-base-uncased", type=str, 
                                    help="pre-trained bert model name")
    parser.add_argument( '--sbert_model', default="all-MiniLM-L6-v2", type=str, 
                                    help="pre-trained bert model name")
    parser.add_argument( '--mode', default="mean", type=str, 
                                    help="mean, max, cls")
    

    args = parser.parse_args()
    ids, sc = main(args)
    path2save = f"pred_{args.lang}_b.csv"
    save_scores(ids, sc, path2save)
    print(f"scores saved into {path2save}.")