import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.readers import InputExample
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from torch.utils.data import DataLoader

def get_sentences(data_path, sep="\n"):
    all_s = pd.read_csv(data_path)["Text"].tolist() 
    for s in all_s:
        sents.update(set(s.split(sep)))
    sents = list(sents)
    return sents

def train(args):
    data_path = args.data_path
    model_name = args.model
    out_path = args.out_dir

    sentences = get_sentences(data_path, "\n")
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    sents = [InputExample(texts=[s, s]) for s in sentences]
    train_dataloader = DataLoader(sents, batch_size=64, shuffle=True, drop_last=True)
    train_loss = losses.ContrastiveTensionLossInBatchNegatives(model)

    warmup_steps = math.ceil(len(train_dataloader) * 25 * 0.1)  

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=25,
        warmup_steps=warmup_steps,
        show_progress_bar=True
    )
    model.save(out_path)


def evaluate(model, df, out_name):
    df1 = pd.DataFrame()
    df1["PairID"] = df["PairID"]
    sent1 = []
    sent2 = []
    for i in df["Text"].tolist():
        a,b = i.split("\n")
        sent1 += [a]
        sent2 += [b]
    embeddings1 = model.encode(sent1,batch_size=64,convert_to_numpy=True)
    embeddings2 = model.encode(sent2, batch_size=64,convert_to_numpy=True)
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    df1["Pred_Score"] = cosine_scores
    df1.to_csv(out_name, index=False)

def predict(args):
    trained_model_path = args.out_dir
    eval_file = args.eval_file
    out_file = args.out_file
    df = pd.read_csv(eval_file)
    model = SentenceTransformer(trained_model_path)
    evaluate(model, df, out_file)





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CT')
    parser.add_argument('-d', '--data_path', default="./data/eng_train.csv", type=str, 
                                    help="Path to data")
    parser.add_argument('-l', '--lang', default="eng", type=str, 
                                    help="languages eg. eng, mar, tel, amh")
    parser.add_argument('-m', '--model', default="bert-base-uncased", type=str, 
                                    help="pretrained huggingface mpdel name")
    parser.add_argument('-o', '--out_dir', default="./out/eng/", type=str, 
                                    help="path to save trained model")
    parser.add_argument('-e', '--eval_file', default="./data/eng_dev.csv", type=str, 
                                    help="Path to eval data")
    parser.add_argument('-s', '--out_file', default="./pred_eng_b.csv", type=str, 
                                    help="Path to out file")
    
    
    args = parser.parse_args()
    train()
    predict()
