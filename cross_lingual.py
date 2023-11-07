import pandas as pd
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.readers import InputExample
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import models, datasets, losses


def get_sentences(lang, sep="\n", labels=True):
    data_path = f"Semantic_Relatedness_SemEval2024/Track A/{lang}/{lang}_train.csv"
    all_s = pd.read_csv(data_path)
    sents = []
    for s in all_s["Text"].tolist():
        sents.append(s.split(sep))
    if labels:
        return sents,all_s["Score"].tolist()
    else:
        return sents


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
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2)/2)
    df1["Pred_Score"] = cosine_scores
    df1.to_csv(out_name, index=False)

def train_finetuning():
    langs = ["eng", "esp", "ary", "arq"]
    model_name = "bert-base-multilingual-cased"
    for i in range(len(langs)):
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        sents,sr = [], []
        xx = []
        name = []
        for j in range(len(langs)):
            if (i!=j):
                s,c = get_sentences(langs[j])
                sents += s
                sr += c
                name += [langs[j]]
        name = "_".join(name)
        for s,c in zip(sents, sr):
            xx.append(InputExample(texts=s, label=c))
        train_dataloader = DataLoader(xx, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model=model)
        warmup_steps = math.ceil(len(train_dataloader) * 20  * 0.1) 
        model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=20,
            warmup_steps=warmup_steps,
            output_path=f"./out/{name}")
        

def TSDAE():
    langs = ["eng", "esp", "ary", "arq"]
    for l in langs:
        model_name = "out/eng_ary_arq"
        model = SentenceTransformer(model_name)
        sents = get_sentences(l, False)
        train_dataset = datasets.DenoisingAutoEncoderDataset(sents)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=10,
            show_progress_bar=True
        )
        model.save(f'out2/{l}_tsdae')



if __name__=='__main__':
    train_finetuning()
    TSDAE()
    langs = ["eng", "esp", "ary", "arq"]
    for i in range(4):
        df = pd.read_csv(f"/kaggle/working/Semantic_Relatedness_SemEval2024/Track A/{langs[i]}/{langs[i]}_dev.csv")
        name = "./out/"+"_".join([langs[j] for j in range(4) if j!=i])
        model_finetuned = SentenceTransformer(name)
        evaluate(model_finetuned, df, f"pred_{langs[i]}_c.csv")
        model_tsdae = SentenceTransformer(f"out2/{langs[i]}_tsdae")
        evaluate(model_tsdae, df, f"pred_{langs[i]}_c1.csv")
