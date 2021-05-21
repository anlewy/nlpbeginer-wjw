import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def prepare_data():
    tfidf_vector = TfidfVectorizer(max_features=768)
    count_vector = CountVectorizer(max_features=768)
    data = pd.read_csv('data/waimai_10k.csv')
    texts = data['review']
    label = data['label']

    # label
    if not os.path.exists('data/label.npy'):
        label = np.array(label)
        np.save('data/label.npy', label)

    # tfidf向量化
    if not os.path.exists('data/X_tfidf.npy'):
        X_tfidf = tfidf_vector.fit_transform(texts).toarray()
        np.save('data/X_tfidf.npy', X_tfidf)

    # 计数向量化
    if not os.path.exists('data/X_count.npy'):
        X_count = count_vector.fit_transform(texts).toarray()
        np.save('data/X_count.npy', X_count)

    # bert的向量化
    if not os.path.exists('data/X_bert.npy'):
        X_bert = np.zeros((0, 768))
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        bert_model = BertModel.from_pretrained('bert-base-chinese')
        for i in tqdm(range(0, len(data), 100)):
            ii = min(i + 100, len(data))
            X_token_i = tokenizer(list(data['review'][i:ii]), return_tensors='pt', padding=True)
            X_embed_i = bert_model(**X_token_i)
            print(X_embed_i.pooler_output.detach().numpy().shape)
            X_bert = np.vstack((X_bert, X_embed_i.pooler_output.detach().numpy()))
            del X_token_i
            del X_embed_i
            gc.collect()
        np.save('data/X_bert.npy', X_bert)
