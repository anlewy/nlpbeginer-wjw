import pandas as pd
import time
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from torch.utils.data import TensorDataset, random_split

df = pd.read_csv('data/train.tsv', sep='\t')
time_length = []

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

for i in tqdm(range(32, 512, 32)):
    texts = df['Phrase'][:i].tolist()
    time_start = time.time()
    texts_tokenized = tokenizer(texts, padding=True, return_tensors='pt')
    texts_embed = model(**texts_tokenized)
    time_end = time.time()
    time_length.append(time_end-time_start)
    print("i = ", i, "  time gap = ", time_end-time_start)

plt.plot(time_length)
