from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

texts = ['my name is wjw', 'wjw love wy very much wowowo']
encoded_texts = tokenizer(texts, return_tensors='pt', padding=True)
embeded_texts = bert(**encoded_texts)
