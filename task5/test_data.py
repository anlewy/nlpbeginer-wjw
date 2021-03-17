from task5.data import get_data_iter
from tqdm import tqdm

train_iter, valid_iter, test_iter, TEXT = get_data_iter()
print("len(TEXT.vocab) is \n", len(TEXT.vocab))

for batch in tqdm(train_iter):
    poetry, poetry_lens = batch.text
    print("poetry: {}, \n\n lens: {}".format(poetry, poetry_lens))
