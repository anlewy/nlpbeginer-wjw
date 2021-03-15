import numpy as np
import re


def do_split(text):
    res = re.split(r'[\?,;\.\s]\s*', text)
    res2 = []
    for e in res:
        if len(e) >= 3:
            res2.append(e)
    return [word.lower() for word in res2]


class Ngram:
    def __init__(self, grams=1):
        self.length = 0
        self.word_seq = []
        self.word2id = {}
        self.id2word = {}
        self.gram_length = grams

    def get_grams(self, text):
        words = do_split(text)
        res = []
        for i in range(len(words)):
            if i + self.gram_length <= len(words):
                tmp = words[i]
                for j in range(1, self.gram_length):
                    tmp += ' ' + words[i + j]
                res.append(tmp)
        return res

    def fit(self, texts):
        word_set = set()
        for text in texts:
            word_set.update(self.get_grams(text))
        self.length = len(word_set)
        idx = 0
        for word in word_set:
            self.word2id[word] = idx
            self.id2word[idx] = word
            idx = idx + 1

    def transform(self, texts):
        n = len(texts)
        res = np.zeros((n, self.length))
        for i, text in enumerate(texts):
            words = self.get_grams(text)
            for word in words:
                if word in self.word2id:
                    res[i][self.word2id[word]] += 1
        return res

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


class BagOfWord(Ngram):
    def __init__(self):
        Ngram.__init__(self, 1)


# tf是单词在文档中的频率，是只需要一个文档文本就可以求出来的，
# idf涉及到总的文档，故不能仅仅通过单个文档，而是需要fit时提供的总体文档

# fit的作用是收录单词序列，以及求出每个出现的单词的idf值并储存
# transform的作用是求出单词在单个文档中出现的频率作为tf值

class Tfidf:
    def __init__(self):
        self.feature_names = []
        self.feature_num = 0
        self.idf_map = {}
        self.idf = []

    def fit(self, texts):
        word_set = set()
        for text in texts:
            words = set(do_split(text))
            word_set.update(words)
            for word in words:
                if word not in self.idf_map:
                    self.idf_map[word] = 1
                else:
                    self.idf_map[word] += 1

        self.feature_names = list(word_set)
        self.feature_names.sort()
        self.feature_num = len(word_set)
        for word in self.idf_map:
            self.idf_map[word] = np.log(self.feature_num / self.idf_map[word])
        for feature in self.feature_names:
            self.idf.append(self.idf_map[feature])

    def transform(self, texts):
        n = len(texts)
        res = np.zeros((n, self.feature_num))
        for i, text in enumerate(texts):
            words = do_split(text)
            ll = len(words)
            tf = {}
            for word in words:
                if word not in tf:
                    tf[word] = 1
                else:
                    tf[word] += 1

            for word in tf:
                tf[word] /= ll

            for j, feature in enumerate(self.feature_names):
                if feature not in tf:
                    res[i][j] = 0
                else:
                    res[i][j] = tf[feature] * self.idf_map[feature]
        return res.astype(np.float32)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
