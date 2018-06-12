#coding=utf-8
import pickle

vocab = pickle.load(open("vocab.pkl", "rb"))

dataset = []

with open("p7.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        i = 0
        line = line.split()
        d = []
        for l in line:
            sent_vec = []
            for w in l:
                sent_vec.append(vocab[w])
            d.append(sent_vec)
            if i == 1:
                i = 0
                assert len(d) == 2
                dataset.append(d)
                d = []
            else:
                i += 1

pickle.dump(dataset, open("dataset7.pkl", "wb"))

