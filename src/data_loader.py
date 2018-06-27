import numpy as np


def load_embedding(vocab, size=300):
    random_bound = np.sqrt(3./size)
    init_embedding = np.zeros(shape=[len(vocab), size], dtype=np.float32)
    embedding_dict = {}
    with open("../data/sgns.sikuquanshu.word", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.split()
            assert len(line[1:]) == size
            embedding_dict[line[0]] = line[1:]
    c = 0
    for key in vocab.keys():
        if key in embedding_dict.keys():
            init_embedding[vocab[key]] = embedding_dict[key]
        else:
            c += 1
            init_embedding[vocab[key]] = np.random.uniform(-random_bound, random_bound, size)
    print("oov rate {}".format(float(c)/len(vocab)))
    return init_embedding
