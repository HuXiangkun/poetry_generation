import numpy as np
import random
import pickle


def get_batch(data, batch_size, shuffle=False):
    if shuffle:
        random.shuffle(data)
    batch_num = int(len(data) / batch_size)
    for i in range(batch_num):
        batch = data[i*batch_size: (i+1)*batch_size]
        batch = np.array(batch, dtype=np.int32)
        w1 = batch[:, :, 0][:, 0]
        w2 = batch[:, :, 0][:, 1]
        x1 = batch[:, :, 1:][:, 0]
        x2 = batch[:, :, 1:][:, 1]
        sent1 = batch[:, 0]
        sent2 = batch[:, 1]
        yield w1, w2, x1, x2, sent1, sent2



if __name__ == "__main__":
    data5 = pickle.load(open("../../data/dataset5.pkl", "rb"))
    for w1, w2, x1, x2, sent1, sent2 in get_batch(data5, 10):
        print(x1)
        print(x2)
        break
