from models.generator import Generator
import pickle
from ckpt_manage import CheckpointManager
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    vocab = pickle.load(open("../data/vocab.pkl", "rb"))
    id2word = pickle.load(open("../data/id2word.pkl", "rb"))

    G = Generator(sent_len=5,
                  vocab_size=len(vocab),
                  embedding_size=300,
                  init_embedding=None,
                  rnn_h_dim=150,
                  batch_size=1000)
    ckpt = CheckpointManager("model_100_5")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if not ckpt.restore(sess):
            print("No Saved model found!!!")
            exit(1)
        while True:
            s = input("Input Your Name: ")
            w1 = [vocab[s[0]]]
            w2 = [vocab[s[1]]]
            w1 = np.stack(w1 * 1000)
            w2 = np.stack(w2 * 1000)
            sent1, sent2, _, _ = G.generate(sess, w1, w2)
            str1 = ""
            str2 = ""
            for w in sent1[0]:
                str1 += id2word[w]
            for w in sent2[0]:
                str2 += id2word[w]
            print(str1)
            print(str2)

