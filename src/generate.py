from models.generator import Generator
from ckpt_manage import CheckpointManager
import tensorflow as tf
import numpy as np


class Poetry:
    def __init__(self, sent_len, vocab_size):
        graph = tf.Graph()
        with graph.as_default():
            self.G = Generator(sent_len=sent_len,
                               vocab_size=vocab_size,
                               embedding_size=300,
                               init_embedding=None,
                               rnn_h_dim=150,
                               batch_size=50)
            ckpt = CheckpointManager("model_50_" + str(sent_len))
            self.sess = tf.Session(graph=graph)
            ckpt.restore(self.sess)

    def generate(self, w1, w2):
        w1 = np.stack(w1 * 50)
        w2 = np.stack(w2 * 50)
        sent1, sent2, _, _ = self.G.generate(self.sess, w1, w2)
        return sent1, sent2

    # def __del__(self):
    #     self.sess.close()
