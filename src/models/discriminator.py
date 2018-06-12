import tensorflow as tf
import numpy as np
from .net_utils import *


class Discriminator:
    def __init__(self, sent_len, vocab_size, embedding_size, init_embedding, batch_size):
        self.sent_len = sent_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        with tf.variable_scope("Discriminator"):
            self.build_net(init_embedding)

    def build_net(self, init_embedding):
        self.sent1_real = tf.placeholder(tf.int32, [self.batch_size, self.sent_len], name="sent1_real")
        self.sent2_real = tf.placeholder(tf.int32, [self.batch_size, self.sent_len], name="sent2_real")
        self.sent1_fake = tf.placeholder(tf.int32, [self.batch_size, self.sent_len], name="sent1_fake")
        self.sent2_fake = tf.placeholder(tf.int32, [self.batch_size, self.sent_len], name="sent2_fake")
        # self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        x_real = tf.concat([self.sent1_real, self.sent2_real], -1)
        x_fake = tf.concat([self.sent1_fake, self.sent2_fake], -1)

        with tf.variable_scope("embedding"):
            self.embedding = weight_variable(name="embedding",
                                             shape=[self.vocab_size, self.embedding_size],
                                             initial_value=init_embedding)
            if init_embedding:
                print("Discriminator gotta initial embedding!")

        x_real_embed = tf.nn.embedding_lookup(self.embedding, x_real)
        x_fake_embed = tf.nn.embedding_lookup(self.embedding, x_fake)

        logits_real = self.logits(x_real_embed)
        logits_fake = self.logits(x_fake_embed)

        # compute loss and reward
        y_real = tf.one_hot(tf.constant(1, tf.int32, [self.batch_size]), depth=2)
        y_fake = tf.one_hot(tf.constant(0, tf.int32, [self.batch_size]), depth=2)

        loss_real = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_real, logits=logits_real)
        loss_fake = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_fake, logits=logits_fake)

        self.loss = tf.reduce_mean(loss_real) + tf.reduce_mean(loss_fake)
        self.reward = tf.nn.softmax(logits_fake)[:, 1]

        d_params = tf.trainable_variables("Discriminator")
        grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, d_params), 5.0)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(grad, d_params))

    def logits(self, sentence):
        with tf.variable_scope("score", reuse=tf.AUTO_REUSE):
            cell_fw = tf.nn.rnn_cell.LSTMCell(100)
            cell_bw = tf.nn.rnn_cell.LSTMCell(100)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                      sentence, dtype=tf.float32,
                                                                      scope="bi-lstm")
            h = tf.concat([state_fw[0], state_bw[0]], -1)
            logits = mlp(h, [128, 2], [tf.nn.relu, tf.nn.relu, tf.nn.relu, None], "score")
            return logits

    def train(self, sess, sent1_real, sent2_real, sent1_fake, sent2_fake):
        _, loss = sess.run(fetches=[self.train_op, self.loss],
                           feed_dict={self.sent1_real: sent1_real,
                                      self.sent2_real: sent2_real,
                                      self.sent1_fake: sent1_fake,
                                      self.sent2_fake: sent2_fake})
        return loss
