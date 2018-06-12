import tensorflow as tf
from .net_utils import *
import numpy as np


class Generator:
    def __init__(self, sent_len, vocab_size, embedding_size, init_embedding, rnn_h_dim, batch_size):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.rnn_h_dim = rnn_h_dim

        # length of generated sentence
        self.gen_len = sent_len - 1

        with tf.variable_scope("Generator"):
            self.build_net(init_embedding)

    def build_net(self, init_embedding):
        # start token for two sentences
        self.w1 = tf.placeholder(tf.int32, [self.batch_size], "w1")
        self.w2 = tf.placeholder(tf.int32, [self.batch_size], "w2")

        self.x1 = tf.placeholder(tf.int32, [self.batch_size, self.gen_len], "x1")
        self.x2 = tf.placeholder(tf.int32, [self.batch_size, self.gen_len], "x2")

        self.reward = tf.placeholder(tf.float32, [self.batch_size], "reward")

        with tf.variable_scope("embedding"):
            self.embedding = weight_variable(name="embedding",
                                             shape=[self.vocab_size, self.embedding_size],
                                             initial_value=init_embedding)
            if init_embedding:
                print("Generator gotta initial embedding!")

        w1_embed = tf.nn.embedding_lookup(self.embedding, self.w1)
        w2_embed = tf.nn.embedding_lookup(self.embedding, self.w2)

        x1_embed = tf.nn.embedding_lookup(self.embedding, self.x1)
        x2_embed = tf.nn.embedding_lookup(self.embedding, self.x2)

        self.rnn_cell = self.create_lstm_cell(self.rnn_h_dim)

        gen_logits_arr1 = tf.TensorArray(tf.float32, size=self.gen_len, infer_shape=True)
        gen_token_arr1 = tf.TensorArray(tf.int64, size=self.gen_len, infer_shape=True)

        gen_logits_arr2 = tf.TensorArray(tf.float32, size=self.gen_len, infer_shape=True)
        gen_token_arr2 = tf.TensorArray(tf.int64, size=self.gen_len, infer_shape=True)

        def _gen_step(i, x, h, c, logits_arr, token_arr):
            new_h, new_c = self.rnn_cell(x, (h, c))
            logits = self.sample_token_logits(new_h)
            next_token = tf.squeeze(tf.multinomial(logits, 1))
            new_x = tf.nn.embedding_lookup(self.embedding, next_token)
            logits_arr = logits_arr.write(i, logits)
            token_arr = token_arr.write(i, next_token)
            return [i + 1, new_x, new_h, new_c, logits_arr, token_arr]

        # Generate first sentence
        _, _, final_h, final_c, self.gen_logits_arr1, self.gen_token_arr1 = \
            tf.while_loop(cond=lambda i, _1, _2, _3, _4, _5: i < self.gen_len,
                          body=_gen_step,
                          loop_vars=[0, w1_embed,
                                     tf.zeros([self.batch_size, self.rnn_h_dim]),
                                     tf.zeros([self.batch_size, self.rnn_h_dim]),
                                     gen_logits_arr1, gen_token_arr1])
        # Generate second sentence
        _, _, _, _, self.gen_logits_arr2, self.gen_token_arr2 = \
            tf.while_loop(cond=lambda i, _1, _2, _3, _4, _5: i < self.gen_len,
                          body=_gen_step,
                          loop_vars=[0, w2_embed, final_h, final_c,
                                     gen_logits_arr2, gen_token_arr2])
        self.gen_token_arr1 = tf.transpose(self.gen_token_arr1.stack(), [1, 0])
        self.gen_token_arr2 = tf.transpose(self.gen_token_arr2.stack(), [1, 0])

        # Pretrain a Language Model using training data
        pretrain_logits_arr1 = tf.TensorArray(tf.float32, size=self.gen_len, infer_shape=True)
        pretrain_logits_arr2 = tf.TensorArray(tf.float32, size=self.gen_len, infer_shape=True)

        x1_arr = tf.TensorArray(tf.float32, size=self.gen_len).unstack(tf.transpose(x1_embed, [1, 0, 2]))
        x2_arr = tf.TensorArray(tf.float32, size=self.gen_len).unstack(tf.transpose(x2_embed, [1, 0, 2]))

        def _pretrain_gen_step(i, x, h, c, logits_arr, x_arr):
            new_h, new_c = self.rnn_cell(x, (h, c))
            logits = self.sample_token_logits(new_h)
            new_x = x_arr.read(i)
            logits_arr = logits_arr.write(i, logits)
            return [i + 1, new_x, new_h, new_c, logits_arr, x_arr]

        _, _, final_h_pretrain, final_c_pretrain, self.pretrain_logits_arr1, _ = \
            tf.while_loop(cond=lambda i, _1, _2, _3, _4, _5: i < self.gen_len,
                          body=_pretrain_gen_step,
                          loop_vars=[0, w1_embed,
                                     tf.zeros([self.batch_size, self.rnn_h_dim]),
                                     tf.zeros([self.batch_size, self.rnn_h_dim]),
                                     pretrain_logits_arr1, x1_arr])
        _, _, _, _, self.pretrain_logits_arr2, _ = \
            tf.while_loop(cond=lambda i, _1, _2, _3, _4, _5: i < self.gen_len,
                          body=_pretrain_gen_step,
                          loop_vars=[0, w1_embed, final_h_pretrain,
                                     final_c_pretrain,
                                     pretrain_logits_arr2, x2_arr])

        labels_1 = tf.one_hot(self.x1, depth=self.vocab_size, dtype=tf.float32)
        labels_2 = tf.one_hot(self.x2, depth=self.vocab_size, dtype=tf.float32)

        self.g_params = tf.trainable_variables("Generator")

        # pre-train loss
        self.pretrain_loss = self.compute_loss(labels_1, self.pretrain_logits_arr1.stack(),
                                               labels_2, self.pretrain_logits_arr2.stack(), False)
        pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), 5.)
        self.pretrain_op = tf.train.AdamOptimizer().apply_gradients(zip(pretrain_grad, self.g_params))

        # RL train loss
        self.rl_train_loss = self.compute_loss(labels_1, self.pretrain_logits_arr1.stack(),
                                               labels_2, self.pretrain_logits_arr2.stack(), True)
        rl_grad, _ = tf.clip_by_global_norm(tf.gradients(self.rl_train_loss, self.g_params), 5.)
        self.rl_train_op = tf.train.AdamOptimizer().apply_gradients(zip(rl_grad, self.g_params))

    def create_lstm_cell(self, num_unit):
        with tf.variable_scope("LSTM_Cell", reuse=tf.AUTO_REUSE):
            self.Wi = weight_variable("Wi", [self.embedding_size, num_unit])
            self.Ui = weight_variable("Ui", [num_unit, num_unit])
            self.bi = bias_variable("bi", [num_unit])

            self.Wf = weight_variable("Wf", [self.embedding_size, num_unit])
            self.Uf = weight_variable("Uf", [num_unit, num_unit])
            self.bf = bias_variable("bf", [num_unit])

            self.Wo = weight_variable("Wo", [self.embedding_size, num_unit])
            self.Uo = weight_variable("Uo", [num_unit, num_unit])
            self.bo = bias_variable("bo", [num_unit])

            self.Wc = weight_variable("Wc", [self.embedding_size, num_unit])
            self.Uc = weight_variable("Uc", [num_unit, num_unit])
            self.bc = bias_variable("bc", [num_unit])

        def cell(x, state):
                h, c = state
                # Input gate
                i = tf.sigmoid(
                    tf.matmul(x, self.Wi) +
                    tf.matmul(h, self.Ui) + self.bi
                )
                # Forget gate
                f = tf.sigmoid(
                    tf.matmul(x, self.Wf) +
                    tf.matmul(h, self.Uf) + self.bf
                )
                # Output gate
                o = tf.sigmoid(
                    tf.matmul(x, self.Wo) +
                    tf.matmul(h, self.Uo) + self.bo
                )
                # Cell memory
                new_c = i * tf.tanh(
                    tf.matmul(x, self.Wc) +
                    tf.matmul(h, self.Uc) + self.bc
                ) + f*c
                # Output new h
                new_h = o * tf.tanh(new_c)
                return new_h, new_c

        return cell

    def sample_token_logits(self, h):
        logits = mlp(h, [self.vocab_size, self.vocab_size], [tf.nn.relu, None], "sample_logits")
        return logits

    def compute_loss(self, label1, logits1, label2, logits2, with_reward):
        def _cross_entropy(elems):
            labels, logits = elems
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
            if with_reward:
                loss *= self.reward
            loss = tf.reduce_mean(loss)
            return [loss, logits]

        loss_sent1 = tf.map_fn(_cross_entropy,
                               [tf.transpose(label1, [1, 0, 2]), logits1])[0]
        loss_sent1 = tf.reduce_sum(loss_sent1)
        loss_sent2 = tf.map_fn(_cross_entropy,
                               [tf.transpose(label2, [1, 0, 2]), logits2])[0]
        loss_sent2 = tf.reduce_sum(loss_sent2)

        loss = (loss_sent1 + loss_sent2) / (self.gen_len * 2)
        return loss

    def generate(self, sess, w1, w2):
        gen_x1, gen_x2 = sess.run(fetches=[self.gen_token_arr1, self.gen_token_arr2],
                                feed_dict={self.w1: w1,
                                           self.w2: w2})
        sent1 = np.concatenate([np.expand_dims(w1, -1), gen_x1], axis=-1)
        sent2 = np.concatenate([np.expand_dims(w2, -1), gen_x2], axis=-1)
        return sent1, sent2, gen_x1, gen_x2

    def pretrain(self, sess, w1, w2, x1, x2):
        _, loss = sess.run(fetches=[self.pretrain_op, self.pretrain_loss],
                           feed_dict={self.w1: w1,
                                      self.w2: w2,
                                      self.x1: x1,
                                      self.x2: x2})
        return loss

    def rl_train(self, sess, w1, w2, x1, x2, reward):
        _, loss = sess.run(fetches=[self.rl_train_op, self.rl_train_loss],
                           feed_dict={self.w1: w1,
                                      self.w2: w2,
                                      self.x1: x1,
                                      self.x2: x2,
                                      self.reward: reward})
        return loss
