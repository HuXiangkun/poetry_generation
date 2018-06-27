import tensorflow as tf
from .net_utils import *
import numpy as np


class ROLLOUT:
    def __init__(self, lstm, update_rate, sent_len, batch_size):
        self.lstm = lstm
        self.update_rate = update_rate
        self.gen_len = sent_len - 1
        self.batch_size = batch_size
        self.rnn_h_dim = self.lstm.rnn_h_dim
        self.vocab_size = self.lstm.vocab_size

        # start token for two sentences
        self.w1 = tf.placeholder(tf.int64, [self.batch_size], "w1")
        self.w2 = tf.placeholder(tf.int64, [self.batch_size], "w2")

        self.x1 = tf.placeholder(tf.int64, [self.batch_size, self.gen_len], "x1")
        self.x2 = tf.placeholder(tf.int64, [self.batch_size, self.gen_len], "x2")

        self.given_num1 = tf.placeholder(tf.int32, [], "given_num1")
        self.given_num2 = tf.placeholder(tf.int32, [], "given_num2")

        self.embedding = tf.identity(self.lstm.embedding)

        w1_embed = tf.nn.embedding_lookup(self.embedding, self.w1)
        w2_embed = tf.nn.embedding_lookup(self.embedding, self.w2)

        self.rnn_cell = self.create_lstm_cell()
        self.output_unit = self.create_output_unit()

        x1_arr = tf.TensorArray(tf.int64, size=self.gen_len).unstack(tf.transpose(self.x1, [1, 0]))
        x2_arr = tf.TensorArray(tf.int64, size=self.gen_len).unstack(tf.transpose(self.x2, [1, 0]))

        gen_token_arr1 = tf.TensorArray(tf.int64, size=self.gen_len, infer_shape=True)
        gen_token_arr2 = tf.TensorArray(tf.int64, size=self.gen_len, infer_shape=True)

        def _recurrence_put_1(i, x, h, c, token_arr):
            new_h, new_c = self.rnn_cell(x, (h, c))
            next_token = x1_arr.read(i)
            new_x = tf.nn.embedding_lookup(self.embedding, next_token)
            token_arr = token_arr.write(i, next_token)
            return [i+1, new_x, new_h, new_c, token_arr]

        _, x, h, c, self.gen_x1 = tf.while_loop(cond=lambda i, _1, _2, _3, _4: i < self.given_num1,
                                                body=_recurrence_put_1,
                                                loop_vars=[0, w1_embed,
                                                           tf.zeros([self.batch_size, self.rnn_h_dim]),
                                                           tf.zeros([self.batch_size, self.rnn_h_dim]),
                                                           gen_token_arr1])

        def _recurrence_gen_1(i, x, h, c, token_arr):
            new_h, new_c = self.rnn_cell(x, (h, c))
            logits = self.output_unit(new_h)
            next_token = tf.squeeze(tf.multinomial(logits, 1))
            new_x = tf.nn.embedding_lookup(self.embedding, next_token)
            token_arr = token_arr.write(i, next_token)
            return [i + 1, new_x, new_h, new_c, token_arr]

        _, x, h, c, self.gen_x1 = tf.while_loop(cond=lambda i, _1, _2, _3, _4: i < self.gen_len,
                                                body=_recurrence_gen_1,
                                                loop_vars=[self.given_num1, x, h, c, self.gen_x1])
        self.gen_x1 = self.gen_x1.stack()
        self.gen_x1 = tf.transpose(self.gen_x1, [1, 0])

        def _recurrence_put_2(i, x, h, c, token_arr):
            new_h, new_c = self.rnn_cell(x, (h, c))
            next_token = x2_arr.read(i)
            new_x = tf.nn.embedding_lookup(self.embedding, next_token)
            token_arr = token_arr.write(i, next_token)
            return [i+1, new_x, new_h, new_c, token_arr]

        _, x, h, c, self.gen_x2 = tf.while_loop(cond=lambda i, _1, _2, _3, _4: i < self.given_num2,
                                                body=_recurrence_put_2,
                                                loop_vars=[0, w2_embed,
                                                           tf.zeros([self.batch_size, self.rnn_h_dim]),
                                                           tf.zeros([self.batch_size, self.rnn_h_dim]),
                                                           gen_token_arr2])

        def _recurrence_gen_2(i, x, h, c, token_arr):
            new_h, new_c = self.rnn_cell(x, (h, c))
            logits = self.output_unit(new_h)
            next_token = tf.squeeze(tf.multinomial(logits, 1))
            new_x = tf.nn.embedding_lookup(self.embedding, next_token)
            token_arr = token_arr.write(i, next_token)
            return [i + 1, new_x, new_h, new_c, token_arr]

        _, _, h, c, self.gen_x2 = tf.while_loop(cond=lambda i, _1, _2, _3, _4: i < self.gen_len,
                                                body=_recurrence_gen_2,
                                                loop_vars=[self.given_num2, x, h, c, self.gen_x2])
        self.gen_x2 = self.gen_x2.stack()
        self.gen_x2 = tf.transpose(self.gen_x2, [1, 0])

    def create_lstm_cell(self):
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wo = tf.identity(self.lstm.Wo)
        self.Uo = tf.identity(self.lstm.Uo)
        self.bo = tf.identity(self.lstm.bo)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

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

    def create_output_unit(self):
        self.Ws = tf.identity(self.lstm.Ws)
        self.bs = tf.identity(self.lstm.bs)

        def unit(h):
            logits = tf.matmul(h, self.Ws) + self.bs
            return logits
        return unit

    def get_reward(self, sess, w1, w2, x1, x2, mc_num, discriminator):
        rewards1 = []
        rewards2 = []
        for i in range(mc_num):
            for given_num1 in range(1, self.gen_len+1):
                feed_dict = {self.w1: w1, self.w2: w2, self.x1: x1, self.x2: x2,
                             self.given_num1: given_num1, self.given_num2: self.gen_len}
                gen_x1, gen_x2 = sess.run(fetches=[self.gen_x1, self.gen_x2], feed_dict=feed_dict)
                x_fake = np.concatenate([np.expand_dims(w1, -1), gen_x1, np.expand_dims(w2, -1), gen_x2], -1)

                r = discriminator.get_reward(sess, x_fake)
                if i == 0:
                    rewards1.append(r)
                else:
                    rewards1[given_num1-1] += r

            for given_num2 in range(1, self.gen_len+1):
                feed_dict = {self.w1: w1, self.w2: w2, self.x1: x1, self.x2: x2,
                             self.given_num1: self.gen_len, self.given_num2: given_num2}
                gen_x1, gen_x2 = sess.run(fetches=[self.gen_x1, self.gen_x2], feed_dict=feed_dict)
                x_fake = np.concatenate([np.expand_dims(w1, -1), gen_x1, np.expand_dims(w2, -1), gen_x2], -1)

                r = discriminator.get_reward(sess, x_fake)
                if i == 0:
                    rewards2.append(r)
                else:
                    rewards2[given_num2-1] += r

        rewards1 = np.array(rewards1) / (1.0 * mc_num)
        rewards2 = np.array(rewards2) / (1.0 * mc_num)
        return rewards1, rewards2

    def update_params(self):
        # update embedding
        self.embedding = tf.identity(self.lstm.embedding)

        # update lstm vars
        self.Wi = self.update_rate*self.Wi + (1.-self.update_rate)*tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate*self.Ui + (1.-self.update_rate)*tf.identity(self.lstm.Ui)
        self.bi = self.update_rate*self.bi + (1.-self.update_rate)*tf.identity(self.lstm.bi)

        self.Wf = self.update_rate*self.Wf + (1.-self.update_rate)*tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate*self.Uf + (1.-self.update_rate)*tf.identity(self.lstm.Uf)
        self.bf = self.update_rate*self.bf + (1.-self.update_rate)*tf.identity(self.lstm.bf)

        self.Wo = self.update_rate*self.Wo + (1.-self.update_rate)*tf.identity(self.lstm.Wo)
        self.Uo = self.update_rate*self.Uo + (1.-self.update_rate)*tf.identity(self.lstm.Uo)
        self.bo = self.update_rate*self.bo + (1.-self.update_rate)*tf.identity(self.lstm.bo)

        self.Wc = self.update_rate*self.Wc + (1.-self.update_rate)*tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate*self.Uc + (1.-self.update_rate)*tf.identity(self.lstm.Uc)
        self.bc = self.update_rate*self.bc + (1.-self.update_rate)*tf.identity(self.lstm.bc)

        # update output vars
        self.Ws = self.update_rate*self.Ws + (1.-self.update_rate)*tf.identity(self.lstm.Ws)
        self.bs = self.update_rate*self.bs + (1.-self.update_rate)*tf.identity(self.lstm.bs)
