from data_utils.batch_helper import get_batch
import pickle
from models.generator import Generator
import tensorflow as tf
from data_utils.batch_helper import get_batch
from models.discriminator2 import Discriminator
from ckpt_manage import CheckpointManager
from data_loader import load_embedding
import numpy as np
from models.rollout import ROLLOUT


tf.flags.DEFINE_integer("batch_size", 50, "batch size")
tf.flags.DEFINE_integer("sent_len", 5, "sentence length")

FLAGS = tf.flags.FLAGS

if __name__ == "__main__":
    data = pickle.load(open("../data/dataset" + str(FLAGS.sent_len) + ".pkl", "rb"))
    vocab = pickle.load(open("../data/vocab.pkl", "rb"))
    init_embedding = None # load_embedding(vocab, 300)

    print(f"Total Data : {len(data)}")
    print(f"Vocab size: {len(vocab)}")

    G = Generator(sent_len=FLAGS.sent_len,
                  vocab_size=len(vocab),
                  embedding_size=300,
                  init_embedding=init_embedding,
                  rnn_h_dim=150,
                  batch_size=FLAGS.batch_size)

    D = Discriminator(sequence_length=FLAGS.sent_len*2,
                      num_classes=2,
                      vocab_size=len(vocab),
                      embedding_size=300,
                      filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100],
                      l2_reg_lambda=0.2)
    ckpt = CheckpointManager("model_50_" + str(FLAGS.sent_len))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("Pre-training Generator...")
        for epoch in range(30):
            batch_idx = 0
            for w1, w2, x1, x2, _, _ in get_batch(data, FLAGS.batch_size, shuffle=True):
                loss = G.pretrain(sess, w1, w2, x1, x2)
                batch_idx += 1
                if batch_idx % 500 == 0:
                    print(f"Pre-train G epoch {epoch}, batch {batch_idx}, loss {loss}")
            print("\n")
        ckpt.save(sess)

        print("Pre-training Discriminator...")
        for epoch in range(40):
            batch_idx = 0
            for w1, w2, _, _, sent1, sent2 in get_batch(data, FLAGS.batch_size, shuffle=True):
                g_sent1, g_sent2, _, _ = G.generate(sess, w1, w2)
                loss = D.train_step(sess, np.concatenate([sent1, sent2], -1), np.concatenate([g_sent1, g_sent2], -1))
                batch_idx += 1
                if batch_idx % 500 == 0:
                    print(f"Pre-train D epoch {epoch}, batch {batch_idx}, loss {loss}")
            print("\n")
        ckpt.save(sess)

        rollout = ROLLOUT(G, 0.8, FLAGS.sent_len, FLAGS.batch_size)

        print("Adversarial Training...")
        for epoch in range(200):
            batch_idx = 0
            for w1, w2, x1, x2, sent1, sent2 in get_batch(data, FLAGS.batch_size, shuffle=True):
                g_sent1, g_sent2, gen_x1, gen_x2 = G.generate(sess, w1, w2)

                # train generator one time
                reward1, reward2 = rollout.get_reward(sess, w1, w2, gen_x1, gen_x2, 16, D)
                g_loss = G.rl_train(sess, w1, w2, gen_x1, gen_x2, reward1, reward2)

                rollout.update_params()
                # train discriminator five times
                d_loss = 0.
                for _ in range(5):
                    d_loss = D.train_step(sess, np.concatenate([sent1, sent2], -1),
                                          np.concatenate([g_sent1, g_sent2], -1))

                batch_idx += 1
                if batch_idx % 500 == 0:
                    print(f"Train epoch {epoch}, batch {batch_idx}, g_loss {g_loss}, d_loss {d_loss}")
            ckpt.save(sess)
            print("\n")
        ckpt.save(sess)
