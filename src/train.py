from data_utils.batch_helper import get_batch
import pickle
from models.generator import Generator
import tensorflow as tf
from data_utils.batch_helper import get_batch
from models.discriminator import Discriminator
from ckpt_manage import CheckpointManager


tf.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.flags.DEFINE_integer("sent_len", 5, "sentence length")

FLAGS = tf.flags.FLAGS

if __name__ == "__main__":
    data = pickle.load(open("../data/dataset" + str(FLAGS.sent_len) + ".pkl", "rb"))
    vocab = pickle.load(open("../data/vocab.pkl", "rb"))

    print(f"Total Data : {len(data)}")
    print(f"Vocab size: {len(vocab)}")

    G = Generator(sent_len=FLAGS.sent_len,
                  vocab_size=len(vocab),
                  embedding_size=300,
                  init_embedding=None,
                  rnn_h_dim=150,
                  batch_size=FLAGS.batch_size)

    D = Discriminator(sent_len=FLAGS.sent_len,
                      vocab_size=len(vocab),
                      embedding_size=300,
                      init_embedding=None,
                      batch_size=FLAGS.batch_size)
    ckpt = CheckpointManager("model_100_" + str(FLAGS.sent_len))

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
        for epoch in range(50):
            batch_idx = 0
            for w1, w2, _, _, sent1, sent2 in get_batch(data, FLAGS.batch_size, shuffle=True):
                g_sent1, g_sent2, _, _ = G.generate(sess, w1, w2)
                loss = D.train(sess, sent1, sent2, g_sent1, g_sent2)
                batch_idx += 1
                if batch_idx % 500 == 0:
                    print(f"Pre-train D epoch {epoch}, batch {batch_idx}, loss {loss}")
            print("\n")
        ckpt.save(sess)

        print("Adversarial Training...")
        for epoch in range(100):
            batch_idx = 0
            for w1, w2, x1, x2, sent1, sent2 in get_batch(data, FLAGS.batch_size, shuffle=True):
                g_sent1, g_sent2, gen_x1, gen_x2 = G.generate(sess, w1, w2)

                # train generator one time
                reward = sess.run(fetches=[D.reward], feed_dict={D.sent1_fake: g_sent1,
                                                                 D.sent2_fake: g_sent2})
                g_loss = G.rl_train(sess, w1, w2, gen_x1, gen_x2, reward[0])

                # train discriminator five times
                d_loss = 0.
                for _ in range(5):
                    d_loss = D.train(sess, sent1, sent2, g_sent1, g_sent2)

                batch_idx += 1
                if batch_idx % 500 == 0:
                    print(f"Train epoch {epoch}, batch {batch_idx}, g_loss {g_loss}, d_loss {d_loss}")
            ckpt.save(sess)
            print("\n")
        ckpt.save(sess)
