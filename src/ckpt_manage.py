import tensorflow as tf
import os


class CheckpointManager(object):
    def __init__(self, name):
        self.name = name
        self.ckpt_path = "../check_points"
        self.saver = tf.train.Saver()
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        self.model_ckpt_dir = os.path.join(self.ckpt_path, name)
        if not os.path.exists(self.model_ckpt_dir):
            os.mkdir(self.model_ckpt_dir)
        self.model_name = os.path.join(self.model_ckpt_dir, name + ".ckpt")

    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.model_ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        return False

    def save(self, sess):
        saved_path = self.saver.save(sess, self.model_name)
        return saved_path
