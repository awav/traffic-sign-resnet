import tensorflow as tf
import sys
import resnet

flags = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', 'There are three options: train/valid/test')

if __name__ == "__main__":
    tf.app.run(main=resnet.main, argv=flags.mode)
