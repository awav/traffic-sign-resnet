import collections.namedtuple as nt
import tensorflow as tf
import numpy as np
import resnet
import pickle
import os

envconfig = nt("env_config",
                   "train_file,
                    test_file,
                    save_to_dir,
                    save_step,
                    warmup_learning_rate,
                    learning_steps")

env = envconfig(
    train_file = "./data/train.tfrecords",
    test_file = "./data/test.tfrecords",
    save_to_dir = "./resnet_model/",
    save_steps = 100,
    warmup_learning_rate = 0.01,
    learning_steps = [
        (400,     0.01),
        (10000,   0.1),
        (20000,   0.01),
        (40000,   0.001),
        (1000000, 0.0001)])

global_params = resnet.nn_params(
    learning_rate = 0.1,
    num_resunits_per_block = 2,
    batch_size = 128,
    num_labels = 43,
    weight_decay_rate = 0.0001,
    momentum_term = 0.9,
    depths = [16, 64, 128, 256])

def convert_traffic_sign_data(images, labels, directory, name):
    num_examples = images.shape[0]
    filename = os.path.join(directory, name, ".tfrecords")
    with tf.python_io.TFRecordWriter(filename) as w:
        for i in xrange(num_examples):
            

def traffic_sign_data(dir, ratio=0.9):
    with open(f, mode='rb') as fd:
        train_upd = pickle.load(fd)
        inputs, labels = train_upd['features'], train_upd['labels']
        return train_inputs, train_labels, valid_inputs, valid_labels
    

def train(params):
    if not os.path.exists(env.save_to_dir):
        os.makedirs(env.save_to_dir)
    x_train, y_train, x_valid, y_valid = traffic_sign_data(env.train_path)
    rn = resnet.ResNet(x_train, y_train, params, mode="train")
    rn.build()
    with tf.variable_scope("accuracy"):
        true_predictions = tf.argmax(y_train, axis=1)
        predictions = tf.argmax(rn.inference, axis=1)
        hits = tf.cast(tf.equal(predictions, true_predictions), tf.float32)
        accuracy = tf.reduce_mean(hits)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    class LearningRateHook(tf.train.SessionRunHook):
        def begin(self):
            self.learning_rate = env.warmup_learning_rate
        def before_run(self, run_context):
            feed_dict = {rn.learning_rate : self.learning_rate}
            return tf.train.SessionRunArgs(rn.global_step, feed_dict=feed_dict)
        def after_run(self, run_context, run_values):
            curr_step = run_values.results
            for step_change, rate  in env.learning_steps:
                if curr_step < step_change:
                    self.learning_rate = rate
                    break
    summary = tf.summary.merge([rn.summary, accuracy_summary])
    summary_hook = tf.train.SummarySaverHook(
                       save_steps=env.save_step,
                       output_dir=save_to_dir,
                       summary_op=summary)
    logging_hook = tf.train.LoggingTensorHook(
                       every_n_iter=env.save_step,
                       tensors={"accuracy": accurancy,
                                "loss":     rn.loss,
                                "step":     rn.global_step})
    learn_rate_hook = LearningRateHook()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.MonitoredTrainingSession(
                checkpoint_dir=env.save_to_dir,
                save_checkpoint_secs=600,
                save_summaries_steps=None,
                hooks=[log_hook, learning_rate_hook],
                chief_only_hooks=[summary_hook],
                config=config)
            as sess:
        while not sess.shoudl_stop():
             sess.run(rn.train)

def main():
    with tf.device("/gpu:0"):
        train(global_params)

if __name__ == "__main__":
    tf.app.run()
