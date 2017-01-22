from collections  import namedtuple as nt
from functools import reduce
from operator import mul
import tensorflow as tf
import numpy as np
import resnet_arch
import pickle
import os

envconfig = nt("envconfig",
                   "train_dir,"
                   "test_dir,"
                   "save_to_dir,"
                   "save_steps,"
                   "warmup_learning_rate,"
                   "learning_steps")

env = envconfig(
    train_dir = "tf-data/train/",
    test_dir = "tf-data/test/",
    save_to_dir = "tf-data/model/",
    save_steps = 100,
    warmup_learning_rate = 0.01,
    learning_steps = [
        (400,     0.01),
        (10000,   0.1),
        (20000,   0.01),
        (40000,   0.001),
        (1000000, 0.0001)])

global_params = resnet_arch.nn_params(
    learning_rate = 0.1,
    num_resunits_per_block = 2,
    batch_size = 128,
    num_labels = 43,
    weight_decay_rate = 0.0001,
    momentum_term = 0.9,
    depths = [16, 64, 128, 256],
    image_shape = [32, 32, 3])

def save_data_as_tfrecords(images, labels, file_path):
    num_examples = images.shape[0]
    assert num_examples == labels.shape[0]
    with tf.python_io.TFRecordWriter(file_path) as w:
        for i in range(num_examples):
            if i % 1000:
                print("{0}-th image is proccessed".format(i))
            image = map(int, images[i].flatten().tolist())
            label = int(labels[i])
            feature_img = tf.train.Feature(int64_list=tf.train.Int64List(value=image))
            feature_lbl = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            example = tf.train.Example(features=tf.train.Features(feature=
                {"image": feature_img, "label": feature_lbl}))
            w.write(example.SerializeToString())

def create_data_reader(
        directory,
        shape=global_params.image_shape,
        batch_size=global_params.batch_size,
        num_labels=global_params.num_labels,
        mode="train"):
    file_pattern = os.path.join(directory, "*.tfrecords")
    files = tf.gfile.Glob(file_pattern)
    queue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    length = reduce(mul, shape)
    _, record = reader.read(queue)
    features = {
        "image": tf.FixedLenFeature([length], tf.int64),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    example = tf.parse_single_example(record, features=features)
    image = tf.cast(tf.reshape(example["image"], shape), dtype=tf.float32)
    label = tf.cast(example["label"], dtype=tf.int32)
    image = tf.image.per_image_standardization(image)
    queue_shape = [shape, [1]]
    if mode == "train":
        num_threads = 8
        example_queue = tf.RandomShuffleQueue(
            capacity=num_threads*batch_size,
            min_after_dequeue=num_threads//2,
            dtypes=[tf.float32, tf.int32],
            shapes=[[32,32,3], [1]])
    else:
        num_threads = 1
        example_queue = tf.FIFOQueue(
            capacity=num_threads*batch_size+2,
            dtypes=[tf.float32, tf.int32],
            shapes=[[32,32,3], [1]])
    example_enqueue = example_queue.enqueue([image, [label]])
    runner = tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue] * num_threads)
    tf.train.add_queue_runner(runner)
    batch_images, batch_labels = example_queue.dequeue_many(batch_size)
    batch_labels_shape = [batch_size, 1]
    batch_labels = tf.reshape(batch_labels, batch_labels_shape)
    indices = tf.range(0, batch_size, 1)
    indices = tf.reshape(indices, batch_labels_shape)
    indices = tf.concat(1, [indices, batch_labels])
    batch_labels = tf.sparse_to_dense(indices, [batch_size, num_labels], 1.0, 0.0)
    if mode != "train":
        tf.summary.image("images", batch_images)
    return batch_images, batch_labels
    
def train(params):
    x_train, y_train = create_data_reader(env.train_dir)
    rn = resnet_arch.ResNet(x_train, y_train, params, mode="train")
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
                config=config) as sess:
        while not sess.shoudl_stop():
            sess.run(rn.train)

def main(_):
    train(global_params)
    #with tf.device("/gpu:0"):
    #    train(global_params)

if __name__ == '__main__':
    tf.app.run()
