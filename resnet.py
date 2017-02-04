from collections  import namedtuple as nt
from functools import reduce
from operator import mul
from sklearn.metrics import confusion_matrix as cfm
import tensorflow as tf
import numpy as np
import resnet_arch
import pickle
import time
import os

envconfig = nt("envconfig",
                   "train_dir,"
                   "valid_dir,"
                   "test_dir,"
                   "save_to_dir,"
                   "save_each_step,"
                   "warmup_learning_rate,"
                   "checkpoint_sec,"
                   "num_test_batches,"
                   "num_valid_batches,"
                   "valid_each_step,"
                   "learning_steps")

ENV = envconfig(
    train_dir = "tfdir/train/",
    valid_dir = "tfdir/valid/",
    test_dir = "tfdir/test/",
    save_to_dir = "tfdir/model/",
    num_test_batches = 42,
    num_valid_batches = 200,
    valid_each_step = 700,
    save_each_step = 100,
    warmup_learning_rate = 0.0001,
    checkpoint_sec = 180,
    learning_steps = [
        (10000,   0.1),
        (15000,   0.01),
        (25000,   0.001),
        (50000,   0.0001),
        (60000,   0.0001)])

def resnet_params(
            learning_rate=0.1,
            num_resunits_per_block=2,
            batch_size=128,
            num_labels=43,
            weight_decay_rate=0.0003,
            momentum_term=0.9,
            depths=[16, 64, 128, 256],
            image_shape=[32, 32, 3]):
    return resnet_arch.nn_params(
        learning_rate = learning_rate,
        num_resunits_per_block = num_resunits_per_block,
        batch_size = batch_size,
        num_labels = num_labels,
        weight_decay_rate = weight_decay_rate,
        momentum_term = momentum_term,
        depths = depths,
        image_shape = image_shape)

TRAIN_PARAMS = resnet_params()
TEST_PARAMS = resnet_params(batch_size=40)

def create_data_reader(
        directory,
        shape=TRAIN_PARAMS.image_shape,
        batch_size=TRAIN_PARAMS.batch_size,
        num_labels=TRAIN_PARAMS.num_labels,
        mode="train",
        pattern="train*"):
    file_pattern = os.path.join(directory, pattern+".tfrecords")
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
    tf.summary.image("batch_images", batch_images)
    return batch_images, batch_labels

def build_model(x, y, params, mode="train"):
    rn = resnet_arch.Resnet(x, y, params, mode=mode)
    rn.build()
    return rn

def evaluate(params, mode="test", confusion=False):
    if mode == "valid":
        data_dir = ENV.valid_dir
        num_batches = ENV.num_valid_batches
        pattern = "valid"
    elif mode == "test":
        data_dir = ENV.test_dir
        num_batches = ENV.num_test_batches
        pattern = "test"
    else:
        raise("Evaluate doesn't know passed mode")
    with tf.variable_scope("train"):
        x, y = create_data_reader(
                   data_dir,
                   mode="eval",
                   batch_size=params.batch_size,
                   pattern=pattern)
        rn = build_model(x, y, params, mode="eval")
    save_to = os.path.join(ENV.save_to_dir, mode)
    model_dir = os.path.join(ENV.save_to_dir, "train")
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(save_to)
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    tf.train.start_queue_runners(sess)
    while True:
        #time.sleep(30)
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if not checkpoint or not checkpoint.model_checkpoint_path:
            tf.logging.error("Model not found {0}".format(model_dir))
            continue
        tf.logging.info("Start loading checkpoint at {0}".format(model_dir))
        saver.restore(sess, checkpoint.model_checkpoint_path)
        correct, total, total_loss = 0.0, 0.0, 0.0
        cfm_pred = []
        cfm_true = []
        for i in range(num_batches):
            run = [rn.inference, rn.labels, rn.loss, rn.global_step, rn.summary]
            pred, true, loss, step, summary = sess.run(run)
            pred = np.argmax(pred, axis=1)
            true = np.argmax(true, axis=1)
            if confusion:
                cfm_pred = np.append(cfm_pred, pred)
                cfm_true = np.append(cfm_true, true)
            correct += np.sum(true == pred)
            total_loss += loss
            total += pred.shape[0]
        precision = correct / total
        summary_precision = tf.Summary()
        summary_precision.value.add(tag=mode+"_precision", simple_value=precision)
        writer.add_summary(summary_precision, step)
        summary_loss = tf.Summary()
        total_loss /= num_batches
        summary_loss.value.add(tag=mode+"_loss", simple_value=total_loss)
        writer.add_summary(summary_loss, step)
        msg = "{0}_precision: {1:.5f}, {0}_loss: {2:.5f}"
        tf.logging.info(msg.format(mode, precision, total_loss))
        writer.flush()
        if confusion:
            confusion_matrix = cfm(cfm_true, cfm_pred)
            cfm_path = ENV.save_to_dir + "/confusion_matrix.p"
            with open(cfm_path, "wb") as cfm_file:
                pickle.dump(confusion_matrix, cfm_file)
        break

def single_test(name, dataset_path, checkpoint_path, size=1):
    params = resnet_params(batch_size=size)
    with tf.variable_scope("train"):
        x, y = create_data_reader(
                     dataset_path,
                     mode="eval",
                     batch_size=params.batch_size,
                     pattern=name)
        #inputs_shape = np.append(params.batch_size, image.shape)
        #x = tf.placeholder(dtype=tf.float32, shape=inputs_shape)
        #y = tf.placeholder(dtype=tf.int32, shape=one_hot_label.shape)
        rn = build_model(x, y, params, mode="eval")
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    tf.train.start_queue_runners(sess)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    if not checkpoint or not checkpoint.model_checkpoint_path:
        msg = "Model is not found by passed checkpoint path {0}".format(checkpoint_path)
        raise ValueError(msg)
    saver.restore(sess, checkpoint.model_checkpoint_path)
    fetches = rn.inference
    inferences = sess.run(fetches)
    return inferences
    

def train(params):
    save_train_to = os.path.join(ENV.save_to_dir, "train")
    #save_train_to = ENV.save_to_dir
    #save_valid_to = os.path.join(ENV.save_to_dir, "valid")
    x, y = create_data_reader(ENV.train_dir)
    with tf.variable_scope("train"):
        rn = build_model(x, y, params)
        true_predictions = tf.argmax(y, axis=1)
        predictions = tf.argmax(rn.inference, axis=1)
        hits = tf.cast(tf.equal(predictions, true_predictions), tf.float32)
        accuracy = tf.reduce_mean(hits)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    #with tf.variable_scope("valid"):
    #    x_valid, y_valid = create_data_reader(ENV.train_dir, mode="valid")
    #    rn_valid = build_model(x_valid, y_valid, params, mode="eval")
    #    saver = tf.train.Saver()
    #    writer = tf.summary.FileWriter(save_valid_to)
    #    best_valid_precision = 0.0
    class LearningRateHook(tf.train.SessionRunHook):
        def begin(self):
            self.learning_rate = ENV.warmup_learning_rate
        def before_run(self, run_context):
            feed_dict = {rn.learning_rate : self.learning_rate}
            return tf.train.SessionRunArgs(rn.global_step, feed_dict=feed_dict)
        def after_run(self, run_context, run_values):
            current_step = run_values.results
            last_step, _ = ENV.learning_steps[-1]
            if current_step > last_step:
                run_context.request_stop()
                return
            for step_change, rate in ENV.learning_steps:
                if current_step < step_change:
                    if self.learning_rate != rate:
                        tf.logging.info("learning_rate: {0:.5f}".format(rate))
                    self.learning_rate = rate
                    break
    summary = tf.summary.merge([rn.summary, accuracy_summary])
    summary_hook = tf.train.SummarySaverHook(
                       save_steps=ENV.save_each_step,
                       output_dir=save_train_to,
                       summary_op=summary)
    log_hook = tf.train.LoggingTensorHook(
                   every_n_iter=ENV.save_each_step,
                   tensors={
                       "step": rn.global_step,
                       "accuracy": accuracy,
                       "loss": rn.loss})
    learning_rate_hook = LearningRateHook()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.MonitoredTrainingSession(
                checkpoint_dir=save_train_to,
                save_checkpoint_secs=ENV.checkpoint_sec,
                save_summaries_steps=0,
                hooks=[log_hook, learning_rate_hook],
                chief_only_hooks=[summary_hook],
                config=config) as sess:
        while not sess.should_stop():
            sess.run(rn.train)

def main(mode):
    with tf.device("/gpu:0"):
        if mode == "test":
            evaluate(TEST_PARAMS, mode=mode)
        elif mode == "valid":
            evaluate(TRAIN_PARAMS, mode=mode)
        else:
            train(TRAIN_PARAMS)
