from collections  import namedtuple as nt
from functools import reduce
from operator import mul
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

env = envconfig(
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

train_params = resnet_arch.nn_params(
    learning_rate = 0.1,
    num_resunits_per_block = 2,
    batch_size = 128,
    num_labels = 43,
    weight_decay_rate = 0.0003,
    momentum_term = 0.9,
    depths = [16, 64, 128, 256],
    image_shape = [32, 32, 3])

test_params = resnet_arch.nn_params(
    num_labels = 43,
    batch_size = 300,
    num_resunits_per_block = 2,
    learning_rate = 0.1,
    weight_decay_rate = 0.0003,
    momentum_term = 0.9,
    depths = [16, 64, 128, 256],
    image_shape = [32, 32, 3])

def create_data_reader(
        directory,
        shape=train_params.image_shape,
        batch_size=train_params.batch_size,
        num_labels=train_params.num_labels,
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

def evaluate(params, mode="test"):
    if mode == "valid":
        data_dir = env.valid_dir
        num_batches = env.num_valid_batches
        pattern = "valid"
    elif mode == "test":
        data_dir = env.test_dir
        num_batches = env.num_test_batches
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
    save_to = os.path.join(env.save_to_dir, mode)
    train_dir = os.path.join(env.save_to_dir, "train")
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(save_to)
    #best_precision = 0.0
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    tf.train.start_queue_runners(sess)
    while True:
        #time.sleep(30)
        checkpoint = tf.train.get_checkpoint_state(train_dir)
        if not checkpoint or not checkpoint.model_checkpoint_path:
            tf.logging.error("Model not found {0}".format(train_dir))
            continue
        tf.logging.info("Start loading checkpoint at {0}".format(train_dir))
        saver.restore(sess, checkpoint.model_checkpoint_path)
        correct, total, total_loss = 0.0, 0.0, 0.0
        for i in range(num_batches):
            run = [rn.inference, rn.labels, rn.loss, rn.global_step, rn.summary]
            pred, true, loss, step, summary = sess.run(run)
            pred = np.argmax(pred, axis=1)
            true = np.argmax(true, axis=1)
            correct += np.sum(true == pred)
            total_loss += loss
            total += pred.shape[0]
        precision = correct / total
        #best_precision = max(best_precision, precision)
        #summary_best_precision = tf.Summary()
        #summary_best_precision.value.add(tag=mode+"_best_precision", simple_value=best_precision)
        #writer.add_summary(summary_best_precision, step)
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
        break

def train(params):
    save_train_to = os.path.join(env.save_to_dir, "train")
    #save_train_to = env.save_to_dir
    #save_valid_to = os.path.join(env.save_to_dir, "valid")
    x, y = create_data_reader(env.train_dir)
    with tf.variable_scope("train"):
        rn = build_model(x, y, params)
        true_predictions = tf.argmax(y, axis=1)
        predictions = tf.argmax(rn.inference, axis=1)
        hits = tf.cast(tf.equal(predictions, true_predictions), tf.float32)
        accuracy = tf.reduce_mean(hits)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    #with tf.variable_scope("valid"):
    #    x_valid, y_valid = create_data_reader(env.train_dir, mode="valid")
    #    rn_valid = build_model(x_valid, y_valid, params, mode="eval")
    #    saver = tf.train.Saver()
    #    writer = tf.summary.FileWriter(save_valid_to)
    #    best_valid_precision = 0.0
    class LearningRateHook(tf.train.SessionRunHook):
        def begin(self):
            self.learning_rate = env.warmup_learning_rate
        def before_run(self, run_context):
            feed_dict = {rn.learning_rate : self.learning_rate}
            return tf.train.SessionRunArgs(rn.global_step, feed_dict=feed_dict)
        def after_run(self, run_context, run_values):
            current_step = run_values.results
            last_step, _ = env.learning_steps[-1]
            if current_step > last_step:
                run_context.request_stop()
                return
            for step_change, rate in env.learning_steps:
                if current_step < step_change:
                    if self.learning_rate != rate:
                        tf.logging.info("learning_rate: {0:.5f}".format(rate))
                    self.learning_rate = rate
                    break
            #if curren_step != 0 && current_step % env.valid_each_step == 0:
            #    checkpoint = tf.train.get_checkpoint_state(train_dir)
            #    if not checkpoint or not checkpoint.model_checkpoint_path:
            #        tf.logging.error("Model not found {0}".format(train_dir))
            #        return
            #    #tf.logging.info("Start loading checkpoint at {0}".format(train_dir))
            #    saver.restore(sess, checkpoint.model_checkpoint_path)
            #    correct, total, total_loss = 0.0, 0.0, 0.0
            #    for i in range(env.num_valid_batches):
            #        run = [rn.inference, rn.labels, rn.loss, rn.summary]
            #        pred, true, loss, summary = sess.run(run)
            #        pred = np.argmax(pred, axis=1)
            #        true = np.argmax(true, axis=1)
            #        correct += np.sum(true == pred)
            #        total_loss += loss
            #        total += pred.shape[0]
            #    precision = correct / total
            #    best_valid_precision = max(best_precision, precision)
            #    summary_best_precision = tf.Summary()
            #    summary_best_precision.value.add(
            #        tag="valid_best_precision",
            #        simple_value=best_precision)
            #    writer.add_summary(summary_best_precision, current_step)
            #    summary_precision = tf.Summary()
            #    summary_precision.value.add(
            #        tag="valid_precision",
            #        simple_value=precision)
            #    writer.add_summary(summary_precision, current_step)
            #    summary_loss = tf.Summary()
            #    total_loss /= env.num_valid_batches
            #    summary_loss.value.add(tag="valid_loss",
            #        simple_value=total_loss)
            #    writer.add_summary(summary_loss, current_step)
            #    msg = "{0}_loss: {1:.5f}, "
            #          "{0}_precision: {2:.5f}, "
            #          "{0}_best_precision: {3:.5f}, "
            #          "{0}_loss: {4:.5f}"
            #    tf.logging.info(msg.format("valid", loss, precision, best_precision, total_loss))
            #    writer.flush()
    
    summary = tf.summary.merge([rn.summary, accuracy_summary])
    summary_hook = tf.train.SummarySaverHook(
                       save_steps=env.save_each_step,
                       output_dir=save_train_to,
                       summary_op=summary)
    log_hook = tf.train.LoggingTensorHook(
                   every_n_iter=env.save_each_step,
                   tensors={
                       "step": rn.global_step,
                       "accuracy": accuracy,
                       "loss": rn.loss})
    learning_rate_hook = LearningRateHook()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.MonitoredTrainingSession(
                checkpoint_dir=save_train_to,
                save_checkpoint_secs=env.checkpoint_sec,
                save_summaries_steps=0,
                hooks=[log_hook, learning_rate_hook],
                chief_only_hooks=[summary_hook],
                config=config) as sess:
        while not sess.should_stop():
            sess.run(rn.train)

def main(mode):
    with tf.device("/gpu:0"):
        if mode == "test":
            evaluate(test_params, mode=mode)
        elif mode == "valid":
            evaluate(train_params, mode=mode)
        else:
            train(train_params)
