import os
import sys
import importlib
import pickle
import pandas as pd
import imagerich
import tensorflow as tf
import numpy as np
import numpy.random as rnd
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

importlib.reload(imagerich)

def num_labels():
    return 43

def fliplr_labels():
    return [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]

def read(train_file, test_file):
    with open(train_file, mode="rb") as f:
        train = pickle.load(f)
    with open(test_file, mode="rb") as f:
        test = pickle.load(f)
    return train["features"], train["labels"], test["features"], test["labels"]

def label_description(y):
    num = y.shape[0]
    raw = list(Counter(y).items())
    ds = pd.DataFrame(raw, columns=["class", "count"])
    ds = ds.sort(columns="count", ascending=False)
    ds["percentage"] = ds["count"] / np.float32(num)
    ds = ds.set_index(np.arange(num_labels()))
    return ds

def augment(x, y, size, class_jitter=200):
    num = x.shape[0]
    sorted_ids = np.argsort(y)
    onehot_diff = np.diff(np.take(y, sorted_ids))
    split_ids = np.append([0], np.add(np.where(onehot_diff > 0), 1))
    ds = label_description(y)
    label_counts = ds["count"]
    label_probs = (label_counts.size * [label_counts.max()] - label_counts
                  + rnd.randint(class_jitter, size=label_counts.size))
    label_probs /= label_probs.sum()
    imagerich.set_seed()
    rich = imagerich.ImageRich()
    rich_without_fliplr = imagerich.ImageRich(exclude=["fliplr"])
    n_samples = len(sorted_ids)
    n_classes = len(split_ids)
    def __choose(cid):
        nid = cid + 1
        beg, end = cid, nid if nid < n_classes else -1
        beg = split_ids[beg]
        end = split_ids[end] if end != -1 else n_samples
        assert(beg < n_samples)
        assert(end <= n_samples)
        sorted_idx = rnd.randint(beg, end)
        idx = sorted_ids[sorted_idx]
        assert(y[idx] == cid)
        return x[idx]
    def __augment(cid, img):
        if cid in fliplr_labels():
            return rich.augment(img)
        return rich_without_fliplr.augment(img)
    def __id_to_image(cid):
        return __augment(cid, __choose(cid))
    labels = rnd.choice(list(ds['class']), p=list(label_probs), size=size)
    image_type = type(x[0])
    augment_func = np.vectorize(__id_to_image, otypes=[image_type])
    augmented = augment_func(labels)
    return np.array(list(augmented)), np.array(labels, dtype=np.uint8)

def save_as_tfrecords(x, y, file_path):
    num_examples = x.shape[0]
    assert num_examples == y.shape[0]
    with tf.python_io.TFRecordWriter(file_path) as w:
        for i in range(num_examples):
            if i % 10000 == 0 and i != 0:
                print("{0}-th input is proccessed for {1}".format(i, file_path))
            xe = map(int, x[i].flatten().tolist())
            ye = int(y[i])
            xfeature = tf.train.Feature(int64_list=tf.train.Int64List(value=xe))
            yfeature = tf.train.Feature(int64_list=tf.train.Int64List(value=[ye]))
            example = tf.train.Example(features=tf.train.Features(feature=
                {"image": xfeature, "label": yfeature}))
            w.write(example.SerializeToString())

def save_for_training(xtrain, ytrain, xtest, ytest,
                      model_dir="tf-data",
                      valid_size=0.1,
                      num_splits=4,
                      override=False):
    tfdir = os.path.join(model_dir, "data")
    tffile_train = os.path.join(tfdir, "train{0}.tfrecords")
    tffile_valid = os.path.join(tfdir, "valid.tfrecords")
    tffile_test = os.path.join(tfdir, "test.tfrecords")
    xtrain, ytrain = shuffle(xtrain, ytrain)
    xt, xv, yt, yv = train_test_split(xtrain, ytrain, test_size=valid_size)
    xt_split = np.array_split(xt, num_splits)
    yt_split = np.array_split(yt, num_splits)
    os.makedirs(tfdir, exist_ok=True)
    for i, _ in enumerate(xt_split):
        filename = tffile_train.format(i)
        if override or not os.path.exists(filename):
            save_as_tfrecords(xt_split[i], yt_split[i], filename)
        else:
            print("Train tensorflow record {0} already exists".format(filename),
                  file=sys.stderr)
    if override or not os.path.exists(tffile_valid):
        save_as_tfrecords(xv, yv, tffile_valid)
    else:
        print("Valid tensorflow record {0} already exists".format(tffile_valid),
              file=sys.stderr)
    if override or not os.path.exists(tffile_test):
        save_as_tfrecords(xtest, ytest, tffile_test)
    else:
        print("Test tensorflow record {0} already exists".format(tffile_test),
              file=sys.stderr)
    return xt.shape, yt.shape, xv.shape, yv.shape
