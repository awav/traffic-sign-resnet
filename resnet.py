import collections.namedtuple as nt
import tensorflow as tf
import numpy as np

nn_params = nt("nn_params",
                   "learning_rate",
                   "num_resunits",
                   "batch_size",
                   "num_labels",
                   "decay_rate",
                   "momentum_term",
                   "depths")

class Resnet(object):
    def __init__(self, net_batch, labels_batch, params, dtype=tf.float32, mode="eval")
        """
        Residual Neural Network initializer.
        Args:
            inputs_batch: Input batch of images.
                          Shape: [batch_size, image_height, image_width, num_channels]
            labels_batch: Input batch of labels / classes.
                          Shape: [batch_size, labels_count]
            params:       `resnn_params`
            mode:         Either `eval` or `train`. Default: `eval`
        """
        self.mode = mode
        self.inputs = inputs_batch
        self.labels = labels_batch
        self._train_ops = []
        self._dtype = dtype
        self._params = params
        ## self.inference = None
        ## self.loss = None
        ## self.train_op = None
        ## self.learning_rate = None
        ## self.global_step = None
        ## self.train_op = None

    def is_training():
        return self.__mode == "train"

    def graph(self):
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.inference()
        self.loss()
        if is_training():
            self.train()
        self.summaries = tf.summary.merge_all()

    def inference(self):
        #depth = self._params.depths
        depth = [16, 64, 128, 256]
        strides = [0, 1, 2, 2]
        one_stride = [1] * 4
        num_depth = len(depth)
        num_units = self._params.num_resunits
        num_labels = self._params.num_labels
        batch_size = self._params.batch_size
        with tf.variable_scope("init_transition"):
            net = self.inputs
            num_channels = net.get_shape()[-1]
            kernel_size = 3
            net = self._convolution(net, kernel_size, num_channels, filters[0], one_stride)
        for i in xrange(1, num_depth):
            with tf.variable_scope("unit_transition_{0}".format(i)):
                net = self._resunit(net, depth[i-1], depth[i], strides[i])
            for j in xrange(1, num_units):
                with tf.variable_scope("unit_{0}_{1}".format(i, j)):
                    net = self._resunit(net, depth[i], depth[i], one_stride)
        with tf.variable_scope("output_transition"):
            net = self._batch_norm(net)
            net = self._activate(net)
            ## Average pool
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name="avg_pool")
        with tf.variable_scope("fully_connected"):
            net = tf.reshape(net, [batch_size, -1])
            num_params = net.get_shape()[1]
            weights = tf.get_variable(
                             "weights",
                             [num_params, num_labels],
                             initializer=tf.uniform_unit_scaling_initializer())
            biases = tf.get_variable(
                             "biases",
                             [num_labels],
                             initializer=tf.constant_initializer())
            net = tf.nn.xw_plux_b(net, weights, biases)
            self.inference = tf.nn.softmax(net)

    def loss(self):
        decay_rate = self._params.decay_rate
        with tf.variable_scope("loss"):
            losses = []
            for v in tf.trainable_variables():
                if v.op.name.find("weights") != -1:
                    losses.append(tf.nn.l2_loss(v))
            net_entropy = tf.nn.softmax_cross_entropy_with_logits(net, self.labels)
            self.loss = tf.reduce_mean(net_entropy, "loss")
            self.loss += tf.mul(tf.add_n(losses), decay_rate)
            tf.summary.scalar("loss", self.cost)
    
    def train(self, global_step=None):
        if global_step == None:
            self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.learning_rate = tf.constant(self._params, dtype=self._dtype)
        tf.summary.scalar("learning rate", self.learning_rate)
        train_vars = tf.trainable_variables()
        grads = zip(tf.gradients(self.loss, train_vars), train_vars)
        momentum_term = self._params.momentum_term
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum_term)
        apply_grad = optimizer.apply_gradients(
                         grads,
                         global_step=self.global_step,
                         name="train_step")
        train_ops = [apply_grad] + self._train_ops
        self.train_op = tf.group(*train_ops)
            
    def _resunit(self, net, fin, fout, stride):
        res_stide = [1] * 4
        res_fout = fout // 4
        fsizes = [1, 3, 1]
        with tf.variable_scope("resunit_shortcut")
            if fin == fout:
                shortcut = net
            else:
                shortcut = self._convolution(net, 1, fin, fout, stride, name="conv1x1")
        ## This is pre-activation concept appeared in arXiv:1603.05027v[1,2,3]
        with tf.variable_scope("resunit_0"):
            net = self._batch_norm(net, "batch_norm_0")
            net = self._activate(net)
            net = self._convolution(net, fsizes[0], fin, fout, stride, name="conv1x1")
        with tf.variable_scope("resunit_1"):
            net = self._batch_norm(net, "batch_norm_1")
            net = self._activate(net)
            net = self._convolution(net, fsizes[1], fin, res_fout, res_stride, name="conv3x3")
        with tf.variable_scope("resunit_2"):
            net = self._batch_norm(net, "batch_norm_2")
            net = self._activate(net)
            net = self._convolution(net, fsizes[2], fin, res_fout, res_stride, name="conv1x1")
        with tf.variable_scope("resunit_sum"):
            net += shortcut
        return net 

    def _batch_norm(self, net, name="batch_norm"):
        with tf.variable_scope(name):
            ## Input shape is [batch_size, height, width, num_kernels]
            shape = [net.get_shape()[-1]]
            beta = tf.get_variable("beta",
                        shape=shape,
                        dtype=self._dtype,
                        initializer=tf.constant_initializer(0., dtype=self._dtype))
            gamma = tf.get_variable("gamma",
                        shape=shape,
                        dtype=self._dtype,
                        initializer=tf.constant_initializer(1., dtype=self._dtype))
            ## Depending on mode 
            if self.is_training():
                avg, var = tf.nn.moments(net, [0, 1, 2], name="moments")
                mov_avg = tf.get_variable("moving_mean",
                              shape=shape,
                              dtype=self._dtype,
                              trainable=False,
                              initializer=tf.constant_initializer(0., dtype=self._dtype))
                mov_var = tf.get_variable("moving_var",
                              shape=shape,
                              dtype=self._dtype,
                              trainable=False,
                              initializer=tf.constant_initializer(1., dtype=self._dtype))
                decay = 0.85
                mov_avg_prime = tf.python.training.assign_moving_average(mov_avg, avg, decay)
                mov_var_prime = tf.python.training.assign_moving_average(mov_var, var, decay)
                self._train_ops.extend([mov_avg_prime, mov_var_prime])
            else:
                avg = tf.get_variable("moving_mean",
                           shape=shape,
                           dtype=self._dtype,
                           trainable=False,
                           initializer=tf.constant_initializer(0., dtype=self._dtype))
                var  = tf.get_variable("moving_var",
                           shape=shape,
                           dtype=self._dtype,
                           trainable=False,
                           initializer=tf.constant_initializer(1., dtype=self._dtype))
            epsilon = 1e-5
            bn = tf.nn.batch_normalization(net, mean, var, beta, gamma, epsilon)
            bn.set_shape(net.get_shape())
            return bn

    def _activate(self, net, name="relu"):
        return tf.nn.relu(net, name=name)

    def _convolution(self, net, fsize, fin, fout, stride, name="convolution"):
        with tf.variable_scope(name):
            num_net = fsize * fsize * fout
            k = tf.get_variable(
                      "weights",
                      shape=[fsize, fsize, fin, fout],
		      dtype=self._dtype,
                      ## TODO: Consider Xavier initializer,
                      ##       tf.contrib.layer.xavier_initializer
                      initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/num_net)))
            return tf.nn.conv2d(net, k, strides, padding="SAME")

    def 
