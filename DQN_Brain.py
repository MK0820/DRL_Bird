import tensorflow as tf
import numpy
from collections import deque


class Brain:
    def __init__(self, actions):
        self.actions = actions
        self.replayMemory = deque()
        self.epsilon = 0
        self.timeStape = 0
        self.createQnet()

    def createQnet(self):
        X = tf.placeholder(tf.float32, [None, 80, 80, 4])

        with tf.name_scope('conv1'):
            with tf.variable_scope('con1'):
                filter_con1 = tf.get_variable('f1', [8, 8, 4, 32])
                b_con1 = tf.get_variable('b1', [32])
            h_conv1 = tf.nn.relu(tf.nn.conv2d(X, filter_con1, strides=4, padding='SAME') + b_con1)

        with tf.name_scope('conv2'):
            with tf.variable_scope('con2'):
                filter_con2 = tf.get_variable('f2', [4, 4, 32, 64])
                b_con2 = tf.get_variable('b2', [64])
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, filter_con2, strides=2, padding='SAME') + b_con2)

        with tf.name_scope('conv3'):
            with tf.variable_scope('con3'):
                filter_con3 = tf.get_variable('f3', [3, 3, 64, 64])
                b_con3 = tf.get_variable('b3', [64])
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, filter_con3, strides=1, padding='SAME') + b_con3)

        with tf.name_scope('dense1'):
            h_conv3 = tf.reshape(h_conv3, [-1, 1600])
            with tf.variable_scope('dense_v1'):
                w_den1 = tf.get_variable('w1', [1600, 512])
                b_den1 = tf.get_variable('b4', [512])
            h_den1 = tf.nn.relu(tf.matmul(h_conv3, w_den1) + b_den1)

        with tf.name_scope('Q_layer'):
            with tf.variable_scope('Q_v1'):
                w_Q = tf.get_variable('wq', [512, self.actions])
            self.QValue = tf.matmul(h_den1)



