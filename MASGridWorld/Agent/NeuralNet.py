import tensorflow as tf
import numpy as np
import random


class Brain:

    def __init__(self, input_size=(8, 8), learning_rate=1e-2, decay=1e-2, exploration_rate=0.3):
        self.input_size = input_size
        self.exploration_rate = exploration_rate
        self.input_layer = None
        self.target_Q = None
        self.Q_values = None
        self.learning_rate = learning_rate
        self.lr_decay = decay
        self.exploration_rate = exploration_rate
        self.optimizer = None
        self.loss = None
        self.train_op = None
        self.saver = None
        self.sess = None
        self.training = True
        self.build_network()

    def predict(self, input_tensor):

        predictions = self.sess.run(self.Q_values, feed_dict={self.input_layer: input_tensor})

        # Take random move or choose best Q-value and associated action
        if self.training and random.uniform(0, 1) > self.exploration_rate:
            i = random.randint(0, 5)
            return predictions[0][i]
        else:
            return np.argmax(predictions)

    def build_network(self):

        g = tf.Graph()
        with g.as_default():
            # Input Layer
            self.input_layer = tf.placeholder(tf.float32, [None, None, None, 4])
            self.target_Q = tf.placeholder(tf.float32, (1, 5))

            conv1 = tf.layers.conv2d(
                inputs=self.input_layer,
                filters=32,
                kernel_size=[5, 5],
                strides = 1,
                padding="same",
                activation=tf.nn.relu,
                name = "conv1")

            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[3, 3],
                strides = 1,
                padding="same",
                activation=tf.nn.relu,
                name="conv2")

            batch_norm1 = tf.layers.batch_normalization(conv2, axis=1)

            conv3 = tf.layers.conv2d(
                inputs=batch_norm1,
                filters=256,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv3")

            skip_connection1 = tf.concat(conv3, conv1)
            max_pool1 = tf.layers.max_pooling2d(skip_connection1, pool_size=2)

            conv4 = tf.layers.conv2d(
                inputs=max_pool1,
                filters=512,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv4")

            batch_norm2 = tf.layers.batch_normalization(conv4, axis=1)

            conv5 = tf.layers.conv2d(
                inputs=batch_norm2,
                filters=256,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv5")

            skip_connection2 = tf.concat(conv5, max_pool1)

            global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(skip_connection2)

            # Fully connected NN

            # First layer
            fc_1 = tf.layers.dense(
                inputs=global_avg_pool,
                units=256,
                activation=tf.nn.elu, name="fc_nn1")

            normalized_fc_1 = tf.layers.batch_normalization(fc_1, axis=1)

            # Second layer
            fc_2 = tf.layers.dense(
                inputs=normalized_fc_1,
                units=512,
                activation=tf.nn.elu,
                name="fc_nn2")

            normalized_fc_2 = tf.layers.batch_normalization(fc_2, axis=1)

            dropout_fc_2 = tf.layers.dropout(normalized_fc_2, rate=0.5)

            self.Q_values = tf.layers.dense(
                inputs=dropout_fc_2,
                units=8,
                activation=tf.nn.softmax,
                name="q_values")

            # Calculate Loss
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_values))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(loss=self.loss)

            # Initialize all variables
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.sess.run(init)

    def save_model(self, path):
        p = self.saver.save(self.sess, (path + "/model"))
        return p

    def load_model(self, path):
        self.saver.restore(self.sess, path)

