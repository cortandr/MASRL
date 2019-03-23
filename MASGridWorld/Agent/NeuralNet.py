import tensorflow as tf
import numpy as np
import random
import os


class Brain:

    def __init__(self, input_size=(10, 10), learning_rate=1e-2, decay=1e-2, exploration_rate=0.3):
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

        predictions = self.sess.run(self.Q_values,
                                    feed_dict={self.input_layer: input_tensor})

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
            self.input_layer = tf.placeholder(
                tf.float32,
                [None, self.input_size[0], self.input_size[1], 4])
            self.target_Q = tf.placeholder(tf.float32, (1, 8))

            conv1 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[5, 5],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv1")(self.input_layer)

            conv2 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                name="conv2")(conv1)

            batch_norm1 = tf.keras.layers.BatchNormalization(axis=3)(conv2)

            conv3 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                name="conv3")(batch_norm1)

            skip_connection1 = tf.concat([conv3, conv1], axis=3)
            max_pool1 = tf.keras.layers.MaxPool2D(
                strides=1,
                pool_size=2)(skip_connection1)

            conv4 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                name="conv4")(max_pool1)

            batch_norm2 = tf.keras.layers.BatchNormalization(axis=3)(conv4)

            conv5 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                name="conv5")(batch_norm2)

            skip_connection2 = tf.concat([conv5, max_pool1], axis=3)

            # Averaging pooling Layer
            avPool_out = tf.reduce_mean(skip_connection2, axis=(1, 2), keepdims=True)
            avPool_output = tf.keras.layers.Flatten()(avPool_out)

            global_avg_pool = tf.keras.layers.BatchNormalization(axis=1)(avPool_output)

            # Fully connected NN

            # First layer
            fc_1 = tf.keras.layers.Dense(
                units=256,
                activation=tf.nn.elu, name="fc_nn1")(global_avg_pool)

            normalized_fc_1 = tf.keras.layers.BatchNormalization(axis=1)(fc_1)

            # Second layer
            fc_2 = tf.keras.layers.Dense(
                units=512,
                activation=tf.nn.elu,
                name="fc_nn2")(normalized_fc_1)

            normalized_fc_2 = tf.keras.layers.BatchNormalization(axis=1)(fc_2)

            dropout_fc_2 = tf.keras.layers.Dropout(rate=0.5)(normalized_fc_2)

            self.Q_values = tf.keras.layers.Dense(
                units=8,
                activation=tf.nn.tanh,
                name="q_values")(dropout_fc_2)

            # Calculate Loss
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_values))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(loss=self.loss)

            # Initialize all variables
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.sess.run(init)

    def save_model(self, path, name):
        if not os.path.exists(path):
            os.mkdirs(path)
        p = self.saver.save(self.sess, (path + "/model_" + name))
        return p

    def load_model(self, path):
        self.saver.restore(self.sess, path)