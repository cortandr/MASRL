import tensorflow as tf
import numpy as np
import random


class Brain:

    def __init__(self, input_size=(10, 10), learning_rate=1e-4,
                 decay=1e-2, exploration_rate=0.3, discount_rate=0.7):
        self.input_size = input_size
        self.exploration_rate = exploration_rate
        self.input_layer = None
        self.target_Q = None
        self.Q_values = None
        self.learning_rate = learning_rate
        self.lr_decay = decay
        self.exploration_rate = exploration_rate
        self.discount_rate = discount_rate
        self.merged_summary = None
        self.loss = None
        self.train_op = None
        self.global_step = None
        self.saver = None
        self.sess = None
        self.writer = None
        self.training = True
        self.temp = 0
        self.build_network()

    def predict(self, input_tensor, allowed_moves):

        predictions = self.sess.run(
            self.Q_values,
            feed_dict={
                self.input_layer: input_tensor,
            })

        moves_mask = np.array([1 if pos else np.nan for pos in allowed_moves])

        masked_q_values = predictions * moves_mask
        valid_idx = [i for i in range(len(masked_q_values[0]))
                     if not np.isnan(masked_q_values[0][i])]

        # Take random move or choose best Q-value and associated action
        if self.training and random.uniform(0, 1) < self.exploration_rate:
            return random.choice(valid_idx)
        else:
            return np.nanargmax(masked_q_values)

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
                name="Conv_1")(self.input_layer)

            conv2 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                name="Conv_2")(conv1)

            batch_norm1 = tf.keras.layers.BatchNormalization(
                axis=3,
                name="BatchNorm1"
            )(conv2)

            conv3 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                name="Conv_3")(batch_norm1)

            skip_connection1 = tf.concat(
                [conv3, conv1],
                axis=3,
                name="SkipConnection1"
            )
            max_pool1 = tf.keras.layers.MaxPool2D(
                strides=1,
                pool_size=2,
                name="MaxPool1"
            )(skip_connection1)

            conv4 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                name="Conv_4")(max_pool1)

            batch_norm2 = tf.keras.layers.BatchNormalization(
                axis=3,
                name="BatchNorm2"
            )(conv4)

            conv5 = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                name="Conv_5")(batch_norm2)

            skip_connection2 = tf.concat(
                [conv5, max_pool1],
                axis=3,
                name="SkipConnection2"
            )

            # Averaging pooling Layer
            with tf.name_scope("AveragePooling"):
                avg_pool_out = tf.reduce_mean(skip_connection2, axis=(1, 2),
                                              keepdims=True)
                avg_pool_output = tf.keras.layers.Flatten()(avg_pool_out)
                global_avg_pool = tf.keras.layers.BatchNormalization(
                    axis=1)(avg_pool_output)

            # Fully connected NN
            # First layer
            fc_1 = tf.keras.layers.Dense(
                units=128,
                activation=tf.nn.elu,
                name="fc_1"
            )(global_avg_pool)

            normalized_fc_1 = tf.keras.layers.BatchNormalization(
                axis=1,
                name="BatchNorm_fc_1"
            )(fc_1)

            dropout_fc_2 = tf.keras.layers.Dropout(
                rate=0.5,
                name="Dropout"
            )(normalized_fc_1)

            self.Q_values = tf.keras.layers.Dense(
                    units=8,
                    activation=tf.nn.tanh,
                    name="q_values"
            )(dropout_fc_2)

            # Calculate Loss
            with tf.name_scope("Loss"):
                self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_values))

            with tf.name_scope("Global_Step"):
                self.global_step = tf.Variable(0, trainable=False)

            with tf.name_scope("Learning_Rate"):
                self.learning_rate = tf.train.exponential_decay(
                    learning_rate=0.01,
                    global_step=self.global_step,
                    decay_steps=100000,
                    decay_rate=0.96,
                    staircase=True
                )

            with tf.name_scope("Train"):
                self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                ).minimize(self.loss, global_step=self.global_step)

            # Create tensor board summaries
            tf.summary.scalar("Learning_Rate", self.learning_rate)
            tf.summary.scalar("Global_Step", self.global_step)
            tf.summary.scalar("Loss", self.loss)

            self.merged_summary = tf.summary.merge_all()

            # Initialize all variables
            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.sess.run(init)
            self.writer = tf.summary.FileWriter("tensorboard/mas/1")
            self.writer.add_graph(self.sess.graph)

    def save_model(self, path, name):
        p = self.saver.save(self.sess, (path + "model_" + name))
        return p

    def load_model(self, path):
        self.saver.restore(self.sess, path)
