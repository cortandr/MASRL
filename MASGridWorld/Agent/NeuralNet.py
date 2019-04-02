from keras.models import model_from_json
import numpy as np
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, \
    Flatten, Dropout, Activation, Reshape, Input, MaxPool2D, add
from keras.optimizers import Adam
import random
import pickle


class Brain:

    def __init__(self, training, input_size=(10, 10), learning_rate=1e-4,
                 decay=1e-2, exploration_rate=1.0, discount_rate=0.7,
                 output_dim=8):
        self.input_size = input_size
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.lr_decay = decay
        self.discount_rate = discount_rate
        self.loss = None
        self.temp = 1.0
        self.training = training
        self.output_dim = output_dim
        self.model = self.build_network()

    def predict(self, input_tensor, allowed_moves, exploration_type='boltzmann'):

        q_values = self.model.predict(input_tensor)

        # Use softmax for exploration
        q_dist = np.exp(q_values[0]/self.temp)/sum(np.exp(q_values[0]/self.temp))
        q_dist = np.reshape(q_dist, newshape=(1, self.output_dim))

        if exploration_type == 'boltzmann':
            # Mask distribution bbased on allowed moves
            moves_mask = np.array([1 if pos else 0 for pos in allowed_moves])
            masked_q_values = q_dist * moves_mask

            # Apply random choice with probability given by softmax op
            prob_sum = sum(masked_q_values[0])
            random_sum = random.uniform(0, prob_sum)
            masked_q_values[masked_q_values == 0] = None

            # Cover case in which softmax gives a 1 for a not allowed move
            if all(np.isnan(masked_q_values[0])):
                return random.choice([i for i in range(len(allowed_moves))
                                      if allowed_moves[i] is not None])

            # Exploit if not training
            if not self.training:
                return np.nanargmax(masked_q_values)

            for idx, el in enumerate(masked_q_values[0]):
                if el and random_sum <= el:
                    return idx
                random_sum = random_sum - el if not np.isnan(el) else random_sum
        elif exploration_type == 'e-greedy':
            # Mask distribution based on allowed moves
            moves_mask = np.array([1 if pos else np.nan for pos in allowed_moves])
            masked_q_values = q_values * moves_mask

            # e-greedy exploration
            if self.training and random.uniform(0, 1) < self.exploration_rate:
                action_value = random.choice(
                    [el for el in masked_q_values[0] if not np.isnan(el)])
                return np.argwhere(masked_q_values[0] == action_value)[0][0]
            else:
                return np.nanargmax(masked_q_values[0])

    def build_network(self):

        input_layer = Input(shape=(self.input_size[0], self.input_size[1], 4))

        conv1 = Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same',
        )(input_layer)

        conv2 = Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same',
        )(conv1)

        batch_norm1 = BatchNormalization()(conv2)

        conv3 = Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same',
        )(batch_norm1)

        skip_conn1 = add([conv1, conv3])

        max_pool1 = MaxPool2D()(skip_conn1)

        conv4 = Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same',
        )(max_pool1)

        batch_norm2 = BatchNormalization()(conv4)

        conv5 = Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same',
        )(batch_norm2)

        skip_conn1 = add([max_pool1, conv5])

        flattened = Flatten()(skip_conn1)

        fc_1 = Dense(
            units=256,
        )(flattened)
        fc_batch_norm = BatchNormalization()(fc_1)
        fc_act = Activation(activation='relu')(fc_batch_norm)
        dropout = Dropout(rate=0.5)(fc_act)

        fc_2 = Dense(
            units=self.output_dim,
            activation='linear'
        )(dropout)

        output = Reshape((-1, self.output_dim))(fc_2)

        model = Model(inputs=input_layer, outputs=output)

        optimizer = Adam(
            lr=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False
        )
        model.compile(
            loss="mean_squared_error",
            optimizer=optimizer,
            metrics=["mse"]
        )

        return model

    def train(self, X, y, batch_size):
        history = self.model.fit(
            X, y, batch_size=batch_size, epochs=1, verbose=0)
        return history

    def save_model(self, path, name):
        # with open(path + "model_" + name + ".pkl", "wb") as f:
        #     pickle.dump(self.model, f)
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path+"model_"+name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(path+"model_"+name+".h5")

    def load_model(self, path, name):
        # with open(path + "model_" + name + ".pkl", "rb") as f:
        #     self.model = pickle.load(f)
        # load json and create model
        json_file = open(path+"model_"+name+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(path+"model_"+name+".h5")
        optimizer = Adam(
            lr=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False
        )
        self.model.compile(
            loss="mean_squared_error",
            optimizer=optimizer,
            metrics=["mse"]
        )
