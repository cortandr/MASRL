from keras.models import model_from_json
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, \
    Flatten, Dropout, Activation, Reshape, Input, MaxPool2D, add
from keras.optimizers import Adam
from .Policy import BoltzmannPolicy, EGreedyPolicy
from keras.callbacks import TensorBoard
import os


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
        self.policy = {
            "boltzmann": BoltzmannPolicy(exploration_rate=exploration_rate),
            "e-greedy": EGreedyPolicy(exploration_rate=exploration_rate)
        }
        self.model = self.build_network()
        self.tensorboard = None

    def predict(self, input_tensor, allowed_moves, exploration_type='boltzmann'):

        q_values = self.model.predict(input_tensor)

        return self.policy[exploration_type].follow_policy(
            q_values=q_values[0],
            allowed_moves=allowed_moves,
            training=self.training
        )

    def build_network(self):

        def residual_unit(res_input):

            conv1 = Conv2D(
                filters=32,
                kernel_size=3,
                activation='relu',
                padding='same',
            )(res_input)

            batch_norm = BatchNormalization()(conv1)

            conv2 = Conv2D(
                filters=32,
                kernel_size=3,
                activation='relu',
                padding='same',
            )(batch_norm)

            return add([res_input, conv2])

        input_layer = Input(shape=(self.input_size[0], self.input_size[1], 4))

        conv1 = Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same',
        )(input_layer)

        # First residual unit
        res1 = residual_unit(conv1)

        # Max pooling layer
        max_pool = MaxPool2D()(res1)

        # Second residual unit
        res2 = residual_unit(max_pool)

        # Prepare FCN input tensor
        flattened = Flatten()(res2)

        # Fully connected layers
        fc_1 = Dense(units=256)(flattened)
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

        save_path = 'tensorboard/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.tensorboard = TensorBoard(log_dir=save_path, write_graph=True)

        model.compile(
            loss="mean_squared_error",
            optimizer=optimizer,
            metrics=["mse"],
        )

        return model

    def train(self, X, y, batch_size):
        return self.model.fit(X, y,
                              batch_size=batch_size,
                              epochs=1,
                              verbose=0,
                              callbacks=self.tensorboard)

    def save_model(self, path, name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path+"model_"+name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(path+"model_"+name+".h5")

    def load_model(self, path, name):
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
