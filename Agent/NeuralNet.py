from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D
from keras.layers import BatchNormalization, MaxPool2D, concatenate
import pickle


class Brain:

    def __init__(self, input_size=(8, 8), learning_rate=1e-2, decay=1e-2, exploration_rate=0.3):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.lr_decay = decay
        self.exploration_rate = exploration_rate
        self.optimizer = None
        self.epoch = 100
        self.model = None
        self.build_network()

    def build_network(self):

        # This returns a tensor
        input_tensor = Input(shape=(self.input_size[0], self.input_size[1], 4))

        # add model layers
        conv1 = Conv2D(
            filters=32,
            kernel_size=5,
            activation="relu")(input_tensor)

        conv2 = Conv2D(
            filters=32,
            kernel_size=4,
            activation="relu")(conv1)

        batch_norm1 = BatchNormalization()(conv2)

        conv3 = Conv2D(
            filters=32,
            kernel_size=3,
            activation="relu")(batch_norm1)

        maxpool1 = MaxPool2D(pool_size=2)(concatenate([conv3, conv1]))

        conv4 = Conv2D(
            filters=32,
            kernel_size=3,
            activation="relu")(maxpool1)

        batch_norm2 = BatchNormalization()(conv4)

        conv5 = Conv2D(
            filters=32,
            kernel_size=3,
            activation="relu")(batch_norm2)

        gap = GlobalAveragePooling2D()(concatenate([maxpool1, conv5]))

        dense1 = Dense(
            units=32,
            activation="relu"
        )(gap)

        dense2 = Dense(
            units=512,
            activation="relu"
        )(dense1)

        predictions = Dense(
            units=5,
            activation="softmax"
        )(dense2)

        self.optimizer = Adam(lr=self.learning_rate,
                              beta_1=0.9,
                              beta_2=0.999,
                              epsilon=None,
                              decay=self.lr_decay, amsgrad=False)

        self.model = Model(inputs=input_tensor, outputs=predictions)

        self.model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy'
        )

    def train(self, X, y):

        self.model.fit(X, y, batch_size=1, nb_epoch=self.epoch)

    def save_model(self):

        with open("brain_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self):

        with open("brain_model.pkl", "rb") as f:
            self.model = pickle.load(f)
