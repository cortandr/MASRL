from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose


class Brain:

    def __init__(self, learning_rate=1e-2, decay=1e-2, exploration_rate=0.3):
        self.learning_rate = learning_rate
        self.lr_decay = decay
        self.exploration_rate = exploration_rate
        self.model = self.build_network()

    def build_network(self):
        model = Sequential()
        return model

    def save_model(self):
        pass

    def load_model(self):
        pass
