import keras


class Agent:
    def __init__(self, position):
        self.position = position
        # Neural net
        self.controller = None

    def choose_action(self, allowed_moves):
        return 'u'


class DummyAgent:
    def __init__(self, position):
        self.state = None
        self.position = position
