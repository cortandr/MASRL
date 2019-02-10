import uuid


class Agent:
    def __init__(self, position, training=False):
        self.agentID = uuid.uuid4()
        self.position = position
        self.training = training
        self.chosen_action = None
        self.controller = None

    def choose_action(self, allowed_moves):
        return 'u'

    def get_chosen_action(self):
        return self.chosen_action

    def get_position(self):
        return self.position

    def set_position(self, new_pos):
        self.position = new_pos


class DummyAgent:
    def __init__(self, position):
        self.agentID = uuid.uuid4()
        self.state = None
        self.position = position

    def get_position(self):
        return self.position

    def set_position(self, new_pos):
        self.position = new_pos
