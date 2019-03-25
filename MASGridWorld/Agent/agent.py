import uuid
import numpy as np
from utils import *


class Agent:
    def __init__(self, position, training=False):
        self.agentID = uuid.uuid4()
        self.position = position
        self.training = training
        self.chosen_action = None
        self.brain = None

    def choose_action(self, allowed_moves, state):

        best_move_idx = self.brain.predict(state, allowed_moves)

        self.position = allowed_moves[best_move_idx]
        self.chosen_action = best_move_idx

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

    def choose_action(self, allowed_moves, agents):

        # Get agents positions
        agent_pos = [a.get_position() for a in agents]

        # Get agents distances from current dummy agent
        agent_distance = [point_dist(self.position, a_pos)
                          for a_pos in agent_pos]

        closest_agent = min(agent_distance)
        closest_pos = agent_distance.index(closest_agent)

        dists = [point_dist(agents[closest_pos].get_position(), pos)
                 if pos else -1 for pos in allowed_moves]
        furthest_dist = max(dists)
        self.position = allowed_moves[dists.index(furthest_dist)]

    def get_position(self):
        return self.position

    def set_position(self, new_pos):
        self.position = new_pos
