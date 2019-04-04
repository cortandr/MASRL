import random
import numpy as np


class Policy:

    def follow_policy(self, q_values, allowed_moves, training):
        pass

    def update_exploration_rate(self, new_er):
        pass


class BoltzmannPolicy(Policy):

    def __init__(self, exploration_rate):
        self.temperature = exploration_rate

    def follow_policy(self, q_values, allowed_moves, training):

        # Apply softmax to q vector
        q_dist = np.exp(q_values[0] / self.temperature) / sum(
            np.exp(q_values[0] / self.temperature))
        q_dist = np.reshape(q_dist, newshape=(1, q_values.shape[1]))

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
        if not training:
            return np.nanargmax(masked_q_values)

        for idx, el in enumerate(masked_q_values[0]):
            if el and random_sum <= el:
                return idx
            random_sum = random_sum - el if not np.isnan(el) else random_sum

    def update_exploration_rate(self, new_er):
        self.temperature -= new_er


class EGreedyPolicy(Policy):

    def __init__(self, exploration_rate):
        self.exploration_rate = exploration_rate

    def follow_policy(self, q_values, allowed_moves, training):
        # Mask distribution based on allowed moves
        moves_mask = np.array([1 if pos else np.nan for pos in allowed_moves])
        masked_q_values = q_values * moves_mask

        # e-greedy exploration
        if training and random.uniform(0, 1) < self.exploration_rate:
            action_value = random.choice(
                [el for el in masked_q_values[0] if not np.isnan(el)])
            return np.argwhere(masked_q_values[0] == action_value)[0][0]
        else:
            return np.nanargmax(masked_q_values[0])

    def update_exploration_rate(self, new_er):
        self.exploration_rate -= new_er

