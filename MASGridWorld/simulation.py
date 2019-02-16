import numpy as np
from environment import Environment
import random
import copy
from Agent.config import *


class Sim:

    def __init__(self, allies, opponents, world_size, n_games,
                 train_batch_size):
        self.experience_replay = list()
        self.training_batch_size = train_batch_size
        self.n_games = n_games
        self.environment = Environment(
            n_rows=world_size[0],
            n_cols=world_size[1],
            n_agents=allies,
            n_opponents=opponents)

        self.metrics = {
            "reward": list(),
            "loss": list()
        }

    def run(self):
        """
        Runs general simulation and takes care of experience table population
        and agents training
        :return:
        """

        sim = 0

        while sim < self.n_games:

            sim_ended = False

            training_agent = next(
                filter(lambda ag: ag.training, self.environment.agents))

            while not sim_ended:

                # Current state
                curr_state = self.environment.brain_input_grid

                # Apply step in Environment
                self.environment.step()

                # Get agent chosen action
                action = training_agent.get_chosen_action()

                # Get reward
                reward = 0

                # Get state after chosen action is applied
                next_state = self.environment.brain_input_grid

                # Store transition in replay table
                self.experience_replay.append({
                    "state": curr_state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                })

    def train_ally(self):
        """
        Takes a random batch from experience replay memory and uses it to train
        the agent's brain NN
        :return:
        """
        # Sample batch fro replay memory
        mini_batch = random.sample(
            self.experience_replay,
            self.training_batch_size)

        # Get agent under training
        training_agent = next(
            filter(lambda ag: ag.training, self.environment.agents))
        exploration_rate = training_agent.brain.exploration_rate

        # Get target agent
        target_agent = next(
            filter(lambda ag: not ag.training, self.environment.agents))

        for transition in mini_batch:

            # Compute Q value of current training network
            q = training_agent.sess.run(
                training_agent.brain.Q_values,
                feed_dict={training_agent.brain.input_layer: transition["state"]})

            # Check for possible ending state
            if transition["next_state"] is None:
                # Assign reward as target Q values
                target = transition["reward"]
            else:
                # Compute Q values on next state
                q_next = target_agent.sess.run(
                    target_agent.brain.Q_values,
                    feed_dict={
                        target_agent.brain.input_layer: transition["next_state"]
                    })

                # Compute target Q values
                target = transition["reward"] + exploration_rate * (np.amax(q_next))

            # Update Q values vector with target value
            target_q = copy.deepcopy(q)
            target_q[ACTIONS[transition["action"]]] = target

            # Train neural net
            l, _ = target_agent.sess.run(
                [target_agent.brain.loss, target_agent.brain.train_op],
                feed_dict={
                    target_agent.brain.input_layer: transition["state"],
                    target_agent.brain.target_Q: target,
                    target_agent.brain.Q_values: q
                })

            # Add loss and reward to sim metrics for later evaluation
            self.metrics["loss"].append(l)
            self.metrics["reward"].append(transition["reward"])


if __name__ == '__main__':
    print()
