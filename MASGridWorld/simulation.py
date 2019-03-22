import numpy as np
from environment import Environment
import random
import copy
from viz import Viz


class Sim:

    def __init__(self, allies, opponents, world_size, n_games,
                 train_batch_size, viz=None, viz_execution=None):
        self.allies = allies
        self.opponents = opponents
        self.world_size = world_size
        self.moves_limit = 20
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

        self.viz = viz
        self.viz_execution = viz_execution

    def run(self):
        """
        Runs general simulation and takes care of experience table population
        and agents training
        :return:
        """

        sim = 0

        while sim < self.n_games:

            sim_moves = 0

            # Get agent that is training
            training_agent = next(
                filter(lambda ag: ag.training, self.environment.agents))

            while sim_moves < self.moves_limit and not self.environment.is_over():

                # Current state
                curr_state = self.environment.brain_input_grid

                # Apply step in Environment
                self.environment.step(curr_state.copy())

                # Get agent chosen action
                action = training_agent.get_chosen_action()

                # Get reward
                reward = self.get_reward()

                # Get state after chosen action is applied
                next_state = self.environment.brain_input_grid

                # Store transition in replay table
                self.experience_replay.append({
                    "state": curr_state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                })

                sim_moves += 1

            # Train every 2 simulations
            if (sim+1) % 2 == 0:
                self.train_ally()

            # Update training net every 10 simulations
            if (sim+1) % 10 == 0:
                self.update_target_net()

            if self.viz and self.viz_execution and self.viz_execution(sim):
                self.environment.reset()
                sim_moves = 0
                env_seq = [self.environment]
                while sim_moves < self.moves_limit and not self.environment.is_over():
                    curr_state = self.environment.brain_input_grid
                    self.environment.step(curr_state.copy())
                    env_seq.append(self.environment)
                    sim_moves += 1
                frames = [self.viz.single_frame(env) for env in env_seq]
                viz.create_gif(frames,
                               name='simulation_%d' % sim)

            self.environment.reset()
            sim += 1

    def update_target_net(self):
        self.environment.target_net = copy.deepcopy(self.environment.training_net)

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
            target_q[transition["action"]] = target

            # Train neural net
            l, _ = target_agent.sess.run(
                [target_agent.brain.loss, target_agent.brain.train_op],
                feed_dict={
                    target_agent.brain.input_layer: transition["state"],
                    target_agent.brain.target_Q: target_q,
                    target_agent.brain.Q_values: q
                })

            # Add loss and reward to sim metrics for later evaluation
            self.metrics["loss"].append(l)
            self.metrics["reward"].append(transition["reward"])

    def get_reward(self):
        # Reward 1 -> number of agents
        reward1 = len(self.environment.agents) - \
                  (len(self.environment.opponents) ** 2)

        bottom_limit = self.allies - (self.opponents ** 2)
        # top limit is self.allies
        reward_range = self.allies - bottom_limit
        shift = (reward_range / 2) - self.allies
        reward1 = (reward1 + shift) / (reward_range / 2)

        # Reward 2 -> board coverage
        def floodfill(agent, env):
            matrix = np.zeros((env.n_rows, env.n_cols))
            open_list = [agent.get_position()]
            done_list = list()
            while len(open_list) > 0:
                current = open_list.pop()
                done_list.append(current)
                adjacent_pos = env.allowed_moves()
                for p in adjacent_pos:
                    if p not in done_list:
                        matrix[p[0]][p[1]] = matrix[current[0]][current[1]] + 1
                        open_list.append(p)
            return matrix

        distances_agents = [floodfill(a, self.environment)
                              for a in self.environment.agents]
        distances_opponents = [floodfill(a, self.environment)
                                 for a in self.environment.opponents]

        distances_agents = np.array(distances_agents).min(axis=0)
        distances_opponents = np.array(distances_opponents).min(axis=0)

        combined = distances_agent - distances_opponents
        reward2 = sum((combined < 0).astype(int)) - sum((combined > 0).astype(int))
        reward2 = reward2 / (env.n_rows * env.n_cols)

        return (reward1 + reward2) / 2


if __name__ == '__main__':
    viz = Viz(600, save_dir='gifs/')
    def viz_execution(sim_number):
        return sim_number in [5, 15, 30, 50, 100]
    sim = Sim(5, 5, (10, 10), 10, 32, viz, viz_execution)
    sim.run()
