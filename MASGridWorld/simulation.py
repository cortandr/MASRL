import numpy as np
from environment import Environment
import pickle
from viz import Viz
import os
import copy
from ReplayMemory import ReplayMemory
from collections import OrderedDict
import pandas as pd


class Sim:

    def __init__(
            self,
            allies,
            opponents,
            world_size,
            n_games,
            train_batch_size,
            replay_mem_limit,
            training_rate=10,
            update_rate=500,
            sim_moves_limit=30,
            exploration_steps=200000,
            exploration_range=(0.1, 1.0),
            viz=None,
            viz_execution=None,
            train_saving=None):

        self.allies = allies
        self.opponents = opponents
        self.world_size = world_size
        self.moves_limit = sim_moves_limit
        self.training_rate = training_rate
        self.policy_dist_rate = update_rate
        self.exploration_steps = exploration_steps
        self.exploration_range = exploration_range
        self.exploration_step_value = \
            (exploration_range[1]-exploration_range[0])/exploration_steps
        self.experience_replay = ReplayMemory(
            batch_size=train_batch_size,
            table_size=replay_mem_limit
        )
        self.training_batch_size = train_batch_size
        self.n_games = n_games
        self.replay_mem_limit = replay_mem_limit
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
        self.train_saving = train_saving

    def run(self):
        """
        Runs general simulation and takes care of experience table population
        and agents training
        :return:
        """

        sim = 0

        while sim < self.n_games:

            sim_moves = 0

            # Prune replay memory by 1/5 if over limit size
            if self.experience_replay.is_full():
                self.experience_replay.refresh()

            # Get agent that is training
            training_agent = next(
                filter(lambda ag: ag.training, self.environment.agents))

            episode_reward = []

            while sim_moves < self.moves_limit and not self.environment.is_over():

                # Apply step in Environment
                curr_state, next_state, reward = self.environment.step(
                    terminal_state=sim_moves == self.moves_limit - 1
                )

                # Get agent chosen action
                action = training_agent.get_chosen_action()

                episode_reward.append(reward)

                # Store transition in replay table
                self.experience_replay.insert({
                    "state": curr_state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward["reward_value"],
                })

                sim_moves += 1

            episode_reward = pd.DataFrame(episode_reward)

            # Append new collected avg episode reward
            self.metrics["reward"].append(OrderedDict({
                "Simulation No.": sim,
                "Avg Cumulative Reward": episode_reward["reward_value"].mean(),
                "Global Capture Reward": episode_reward["Global Capture Reward"].max(),
                "Local Capture Reward": episode_reward["Local Capture Reward"].mean(),
                "Reachability Reward": episode_reward["Reachability Reward"].mean()
            }))

            sim += 1

            # Diminishing exploration rate
            if sim < self.exploration_steps:
                training_agent.brain.policy["boltzmann"].update_exploration_rate(
                    new_er=self.exploration_step_value
                )

            # Train every N simulations
            if sim % self.training_rate == 0:
                self.train_ally()

            # Update training net every 10 simulations
            if sim % self.policy_dist_rate == 0:
                self.update_target_net()
                print("---------------------------------------------")
                print("Sim No. : {}".format(sim))
                print("Average Loss : {}".format(
                    sum(self.metrics["loss"])/len(self.metrics["loss"])))
                mdf = pd.DataFrame(self.metrics["reward"])
                print("Average reward : {}".format(mdf["Avg Cumulative Reward"].mean()))
                print("Average GCR : {}".format(mdf["Global Capture Reward"].mean()))
                print("Average LCR : {}".format(mdf["Local Capture Reward"].mean()))
                print("Average RR : {}".format(mdf["Reachability Reward"].mean()))

            # Create GIF
            self.visualize_gif(sim)

            # Save model checkpoint
            self.save_checkpoint(training_agent, sim)

            # Reset game setting
            self.environment.reset()

    def update_target_net(self):
        self.environment.training_net.save_model("Models/", "temporary_update")
        self.environment.target_net.load_model("Models/", "temporary_update")

    def train_ally(self):
        """
        Takes a random batch from experience replay memory and uses it to train
        the agent's brain NN
        :return:
        """
        if not self.experience_replay.can_replay():
            return

        # Sample batch fro replay memory
        mini_batch = self.experience_replay.sample()

        training_net = self.environment.training_net
        target_net = self.environment.target_net

        X, y = self.create_training_batch(
            target_net=target_net,
            training_net=training_net,
            mini_batch=mini_batch,
        )

        history = training_net.train(X, y, self.training_batch_size)

        self.metrics["loss"] += history.history["loss"]

    def create_training_batch(self, target_net, training_net, mini_batch):

        # Build input and target network batches
        input_batch = np.ndarray(
            shape=(
                self.training_batch_size,
                self.environment.n_rows,
                self.environment.n_cols,
                4
            ))

        target_batch = np.ndarray(shape=(self.training_batch_size, 1, 8))

        gamma = training_net.discount_rate

        for i, transition in enumerate(mini_batch):

            # Check for possible ending state
            if transition["next_state"] is None:
                # Assign reward as target Q values
                target = transition["reward"]
            else:
                # Compute Q values on next state
                q_next = target_net.model.predict(transition["next_state"])[0]

                # Filter not allowed moves
                agent_position = np.argwhere(transition["state"][0, :, :, 0])
                allowed_moves = self.environment.allowed_moves(
                    (agent_position[0][0], agent_position[0][1])
                )
                moves_mask = np.array(
                    [1 if pos else np.nan for pos in allowed_moves])
                masked_q_next = q_next * moves_mask

                # Compute target Q value
                target = transition["reward"] + gamma * (
                    np.nanmax(masked_q_next))

            # Update Q values vector with target value
            target_q = training_net.model.predict(transition["state"])[0]
            target_q[0][transition["action"]] = target

            input_state = np.reshape(
                transition["state"],
                newshape=(
                    transition["state"].shape[1],
                    transition["state"].shape[2],
                    transition["state"].shape[3],
                ))

            input_batch[i] = input_state
            target_batch[i] = target_q

        return input_batch, target_batch

    def visualize_gif(self, sim_number):

        # Create GIF
        if self.viz and self.viz_execution and self.viz_execution(sim_number):

            # Reset game
            self.environment.reset(False)

            # play game run
            sim_moves = 0
            env_seq = [copy.deepcopy(self.environment.grid)]
            while sim_moves < self.moves_limit and not self.environment.is_over():
                self.environment.step(False)
                env_seq.append(copy.deepcopy(self.environment.grid))
                sim_moves += 1

            # Connect frame as save gif
            frames = [self.viz.single_frame(env) for env in env_seq]
            viz.create_gif(frames, name='simulation_%d' % sim_number)

    def save_checkpoint(self, training_agent, sim_number):

        if self.train_saving is not None and self.train_saving(sim_number):
            save_path = 'Models/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            with open(save_path + 'metrics' + '.pkl', "wb") as f:
                pickle.dump(self.metrics, f)
            # Save model
            training_agent.brain.save_model(save_path, str(sim_number))


if __name__ == '__main__':

    viz = Viz(600, save_dir='gifs/')

    def viz_execution(sim_number):
        return sim_number % 250 == 0 or sim_number == 1

    sim = Sim(
        allies=5,
        opponents=5,
        world_size=(10, 10),
        n_games=400000,
        train_batch_size=32,
        replay_mem_limit=200000,
        viz=viz,
        viz_execution=viz_execution,
        train_saving=viz_execution)
    sim.run()
