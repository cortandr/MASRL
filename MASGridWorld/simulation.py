import numpy as np
from environment import Environment
import random
import pickle
from utils import bfs
from viz import Viz
import os
import copy


class Sim:

    def __init__(self, allies, opponents, world_size, n_games,
                 train_batch_size, replay_mem_limit,
                 viz=None, viz_execution=None, train_saving=None):
        """

        :param allies: number of allie
        :param opponents: number of opponents
        :param world_size: tuple (n_rows, n_cols)
        :param n_games: number of games to play
        :param train_batch_size: size of training batch
        :param replay_mem_limit: replay memory size limit
        :param viz: Visualization object
        :param viz_execution: Function to decide when to run visualization (returns bool)
        :param train_saving: Function to decide when to save the model (returns bool)
        """
        self.allies = allies
        self.opponents = opponents
        self.world_size = world_size
        self.moves_limit = 20
        self.experience_replay = list()
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
        self.r1_w = 0.5
        self.r2_w = 0.5

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
            if len(self.experience_replay) > self.replay_mem_limit:
                prune = self.replay_mem_limit // 5
                self.experience_replay = self.experience_replay[prune:]

            # Get agent that is training
            training_agent = next(
                filter(lambda ag: ag.training, self.environment.agents))

            while sim_moves < self.moves_limit and not self.environment.is_over():

                # Current state
                curr_state = self.environment.brain_input_grid

                # Apply step in Environment
                reward = self.environment.step()

                # Get agent chosen action
                action = training_agent.get_chosen_action()

                # Get state after chosen action is applied
                next_state = self.environment.brain_input_grid

                self.metrics["reward"].append(reward)

                # Store transition in replay table
                self.experience_replay.append({
                    "state": np.array([curr_state]),
                    "action": action,
                    "next_state": np.array([next_state]) if next_state is not None else None,
                    "reward": reward,
                })

                sim_moves += 1

            sim += 1

            # if sim < 50000:
            #     self.r1_w += 8e-6
            #     self.r2_w -= 8e-6
            if sim < 10000:
                # training_agent.brain.exploration_rate -= 5e-5
                training_agent.brain.temp -= 5e-5

            # Train every 2 simulations
            if sim % 10 == 0:
                self.train_ally(sim/2)

            # Update training net every 10 simulations
            if sim % 250 == 0:
                self.update_target_net()
                print("-------------------------------")
                print("Sim checkpoint : {}".format(sim))
                print("Average Loss : {}".format(
                    sum(self.metrics["loss"])/len(self.metrics["loss"])))
                print("Average reward : {}".format(
                    sum(self.metrics["reward"])/len(self.metrics["reward"])))

            # Create GIF
            if self.viz and self.viz_execution and self.viz_execution(sim):
                self.environment.reset(False)
                sim_moves = 0
                env_seq = [copy.deepcopy(self.environment.grid)]
                while sim_moves < self.moves_limit and not self.environment.is_over():
                    curr_state = self.environment.brain_input_grid
                    self.environment.step()
                    env_seq.append(copy.deepcopy(self.environment.grid))
                    sim_moves += 1
                frames = [self.viz.single_frame(env) for env in env_seq]
                viz.create_gif(frames, name='simulation_%d' % sim)

            if self.train_saving is not None and self.train_saving(sim):
                save_path = 'Models/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                with open(save_path + 'metrics' + '.pkl', "wb") as f:
                    pickle.dump(self.metrics, f)
                # Save model
                training_agent.brain.save_model(save_path, str(sim))

            self.environment.reset()

    def update_target_net(self):
        p = self.environment.training_net.save_model("Models/", "temporary_update")
        self.environment.target_net.load_model(p)

    def train_ally(self, index):
        """
        Takes a random batch from experience replay memory and uses it to train
        the agent's brain NN
        :return:
        """
        if len(self.experience_replay) < self.training_batch_size:
            return

        # Sample batch fro replay memory
        mini_batch = random.sample(
            self.experience_replay,
            self.training_batch_size)

        discount_rate = self.environment.training_net.discount_rate

        training_net = self.environment.training_net
        target_net = self.environment.target_net

        # Build input and target network batches
        input_batch = np.ndarray(
            shape=(
                self.training_batch_size,
                self.environment.n_rows,
                self.environment.n_cols,
                4
            ))

        target_batch = np.ndarray(shape=(self.training_batch_size, 1, 8))

        for i, transition in enumerate(mini_batch):

            # Check for possible ending state
            if transition["next_state"] is None:
                # Assign reward as target Q values
                target = transition["reward"]
            else:
                # Compute Q values on next state
                q_next = target_net.sess.run(
                    target_net.Q_values,
                    feed_dict={
                        target_net.input_layer: transition["next_state"],
                    })

                # Compute target Q value
                target = transition["reward"] + discount_rate * (np.amax(q_next))

            # Update Q values vector with target value
            target_q = target_net.sess.run(
                    target_net.Q_values,
                    feed_dict={
                        target_net.input_layer: transition["state"],
                    })
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

        q = training_net.sess.run(
            training_net.Q_values,
            feed_dict={
                training_net.input_layer: input_batch,
            })

        # Train neural net
        l, _ = training_net.sess.run(
            [training_net.loss, training_net.train_op],
            feed_dict={
                training_net.input_layer: input_batch,
                training_net.target_Q: target_batch,
                training_net.Q_values: q
            })

        self.metrics["loss"].append(l)

        # Get summary
        s = training_net.sess.run(
            training_net.merged_summary,
            feed_dict={
                training_net.input_layer: input_batch,
                training_net.target_Q: target_batch,
                training_net.Q_values: q
            })

        training_net.writer.add_summary(s, index)


if __name__ == '__main__':

    viz = Viz(600, save_dir='gifs/')

    def viz_execution(sim_number):
        return sim_number % 250 == 0 or sim_number == 1

    sim = Sim(
        allies=5,
        opponents=5,
        world_size=(10, 10),
        n_games=200000,
        train_batch_size=32,
        replay_mem_limit=100000,
        viz=viz,
        viz_execution=viz_execution,
        train_saving=viz_execution)
    sim.run()
