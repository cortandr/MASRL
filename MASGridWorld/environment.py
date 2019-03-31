from Agent.agent import Agent, DummyAgent
from Agent.NeuralNet import Brain
from random import randint
from utils import *
import random
import copy
from sklearn.cluster import KMeans
import numpy as np


class Environment:

    """
    Environment class containing all information and operations regarding
    the environment the agents will be acting in.
    Agents are represented in the env as 1s
    Obstacles are represented as -1s
    Opponents are represented as 2s
    Obstacles are generated randomly, same goes for agents.
    The latter can be either grouped by team or spawn randomly in the grid
    """

    def __init__(self, n_rows, n_cols, n_agents, n_opponents):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_agents = n_agents
        self.n_opponents = n_opponents

        self.obstacles = None
        self.allowed_moves_per_position = None
        self.generate_random_obstacles(n_rows, n_cols)
        team_agents, team_opponents = self.generate_random_agents(
            n_agents=n_agents,
            n_opponents=n_opponents,
            rows=n_rows,
            cols=n_cols,
            cluster=False
        )

        self.agents = [Agent(pos, i == 0) for i, pos in enumerate(team_agents)]

        self.training_net = Brain(True)
        self.target_net = Brain(False)

        for a in self.agents:
            if a.training:
                a.brain = self.training_net
            else:
                a.brain = self.target_net

        self.opponents = [DummyAgent(position=pos) for pos in team_opponents]
        self.r1_w = 0.5
        self.r2_w = 0.2
        self.r3_w = 0.3

    @property
    def grid(self):

        # Initialize empty grid
        grid = np.zeros((self.n_rows, self.n_cols), np.int8)

        # Place agents
        for a in self.agents:
            x, y = a.get_position()
            grid[x][y] = 1

        # Place opponents
        for a in self.opponents:
            x, y = a.get_position()
            grid[x][y] = 2

        # Place obstacles
        for a in sum(self.obstacles, []):
            x, y = a
            grid[x][y] = -1

        return grid

    @property
    def brain_input_grid(self):

        if not self.opponents:
            return None

        # Initialize empty grid
        grid = np.zeros((self.n_rows, self.n_cols, 4), np.int8)

        # Background channel
        for a in sum(self.obstacles, []):
            x, y = a
            grid[x][y][0] += 1.0

        # Opponents channel
        for a in self.opponents:
            x, y = a.get_position()
            grid[x][y][1] += 1.0

        # Allies channel
        for a in self.agents:
            x, y = a.get_position()
            grid[x][y][2] += 1.0

        return grid

    def create_allowed_moves(self):
        moves = {}
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                ring = [(row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
                        (row, col + 1), (row + 1, col + 1), (row + 1, col),
                        (row + 1, col - 1), (row, col - 1)]
                ring = map(lambda pos: pos if pos not in sum(self.obstacles, []) and \
                                     0 <= pos[0] < self.n_rows and \
                                     0 <= pos[1] < self.n_cols \
                                  else None, ring)
                moves[(row, col)] = list(ring)
        return moves

    def allowed_moves(self, agent):
        if isinstance(agent, tuple):
            return self.allowed_moves_per_position[agent]
        return self.allowed_moves_per_position[agent.get_position()]

    def step(self):

        for dummy in self.opponents:
            allowed_moves = self.allowed_moves(dummy)
            dummy.choose_action(
                allowed_moves,
                self.agents)

        state_tensor = np.array([self.brain_input_grid.copy()])

        for agent in self.agents:
            allowed_moves = self.allowed_moves(agent)
            temp_state_tensor = copy.deepcopy(state_tensor)
            x, y = agent.get_position()
            temp_state_tensor[0][x][y][3] = 1
            agent.choose_action(allowed_moves, temp_state_tensor)

        # Check for overlapping opponents
        agents = [a.get_position() for a in self.agents]

        # Eaten opponents
        eaten_opponents = [oppo for oppo in self.opponents
                           if oppo.get_position() in agents]

        training_agent = next(filter(lambda a: a.training, self.agents))
        training_ate = training_agent.get_position() in eaten_opponents
        # Delete eaten opponents
        self.opponents = [oppo for oppo in self.opponents
                           if oppo.get_position() not in agents]

        return self.get_reward(int(training_ate))

    def generate_random_agents(self, n_agents, n_opponents, rows, cols, cluster=True):
        """
        Generates and places random agents in the environment
        :param n_agents: number of agents to create
        :param n_opponents: number of opponents to create
        :param rows: x dimension of grid
        :param cols: y dimension of grid
        :param cluster: boolean telling whether or not to cluster
        the team's position or not
        :return:
        """

        n = n_agents+n_opponents

        ag_pos = list()

        # Generate all x, y combinations
        combos = [(x, y) for x in range(0, rows) for y in range(0, cols)]

        obs_list = sum(self.obstacles, [])

        for item in obs_list:
            if item in combos:
                del combos[combos.index(item)]

        for _ in range(n):
            # Randomly choose agents positions
            pos_choice = random.choice(combos)
            ag_pos.append(pos_choice)
            del combos[combos.index(pos_choice)]

        if cluster:
            # Cluster agents in two team
            model = KMeans(n_clusters=2)
            model.fit(ag_pos)
            c1, c2 = model.cluster_centers_

            # Find distances from first center
            dist_c1 = [{"pos": pos, "dist": point_dist(c1, pos)} for pos in ag_pos]

            # Separate positions into teams
            team_agents = sorted(dist_c1, key=lambda k: k["dist"], reverse=True)
            team_agents = [item["pos"] for item in team_agents[:n_agents]]
        else:
            team_agents = [pos for pos, _ in zip(ag_pos, range(n_agents))]

        team_opponents = [pos for pos in ag_pos if pos not in team_agents]

        return team_agents, team_opponents

    def generate_random_obstacles(self, rows, cols):
        """
        Generate random obstacles based on grid dimensions
        :param rows: x dimension  of grid
        :param cols: y dimension of grid
        :return:
        """

        # valid_obs = False
        #
        # while not valid_obs:
        #
        #     # Decide number of obstacles and their length
        #     n_obs = randint(1, int((rows + cols) // 4))
        #     obs_length = [randint(2, int(rows // 2)) for _ in range(n_obs)]
        #
        #     # Generate all x, y combinations
        #     combos = [(x, y) for x in range(0, rows) for y in range(0, cols)]
        #
        #     obs_list = list()
        #
        #     for _ in range(n_obs):
        #         obs = list()
        #         # Random obstacle length
        #         length = random.choice(obs_length)
        #         del obs_length[obs_length.index(length)]
        #
        #         # Random position from grid
        #         choice_pos = random.choice(combos)
        #         obs.append(choice_pos)
        #         del combos[combos.index(choice_pos)]
        #
        #         for _ in range(length - 1):
        #             last_x, last_y = obs[-1]
        #             adj_cells = adjacent_cells(last_x, last_y, rows, cols, obs,
        #                                        combos)
        #             if not adj_cells:
        #                 continue
        #             next_cell = random.choice(adj_cells)
        #             obs.append(next_cell)
        #             del combos[combos.index(next_cell)]
        #
        #         obs_list.append(obs)

        obs_list = [
            [(3, 3), (3, 4), (3, 5), (3, 6)],
            [(4, 3), (4, 4), (4, 5), (4, 6)],
            [(5, 3), (5, 4), (5, 5), (5, 6)],
            [(6, 3), (6, 4), (6, 5), (6, 6)],
        ]

        self.obstacles = obs_list
        self.allowed_moves_per_position = self.create_allowed_moves()

                # Check validity of obs positions / avoid trapping agents
                # start_pos = random.choice(combos)
                # reachability_matrix = bfs(start_pos, self)
                # valid_obs = sum(sum((reachability_matrix > 0).astype(int))) + 1 + len(sum(obs_list, []))
                # valid_obs = valid_obs == self.n_rows * self.n_cols

    def reset(self, training=True):
        self.generate_random_obstacles(self.n_rows, self.n_cols)
        team_agents, team_opponents = self.generate_random_agents(
            n_agents=self.n_agents,
            n_opponents=self.n_opponents,
            rows=self.n_rows,
            cols=self.n_cols,
            cluster=False
        )

        self.agents = [Agent(pos, i == 0) for i, pos in enumerate(team_agents)]
        self.opponents = [DummyAgent(position=pos) for pos in team_opponents]

        for a in self.agents:
            if a.training:
                a.brain = self.training_net
            else:
                a.brain = self.target_net

        self.allowed_moves_per_position = self.create_allowed_moves()
        self.training_net.training = training

    def is_over(self):
        return len(self.opponents) == 0

    def get_reward(self, training_ate):
        # Reward 1 -> number of agents
        reward1 = (self.n_agents - self.n_opponents)**2

        range1 = self.n_agents**2
        # range1 = self.allies ** 2 - (self.allies - self.opponents) ** 2
        # reward1 = (reward1 - (self.allies - (self.opponents ** 2))) / range1
        reward1 = reward1 / range1

        # bottom_limit = self.allies - (self.opponents ** 2)
        # # top limit is self.allies
        # reward_range = self.allies - bottom_limit
        # shift = (reward_range / 2) - self.allies
        # reward1 = (reward1 + shift) / (reward_range / 2)

        # Reward 2 -> board coverage
        combined = self.reachability()
        reward2 = sum(sum((combined < 0).astype(int))) - \
                  sum(sum((combined > 0).astype(int)))
        range2 = (self.n_rows*self.n_cols) - \
                 (-(self.n_rows*self.n_cols))
        reward2 = reward2 / (self.n_rows * self.n_cols)
        reward2 = (reward2 - (-100)) / range2

        return self.r1_w*reward1 + self.r2_w*reward2 + self.r3_w*training_ate

    def reachability(self):
        # Do BFS for ally agents and opponents
        distances_agents = [bfs(a, self)
                            for a in self.agents]
        distances_opponents = [bfs(a, self)
                               for a in self.opponents]

        # In case there is no more opponents the list above is empty
        if not distances_opponents:
            distances_opponents = [np.full((
                self.n_rows,
                self.n_cols),
                self.n_rows * self.n_cols)]

        # Calculate the minimum distance to the cells for the whole teams
        distances_agents = np.array(distances_agents).min(axis=0)
        distances_opponents = np.array(distances_opponents).min(axis=0)

        # Combined has a negative value for cells closer to allies and positive
        # for cells closer to opponents
        return distances_agents - distances_opponents


if __name__ == '__main__':
    test = Environment(10, 10, 3, 3)
    grid = test.brain_input_grid
