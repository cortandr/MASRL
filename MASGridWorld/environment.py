from Agent.agent import Agent, DummyAgent
from random import randint
from utils import *
import random
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

        self.obstacles = self.generate_random_obstacles(n_rows, n_cols)
        team_agents, team_opponents = self.generate_random_agents(
                                                    n_agents=n_agents,
                                                    n_opponents=n_opponents,
                                                    rows=n_rows,
                                                    cols=n_cols
                                                )

        self.agents = [Agent(pos, i == 0) for i, pos in enumerate(team_agents)]
        self.opponents = [DummyAgent(position=pos) for pos in team_opponents]

    @property
    def grid(self):
        grid = np.zeros((self.n_rows, self.n_cols), np.int8)
        for a in self.agents:
            x, y = a.get_position()
            grid[x][y] = 1
        for a in self.opponents:
            x, y = a.get_position()
            grid[x][y] = 2
        for a in sum(self.obstacles, []):
            x, y = a
            grid[x][y] = -1
        return grid

    @property
    def brain_input_grid(self):

        grid = np.zeros((self.n_rows, self.n_cols, 4), np.int8)

        # Background channel
        for a in sum(self.obstacles, []):
            x, y = a
            grid[x][y][0] = 1

        # Opponents channel
        for a in self.opponents:
            x, y = a.get_position()
            grid[x][y][1] = 1

        # Allies channel
        for a in self.agents:
            x, y = a.get_position()
            grid[x][y][2] = 1

        return grid

    def allowed_moves(self, agent):
        allowed = []

        if agent.get_pos().x + 1 <= self.n_cols:
            allowed.append('r')

        if agent.get_pos().x - 1 <= self.n_cols:
            allowed.append('l')

        if agent.get_pos().y + 1 <= self.n_rows:
            allowed.append('u')

        if agent.get_pos().y - 1 <= self.n_rows:
            allowed.append('d')

        return allowed

    def step(self, direction, agent):
        pass

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
        x_axis = [i for i in range(0, rows)]
        y_axis = [j for j in range(0, cols)]
        combos = [(x, y) for x in x_axis for y in y_axis]

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


    @staticmethod
    def generate_random_obstacles(rows, cols):
        """
        Generate random obstacles based on grid dimensions
        :param rows: x dimension  of grid
        :param cols: y dimension of grid
        :return:
        """

        # TODO: avoid square shaped obstacles

        # Decide number of obstacles and their length
        n_obs = int((rows + cols)//4)
        obs_length = [randint(2, int(rows//2)) for _ in range(n_obs)]

        # Generate all x, y combinations
        x_axis = [i for i in range(0, rows)]
        y_axis = [j for j in range(0, cols)]
        combos = [(x, y) for x in x_axis for y in y_axis]

        obs_list = list()

        for _ in range(n_obs):
            obs = list()
            # Random obstacle length
            length = random.choice(obs_length)
            del obs_length[obs_length.index(length)]

            # Random position from grid
            choice_pos = random.choice(combos)
            obs.append(choice_pos)
            del combos[combos.index(choice_pos)]

            for _ in range(length-1):
                last_x, last_y = obs[-1]
                adj_cells = adjacent_cells(last_x, last_y, rows, cols, obs, combos)
                if not adj_cells:
                    continue
                next_cell = random.choice(adj_cells)
                obs.append(next_cell)
                del combos[combos.index(next_cell)]

            obs_list.append(obs)

        return obs_list


if __name__ == '__main__':
    test = Environment(10, 10, 3, 3)
    grid = test.brain_input_grid
