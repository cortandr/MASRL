from mas_deepRL.Agent.agent import Agent, DummyAgent
from random import randint
from mas_deepRL.GridWorld.utils import *
import random
from sklearn.cluster import KMeans


class Environment:

    """
    Environment class containing all information and operations regarding the environment the agents will be acting in.
    Agents are represented in the env as 1s
    Obstacles are represented as -1s
    Opponents are represented as 2s
    Obstacles are generated randomly, same goes for agents. The latter can be either grouped by team or spawn randomly in the grid
    """

    def __init__(self, n_rows, n_cols, n_agents, n_opponents):
        self.n_rows = n_rows
        self.n_cols = n_cols

        obstacles = self.generate_random_obstacles(n_rows, n_cols)
        team_agents, team_opponents = self.generate_random_agents(
                                                    n=n_agents + n_opponents,
                                                    obs=obstacles,
                                                    rows=n_rows,
                                                    cols=n_cols
                                                )

        self.agents = [Agent(position=pos) for pos in team_agents]
        self.opponents = [DummyAgent(position=pos) for pos in team_opponents]

        self.grid = self.grid_init(
            r=n_rows,
            c=n_cols,
            team_agents=team_agents,
            team_opponents=team_opponents,
            obstacles=obstacles
        )

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

    def move_agent(self, direction, agent):
        pass

    def grid_init(self, r, c, team_agents, team_opponents, obstacles):

        grid = [[0 for _ in range(c)] for _ in range(r)]
        obstacles_list = sum(obstacles, [])

        for i in range(r):
            for j in range(c):
                if (i, j) in team_agents:
                    grid[i][j] = 1
                elif (i, j) in team_opponents:
                    grid[i][j] = 2
                elif (i, j) in obstacles_list:
                    grid[i][j] = -1
                else:
                    grid[i][j] = 0

        for row in grid:
            print(row)

        return grid

    @staticmethod
    def generate_random_agents(n, obs, rows, cols, cluster=True):
        """
        Generates and places random agents in the environment
        :param n: number of agents to create
        :param obs: obstacles positions use to avoid conflicting agent creation
        :param rows: x dimension of grid
        :param cols: y dimension of grid
        :param cluster: boolean telling whether or not to cluster the to team's position or not
        :return:
        """

        ag_pos = list()

        x_axis = [i for i in range(0, rows)]
        y_axis = [j for j in range(0, cols)]
        combos = [(x, y) for x in x_axis for y in y_axis]

        obs_list = sum(obs, [])

        for item in obs_list:
            if item in combos:
                del combos[combos.index(item)]

        for _ in range(n):

            pos_choice = random.choice(combos)

            ag_pos.append(pos_choice)

            del combos[combos.index(pos_choice)]

        team_agents = []
        team_opponents = []
        if cluster:
            # TODO: make even sized clusters
            model = KMeans(n_clusters=2)
            model.fit(ag_pos)
            labels = model.labels_
            team_agents = [pos for i, pos in enumerate(ag_pos)
                           if labels[i] == 0]
            team_opponents = [pos for i, pos in enumerate(ag_pos)
                              if labels[i] == 1]
        # TODO: random teams

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

        n_obs = int((rows + cols)//4)
        obs_length = [randint(2, int(rows//2)) for _ in range(n_obs)]

        x_axis = [i for i in range(0, rows)]
        y_axis = [j for j in range(0, cols)]
        combos = [(x, y) for x in x_axis for y in y_axis]

        obs_list = list()

        for _ in range(n_obs):
            obs = list()
            length = random.choice(obs_length)
            del obs_length[obs_length.index(length)]

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
