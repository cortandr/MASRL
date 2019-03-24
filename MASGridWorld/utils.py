import math
import copy
import numpy as np


def adjacent_cells(adj_x, adj_y, max_x, max_y, curr_obs, combos):

    adj = [(adj_x + 1, adj_y), (adj_x - 1, adj_y),
           (adj_x, adj_y + 1), (adj_x, adj_y - 1)]

    return [item for item in adj
            if 0 < item[0] < max_x and 0 < item[1] < max_y and
            item not in curr_obs and item in combos]


def point_dist(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def bfs(agent, env):
    """
    Calculates breadth first search starting on the agent position,
    giving the number of moves / distance to every cell.
    Obstacles will always have distance 0.
    """
    matrix = np.zeros((env.n_rows, env.n_cols))

    # Check if agent/position was passed as arg
    if isinstance(agent, tuple):
        position = agent
    else:
        position = agent.get_position()
    open_list = [position]
    done_list = [position]
    while len(open_list) > 0:
        current = open_list.pop(0)
        adjacent_pos = env.allowed_moves(current)
        current_value = matrix[current[0]][current[1]]
        for p in adjacent_pos:
            if p is not None and p not in done_list:
                # p can be None it is an obstacle or outside bounds
                matrix[p[0]][p[1]] = current_value + 1
                open_list.append(p)
                done_list.append(p)
    return matrix