import math


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