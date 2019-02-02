
def adjacent_cells(adj_x, adj_y, max_x, max_y, curr_obs, combos):

    adj = [(adj_x + 1, adj_y), (adj_x - 1, adj_y),
           (adj_x, adj_y + 1), (adj_x, adj_y - 1)]

    return [item for item in adj
            if 0 < item[0] < max_x and 0 < item[1] < max_y and
            item not in curr_obs and item in combos]

