from environment import Environment
import numpy as np
import cv2


class Viz(object):

    def __init__(self, img_size, colors):
        """
        :param img_size: Size of image side (in pixels).
        :param colors: Dict with keys the matrix values and value the color
                in (B,G,R). Sould contain field "default" for unused cell.
        """
        self.img_size = img_size
        self.colors = colors

    def single_frame(self, environment):
        """
        Creates a single image from the environment.
        :param environment: Enviroment object to visualize.
        :return: Nothing
        """
        if not isinstance(environment, Environment):
            print('An Environment was not provided')
            return

        n_rows = environment.n_rows
        n_cols = environment.n_cols

        cell_height = self.img_size / (n_rows + (n_rows + 1) * 0.1)
        borders_between_rows = cell_height * 0.1
        cell_width = self.img_size / (n_cols + (n_cols + 1) * 0.1)
        borders_between_cols = cell_width * 0.1

        img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img[:, :] = self.colors["default"]

        env_matrix = environment.grid

        for r in range(n_rows):
            for c in range(n_cols):
                top_left = (int(c * cell_width + (c + 1) * borders_between_cols),
                            int(r * cell_height + (r + 1) * borders_between_rows))
                bot_right = (int((c + 1) * (cell_width + borders_between_cols)),
                             int((r + 1) * (cell_height + borders_between_rows)))
                cv2.rectangle(img,
                              top_left,
                              bot_right,
                              self.colors[env_matrix[r][c]],
                              -1)

        cv2.imshow('Env', img)
        cv2.waitKey(0)


if __name__ == '__main__':

    test = Environment(10, 10, 3, 3)
    colors = {
        'default': (150, 150, 150),
         0: (255, 255, 255),
         1: (  0, 255,   0),
         2: (  0,   0, 255),
        -1: (  0,   0,   0)}
    viz = Viz(600, colors)
    viz.single_frame(test)
