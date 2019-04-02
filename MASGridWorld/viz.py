from environment import Environment
import numpy as np
import cv2
from PIL import Image
import os


class Viz(object):

    def __init__(self, img_size, colors=None, save_dir=None):
        """
        :param img_size: Size of image side (in pixels).
        :param colors: Dict with keys the matrix values and value the color
                in (B,G,R). Sould contain field "default" for unused cell.
        """
        self.img_size = img_size
        if colors:
            self.colors = colors
        else:
            self.colors = {'default': (150, 150, 150),
                            0: (255, 255, 255),
                            1: (  0, 255,   0),
                            2: (  0,   0, 255),
                           -1: (  0,   0,   0)}
        self.save_dir = save_dir

    def single_frame(self, environment_grid):
        """
        Creates a single image from the environment.
        :param environment: Enviroment object to visualize.
        :return: frame
        """

        n_rows = len(environment_grid)
        n_cols = len(environment_grid[0])

        cell_height = self.img_size / (n_rows + (n_rows + 1) * 0.1)
        borders_between_rows = cell_height * 0.1
        cell_width = self.img_size / (n_cols + (n_cols + 1) * 0.1)
        borders_between_cols = cell_width * 0.1

        img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img[:, :] = self.colors["default"]

        env_matrix = environment_grid

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

        return img

    def create_gif(self, list_frames, name, duration=500, loop=1):
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if not self.save_dir.endswith('/'): self.save_dir += '/'
            name = self.save_dir + name
        frames = [Image.fromarray(img) for img in list_frames]
        frames[0].save(name + '.gif', format='GIF', append_images=frames[1:],
                       save_all=True, duration=duration, loop=loop)


if __name__ == '__main__':

    #test = Environment(10, 10, 3, 3)
    viz = Viz(600)
    multiple_envs = [Environment(10,10,3,3) for i in range(15)]
    frames = [viz.single_frame(env) for env in multiple_envs]
    viz.create_gif(frames, 'test_gif')
