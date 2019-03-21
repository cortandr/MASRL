from environment import Environment
import numpy as np
import cv2
from PIL import Image


class Viz(object):

    def __init__(self, img_size, colors=None):
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
                           2: (255,   0,   0),
                           -1: (  0,   0,   0)}

    def single_frame(self, environment):
        """
        Creates a single image from the environment.
        :param environment: Enviroment object to visualize.
        :return: frame
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

        return img

    @staticmethod
    def create_gif(list_frames, name, duration=500, loop=1):
        frames = [Image.fromarray(img) for img in list_frames]
        frames[0].save(name + '.gif', format='GIF', append_images=frames[1:],
                       save_all=True, duration=duration, loop=loop)


if __name__ == '__main__':

    #test = Environment(10, 10, 3, 3)
    viz = Viz(600)
    multiple_envs = [Environment(10,10,3,3) for i in range(15)]
    frames = [viz.single_frame(env) for env in multiple_envs]
    viz.create_gif(frames, 'test_gif')
