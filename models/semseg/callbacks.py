import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
import pygame
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from numba.typed import List
from common.utils import to_3channel
from data.label_spec import SEMSEG_CLASS_MAPPING


class ShowPygame():
    """
    Callback to show results in pygame window, custom callback called in custom train step
    """
    def __init__(self, storage_path: str, params):
        """
        :param storage_path: path to directory were the image data should be stored
        """
        self.params = params
        self._storage_path = storage_path
        if not os.path.exists(self._storage_path):
            print("Storage folder does not exist yet, creating: " + self._storage_path)
            os.makedirs(self._storage_path)
        self.display = pygame.display.set_mode((self.params.INPUT_WIDTH, int(self.params.INPUT_HEIGHT + self.params.MASK_HEIGHT*2)), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.step_counter = 0

    def show_semseg(self, inp, y_true, y_pred):
        inp_img = cv2.cvtColor(inp[0].numpy().astype(np.uint8), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_img = pygame.surfarray.make_surface(inp_img)
        self.display.blit(surface_img, (0, 0))
        semseg_true = cv2.cvtColor(to_3channel(y_true[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_true = pygame.surfarray.make_surface(semseg_true)
        self.display.blit(surface_y_true, (0, self.params.INPUT_HEIGHT))
        semseg_pred = cv2.cvtColor(to_3channel(y_pred[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(semseg_pred)
        self.display.blit(surface_y_pred, (0, int(self.params.INPUT_HEIGHT + self.params.MASK_HEIGHT)))

        self.step_counter += 1
        if self.step_counter % 2000 == 0:
            pygame.image.save(self.display, f"{self._storage_path}/train_result_{self.step_counter}.png")

        pygame.display.flip()
