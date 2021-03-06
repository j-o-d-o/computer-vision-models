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
from data.label_spec import SEMSEG_CLASS_MAPPING, OD_CLASS_MAPPING


class ShowPygame():
    """
    Callback to show results in pygame window, custom callback called in custom train step
    """
    def __init__(self, storage_path: str, semseg_params = None, od_params = None):
        """
        :param storage_path: path to directory were the image data should be stored
        """
        self._semseg_params = semseg_params
        self._od_params = od_params

        self._storage_path = storage_path
        if not os.path.exists(self._storage_path):
            print("Storage folder does not exist yet, creating: " + self._storage_path)
            os.makedirs(self._storage_path)

        self.display = None
        if self._semseg_params is not None:
            self.display = pygame.display.set_mode((self._semseg_params.INPUT_WIDTH, int(self._semseg_params.INPUT_HEIGHT + self._semseg_params.MASK_HEIGHT*2)), pygame.HWSURFACE | pygame.DOUBLEBUF)
        elif self._od_params is not None:
            self.display = pygame.display.set_mode((self._od_params.INPUT_WIDTH, int(self._od_params.INPUT_HEIGHT + self._od_params.MASK_HEIGHT*3)), pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        self._step_counter = 0

    def update_step_counter(self):
        self._step_counter += 1

    def show_semseg(self, inp, y_true, y_pred):
        inp_img = cv2.cvtColor(inp[0].numpy().astype(np.uint8), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_img = pygame.surfarray.make_surface(inp_img)
        self.display.blit(surface_img, (0, 0))
        semseg_true = cv2.cvtColor(to_3channel(y_true[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_true = pygame.surfarray.make_surface(semseg_true)
        self.display.blit(surface_y_true, (0, self._semseg_params.INPUT_HEIGHT))
        semseg_pred = cv2.cvtColor(to_3channel(y_pred[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(semseg_pred)
        self.display.blit(surface_y_pred, (0, int(self._semseg_params.INPUT_HEIGHT + self._semseg_params.MASK_HEIGHT)))

        if self._step_counter % 2000 == 0:
            pygame.image.save(self.display, f"{self._storage_path}/train_result_{self._step_counter}.png")

        pygame.display.flip()

    def show_od(self, inp, y_true, y_pred):
        heatmap_true = np.array(y_true[0][:, :, :-1]) # needed because otherwise numba makes mimimi
        heatmap_true = to_3channel(heatmap_true, List(OD_CLASS_MAPPING.items()), 0.01, True, False)
        weights = np.stack([y_true[0][:, :, -1]]*3, axis=-1)
        heatmap_pred = np.array(y_pred[0])
        heatmap_pred = to_3channel(heatmap_pred, List(OD_CLASS_MAPPING.items()), 0.01, True, False)
        # display img
        show_inp_img = inp[0][0]
        show_inp_img = show_inp_img.numpy()
        show_inp_img = show_inp_img.astype(np.uint8)
        inp_img = cv2.cvtColor(show_inp_img, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_img = pygame.surfarray.make_surface(inp_img)
        self.display.blit(surface_img, (0, 0))
        # display heatmap y_pred
        heatmap_pred = cv2.cvtColor(heatmap_pred, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(heatmap_pred)
        self.display.blit(surface_y_pred, (0, self._od_params.INPUT_HEIGHT))
        # display heatmap y_true
        heatmap_true = cv2.cvtColor(heatmap_true, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_true = pygame.surfarray.make_surface(heatmap_true)
        self.display.blit(surface_y_true, (0, int(self._od_params.INPUT_HEIGHT + self._od_params.MASK_HEIGHT)))

        if self._step_counter % 2000 == 0:
            pygame.image.save(self.display, f"{self._storage_path}/train_result_{self._step_counter}.png")

        pygame.display.flip()
