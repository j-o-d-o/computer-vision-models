import tensorflow as tf
import cv2
from tensorflow.keras.losses import Loss, categorical_crossentropy
import pygame
from tensorflow.python.keras.utils import losses_utils
from common.utils import cmap_depth, to_3channel
from data.label_spec import SEMSEG_CLASS_MAPPING
from models.dmds_ref.regularizers import joint_bilateral_smoothing


class SemsegLoss(Loss):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None, save_path=None):
        super().__init__(reduction=reduction, name=name)
        self.display = pygame.display.set_mode((640, 256*2), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.save_path = save_path
        self.step_counter = 0

    def _show_semseg(self, y_true, y_pred):
        semseg_true = cv2.cvtColor(to_3channel(y_true[0].numpy(), list(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_true = pygame.surfarray.make_surface(semseg_true)
        self.display.blit(surface_y_true, (0, 0))
        semseg_pred = cv2.cvtColor(to_3channel(y_pred[0].numpy(), list(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(semseg_pred)
        self.display.blit(surface_y_pred, (0, 256))

        if self.step_counter % 500 == 0:
            pygame.image.save(self.display, f"{self.save_path}/train_result_{self.step_counter}.png")

        pygame.display.flip()

    def call(self, y_true, y_pred):
        cc_loss = categorical_crossentropy(y_true, y_pred, from_logits=True)
        # smoothing_loss = regularizers.joint_bilateral_smoothing(disp * tf.math.reciprocal_no_nan(mean_disp), img1)
        total_loss = cc_loss

        self._show_semseg(y_true, y_pred)
        self.step_counter += 1

        return total_loss
