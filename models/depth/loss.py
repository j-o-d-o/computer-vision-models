import tensorflow as tf
from tensorflow.keras.losses import Loss, categorical_crossentropy


class DepthLoss(Loss):
    def call(self, y_true, y_pred):
        total_loss = categorical_crossentropy(y_true, y_pred, from_logits=True)
        total_loss = total_loss
        return total_loss
