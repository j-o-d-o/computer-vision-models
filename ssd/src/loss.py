import tensorflow as tf


class SSDLoss:
  """
  SSD loss, see https://arxiv.org/abs/1512.02325.
  Examples:
      https://github.com/ManishSoni1908/Mobilenet-ssd-keras/blob/master/misc/keras_ssd_loss.py
      https://github.com/tanakataiki/ssd_kerasV2/blob/master/ssd_training.py
  """
  def __init__(self,
               num_boxes: int,
               num_classes: int,
               neg_pos_ratio: int = 3,
               num_neg_min: int = 1,
               alpha: float = 1.0):
    """
    :param num_boxes: number of prior boxes
    :param num_classes: number of classes
    :param neg_pos_ratio: max ratio of negative vs positive boxes
    :param num_neg_min: minimum negative examples (per img!)
    :param alpha: weight of offset loss vs class loss
    """
    self.num_boxes = num_boxes
    self.num_classes = num_classes
    self.neg_pos_ratio = neg_pos_ratio
    self.num_neg_min = num_neg_min
    self.alpha = alpha

  @staticmethod
  def _smooth_l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    L1 loss for offset
    :param y_true: ground truth offset of [cx, cy, width, height] in normalized image coordinates,
                   tensor of size (batch_size, #prior_boxes, 4)
    :param y_pred: same structure of y_true but the predictions from the net
    :return: smooth l1 loss, a 2D tensor of (batch_size, #prior_boxes)
    """
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)

  @staticmethod
  def _softmax_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    softmax log loss for class
    :param y_true: ground truth data with, tensor of shape (batch_size, #prior_boxes, #classes)
    :param y_pred: class predictions from the net, same structure as y_ture
    :return: softmax loss, 2D tensor of (batch_size, #prior_boxes)
    """
    # y_pred_softmax = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    y_pred_softmax = tf.nn.softmax(y_pred)
    y_pred_softmax = tf.maximum(tf.minimum(y_pred_softmax, 1 - 1e-15), 1e-15)
    softmax_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred_softmax), axis=-1)
    return softmax_loss

  @staticmethod
  def _focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.25, gamma: float = 2):
    """
    focal loss for class
    FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,with alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    :param y_true: ground truth data with, tensor of shape (batch_size, #prior_boxes, #classes)
    :param y_pred: class predictions from the net, same structure as y_ture
    :param alpha: hyper parameter alpha
    :param gamma: hyper-parameter gamma
    :return: softmax loss, 2D tensor of (batch_size, #prior_boxes)
    """
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
    log_y_pred = tf.math.log(y_pred)
    focal_scale = tf.multiply(tf.pow(tf.subtract(1.0, y_pred), gamma), alpha)
    focal_loss = tf.multiply(y_true, tf.multiply(focal_scale, log_y_pred))
    focal_loss = -tf.reduce_sum(focal_loss, axis=-1)
    return focal_loss

  def compute_loss(self, y_true, y_pred):
    """
    SSD loss for class and localization loss
    :param y_true: ground truth, tensor of shape (batch_size, #prior_boxes * (#classes + 4))
    :param y_pred: predicted boxes from the net, same structure as y_true
    :return: loss for prediction, tensor with shape (batch_size,)
    """
    # simple loss for testing
    # total_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    # return total_loss # return single value
    batch_size = tf.shape(y_pred)[0]

    y_true = tf.reshape(y_true, (batch_size, self.num_boxes, self.num_classes + 4)) # (batch_size, #boxes, (#classes + 4))
    y_pred = tf.reshape(y_pred, (batch_size, self.num_boxes, self.num_classes + 4)) # (batch_size, #boxes, (#classes + 4))

    loc_loss = self._smooth_l1_loss(y_true[:, :, self.num_classes:],
                                    y_pred[:, :, self.num_classes:]) # (batch_size, #boxes)
    class_loss = self._focal_loss(y_true[:, :, :self.num_classes],
                                  y_pred[:, :, :self.num_classes]) # (batch_size, #boxes)

    neg = y_true[:, :, 0] # (batch_size, #boxes)
    pos = tf.reduce_max(y_true[:, :, 1:self.num_classes], axis=-1) # (batch_size, #boxes)

    num_pos = tf.reduce_sum(pos)

    # get positive loss
    pos_loc_loss = tf.reduce_sum(loc_loss * pos, axis=-1)  # (batch_size,)
    pos_class_loss = tf.reduce_sum(class_loss * pos, axis=-1) # (batch_size,)

    # get negative loss (only for class, we don't penalize loc loss for negative examples)
    neg_class_loss_per_box = class_loss * neg # (batch_size, #boxes)

    num_neg_by_ratio = tf.cast(self.neg_pos_ratio * num_pos, tf.int32)
    num_neg_per_batch = tf.maximum(num_neg_by_ratio, self.num_neg_min * batch_size)  # int
    num_neg_per_batch = tf.minimum(num_neg_per_batch, tf.cast(self.num_boxes * batch_size - tf.cast(num_pos, tf.int32), tf.int32))
    flattened_neg_class_loss = tf.reshape(neg_class_loss_per_box, [-1])
    values, _ = tf.nn.top_k(flattened_neg_class_loss, num_neg_per_batch, sorted=False)
    neg_class_loss = tf.reduce_sum(values, axis=-1) # (batch_size,)

    # class loss
    overall_class_loss = pos_class_loss + neg_class_loss
    overall_class_loss = tf.reduce_sum(overall_class_loss, axis=-1)
    max_num_neg_batch = tf.cast(num_neg_per_batch, tf.float32)
    num_pos = tf.cast(num_pos, tf.float32)
    overall_class_loss /= tf.maximum(1.0, num_pos + max_num_neg_batch)
    # localization loss
    pos_loc_loss = pos_loc_loss
    pos_loc_loss = tf.reduce_sum(pos_loc_loss, axis=-1)
    pos_loc_loss /= tf.maximum(tf.constant(1.0, tf.float32), num_pos)

    total_loss = overall_class_loss + self.alpha * pos_loc_loss
    # print("Class: " + str(overall_class_loss) + " | Loc: " + str(pos_loc_loss))
    return total_loss
