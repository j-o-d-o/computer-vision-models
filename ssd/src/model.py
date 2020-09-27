from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, Add, Flatten, \
  concatenate
from tensorflow.keras.models import Model
from ssd.src.params import Params


def _bottle_neck_block(input_layer, filter_size: int, expansion_factor: int = 6, stride=1):
  filter_size = max(2, int(filter_size * Params.ALPHA))
  x = input_layer

  # Expansion
  x = Conv2D(filter_size * expansion_factor, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
  x = BatchNormalization()(x)
  x = ReLU(6.)(x)

  # Convolution
  x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False,
                      padding='same' if stride == 1 else 'same')(x)
  x = BatchNormalization()(x)
  x = ReLU(6.)(x)

  # Project
  x = Conv2D(filter_size, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
  x = BatchNormalization()(x)

  # Residual connection
  input_filters = int(input_layer.shape[-1])
  if stride == 1 and input_filters == filter_size:
    x = Add()([input_layer, x])

  return x


def _fm_extractor(input_layer):
  # find ratio count for feature map
  nb_ratios = None
  input_shape = input_layer.shape[1:3]
  for fm in Params.FEATURE_MAPS:
    if fm.size == input_shape:
      nb_ratios = len(fm.ratios)
      break

  if nb_ratios is None:
    raise ValueError("Feature Map with Size: " + str(input_shape) + " not found!")

  output = Conv2D((len(Params.CLASSES) + 4) * nb_ratios, (3, 3), padding='same')(input_layer)
  output = Flatten()(output)

  return output


def create_model():
  # 128 x 512
  inputs = Input(shape=(Params.IMG_HEIGHT, Params.IMG_WIDTH, Params.IMG_CHANNELS))
  x = Conv2D(max(2, int(32 * Params.ALPHA)), 5, padding="same", strides=(2, 2))(inputs) # / 2
  x = BatchNormalization()(x)
  x = ReLU(6.)(x)

  x = _bottle_neck_block(x, 32, expansion_factor=1, stride=1)

  x = _bottle_neck_block(x, 32, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 32, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 32, expansion_factor=6, stride=2) # / 2

  x = _bottle_neck_block(x, 64, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 64, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 64, expansion_factor=6, stride=2) # / 2

  x = _bottle_neck_block(x, 98, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 98, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 98, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 98, expansion_factor=6, stride=2) # / 2

  # (8, 32)
  output_8_32 = _fm_extractor(x)

  x = _bottle_neck_block(x, 128, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 128, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 128, expansion_factor=6, stride=1)

  x = _bottle_neck_block(x, 160, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 160, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 160, expansion_factor=6, stride=2) # / 2

  # (4, 16)
  output_4_16 = _fm_extractor(x)

  x = _bottle_neck_block(x, 320, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 320, expansion_factor=6, stride=1)
  x = _bottle_neck_block(x, 320, expansion_factor=6, stride=2) # / 2
  x = Conv2D(512, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
  x = BatchNormalization()(x)
  x = ReLU(6.)(x)

  # (2, 8)
  output_2_8 = _fm_extractor(x)

  output = concatenate([
    output_8_32,
    output_4_16,
    output_2_8
  ])

  model = Model(inputs, output)
  return model
