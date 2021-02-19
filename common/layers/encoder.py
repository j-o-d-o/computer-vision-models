import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, LayerNormalization
from common.layers import bottle_neck_block, upsample_block


def encoder(filters: int, input_tensor: tf.Tensor, filter_scaling=2.0, output_scaled_down: bool = False, namescope: str = "encoder/"):
    x = input_tensor
    fms = []

    x = Conv2D(filters, 5, padding="same", name=f"{namescope}initial_downsample", strides=(2, 2))(x)
    x = BatchNormalization(name=f"{namescope}initial_batchnorm")(x)
    x = ReLU(6., name=f"{namescope}initial_acitvation")(x)
    fms.append(x)
    # x = bottle_neck_block(f"{namescope}feat_extract_m0.5", x, filters, downsample = True)
    # fms.append(x)

    x = bottle_neck_block(f"{namescope}feat_extract_0", x, filters)
    x = bottle_neck_block(f"{namescope}feat_extract_1", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * filter_scaling)

    x = bottle_neck_block(f"{namescope}feat_extract_3", x, filters)
    x = bottle_neck_block(f"{namescope}feat_extract_4", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * filter_scaling)

    x = bottle_neck_block(f"{namescope}feat_extract_6", x, filters)
    x = bottle_neck_block(f"{namescope}feat_extract_7", x, filters, downsample = True)
    filters = int(filters * filter_scaling)

    dilation_rates = [3, 6, 9, 12]
    concat_tensors = []
    x = Conv2D(filters, kernel_size=1, name=f"{namescope}start_dilation_1x1")(x)
    concat_tensors.append(x)
    for i, rate in enumerate(dilation_rates):
        x = bottle_neck_block(f"{namescope}dilation_{i}", x, filters)
        x = BatchNormalization(name=f"{namescope}dilation_batchnorm_{i}")(x)
        concat_tensors.append(x)

    x = Concatenate(name=f"{namescope}dilation_concat")(concat_tensors)
    fms.append(x)

    for i in range(len(fms) - 2, -1, -1):
        filters = int(filters // filter_scaling)
        fms[i] = Conv2D(filters, (3, 3), padding="same", name=f"{namescope}conv2d_up1_{i}")(fms[i])
        fms[i] = BatchNormalization(name=f"{namescope}batchnorm_up1_{i}")(fms[i])
        fms[i] = ReLU()(fms[i])
        x = upsample_block(f"{namescope}up_{i}", x, fms[i], filters)

    if not output_scaled_down:
        filters = int(filters // filter_scaling)
        fm_f = Conv2D(filters, (3, 3), padding="same", name=f"{namescope}conv2d_up1_to_org")(input_tensor)
        fm_f = BatchNormalization(name=f"{namescope}batchnorm_up1_to_org")(fm_f)
        fm_f = ReLU()(fm_f)
        x = upsample_block(f"{namescope}up_to_org", x, fm_f, filters)

    return x, fms