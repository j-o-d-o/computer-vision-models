import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate
from common.layers import bottle_neck_block, upsample_block


def encoder(filters: int, input_tensor: tf.Tensor):
    x = input_tensor
    fms = []

    x = Conv2D(filters, 5, padding="same", name="initial_downsample", strides=(2, 2))(x) # / 2
    x = BatchNormalization(name="initial_batchnorm")(x)
    x = ReLU(6., name="initial_acitvation")(x)
    fms.append(x)

    x = bottle_neck_block(f"feat_extract_0", x, filters)
    x = bottle_neck_block(f"feat_extract_1", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * 2)

    x = bottle_neck_block(f"feat_extract_2", x, filters)
    x = bottle_neck_block(f"feat_extract_2.5", x, filters)
    x = bottle_neck_block(f"feat_extract_3", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * 2)

    x = bottle_neck_block(f"feat_extract_4", x, filters)
    x = bottle_neck_block(f"feat_extract_4.5", x, filters)
    x = bottle_neck_block(f"feat_extract_5", x, filters, downsample = True)
    filters = int(filters * 2)

    dilation_rates = [3, 5, 7, 9, 11]
    concat_tensors = []
    x = Conv2D(filters, kernel_size=1, name="start_dilation_1x1")(x)
    concat_tensors.append(x)
    for i, rate in enumerate(dilation_rates):
        x = bottle_neck_block(f"dilation_{i}", x, filters)
        x = BatchNormalization(name=f"dilation_batchnorm_{i}")(x)
        concat_tensors.append(x)

    x = Concatenate(name="dilation_concat")(concat_tensors)
    fms.append(x)

    for i in range(len(fms) - 2, -1, -1):
        filters = int(filters // 2)
        fms[i] = Conv2D(filters, (3, 3), padding="same", name=f"conv2d_up1_{i}")(fms[i])
        fms[i] = BatchNormalization(name=f"batchnorm_up1_{i}")(fms[i])
        fms[i] = ReLU()(fms[i])
        fms[i] = Conv2D(filters, (3, 3), padding="same", name=f"conv2d_up2_{i}")(fms[i])
        fms[i] = BatchNormalization(name=f"batchnorm_up2_{i}")(fms[i])
        fms[i] = ReLU()(fms[i])
        x = upsample_block(f"up_{i}", x, fms[i], filters)

    return x