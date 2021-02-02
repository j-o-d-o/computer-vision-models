import numpy as np
import nvidia_smi


def wrap_angle(angle: float):
    """
    Args:
        angle (float): angle in [rad]

    Returns:
        float: Angle in [rad] that is wrapped to [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle 


def set_up_tf_gpu(tf, limit_memory: bool = False):
    """
    Allows tensorflow to increase memory growth for gpu to a certain point

    Args:
        tf (Tensorflow): Tensorflow import
        limit_memory (bool): If true limits memory to 90% of available gpu memory. Defaults to False
    """
    print(f"Using Tensorflow Version: {tf.__version__}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(gpus[0], True)
    if limit_memory:
        # check available memory
        nvidia_smi.nvmlInit()
        # TODO: hardcoded device index for one gpu, change this for multiple gpu setups
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        nvidia_smi.nvmlShutdown()
        free_gpu_memory_mb = int(info.free * 0.9 * 1e-6)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=free_gpu_memory_mb)])
        print(f"Using max GPU Memory: {free_gpu_memory_mb}")
