import tensorflow as tf
import numpy as np


def set_weights(base_model_path, new_model, custom_objects = {}):
    # Store names of base model layers of model in dict
    base_model = tf.keras.models.load_model(base_model_path, custom_objects=custom_objects, compile=False)
    base_layer_dict = dict([(layer.name, layer) for layer in base_model.layers]) 
    # Loop through actual model and see if names are matching, set weights in case they are
    for layer in new_model.layers:
        if layer.name in base_layer_dict:
            print(f"Setting weights for {layer.name}")
            try:
                weights = base_layer_dict[layer.name].get_weights()
                goal_size = layer.get_weights()
                for i in range(len(goal_size)):
                    if goal_size[i].shape != weights[i].shape:
                        print(f"Need to resize from {weights[i].shape} to {goal_size[i].shape}")
                        weights[i] = np.resize(weights[i], goal_size[i].shape)
                layer.set_weights(weights)
            except ValueError as e:
                print(f"ValueError: {e}")
        else:
            print(f"Not found: {layer.name}")
    return new_model
