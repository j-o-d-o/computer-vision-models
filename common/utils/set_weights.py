import tensorflow as tf


def set_weights(base_model_path, new_model):
    # Store names of base model layers of model in dict
    base_model = tf.keras.models.load_model(base_model_path, compile=False)
    base_layer_dict = dict([(layer.name, layer) for layer in base_model.layers]) 
    # Loop through actual model and see if names are matching, set weights in case they are
    for layer in new_model.layers:
        if layer.name in base_layer_dict:
            print(f"Setting weights for {layer.name}")
            try:
                weights = base_layer_dict[layer.name].get_weights()
                layer.set_weights(weights)
            except ValueError as e:
                print(f"ValueError: {e}")
        else:
            print(f"Not found: {layer.name}")
    return new_model
