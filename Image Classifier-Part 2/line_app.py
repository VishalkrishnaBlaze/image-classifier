import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np 

def load_class_names(json_path):
    with open(json_path) as fh:
        flowers = json.load(fh)

    new_flowers = dict()
    for key in flowers.keys():
        new_flowers[str(int(key))] = flowers[key]

    return new_flowers


def load_saved_model(saved_model_path):
    model = tf.keras.models.load_model(saved_model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    return model

def process_image(img):
    processed_image = np.squeeze(img)
    processed_image = tf.image.resize(processed_image, (224, 224))/255.0

    return processed_image