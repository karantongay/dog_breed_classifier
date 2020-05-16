from extract_bottleneck_features import *
from keras.preprocessing import image                  
from tqdm import tqdm
import numpy as np
import pickle
import keras
from keras import backend as K

def path_to_tensor(img_path):
    """
    This function computes the tensor path to the image uploaded by the user.
    Args:
        img_path: Path to the uploaded image by the user

    Returns:
        tensor: Returns 4D tensor
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    tensor = np.expand_dims(x, axis=0)
    return tensor

def algorithm(img_path):
    """
    This function computes the tensor path to the image uploaded by the user.
    Args:
        img_path: Path to the uploaded image by the user

    Returns:
        dog_name: Name of dog breed or spirit animal
    """
    # Load the keras model
    model = keras.models.load_model('../models/weights.best.resnet50.hdf5')
    # Extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # Computer predicted vector
    p_vector = model.predict(bottleneck_feature)
    dog_names = []
    with open ('../data/dog_names.txt', 'rb') as fp:
        dog_names = pickle.load(fp)
    print(dog_names[np.argmax(p_vector)])
    K.clear_session()
    dog_name = dog_names[np.argmax(p_vector)]
    return dog_name

