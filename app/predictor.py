from extract_bottleneck_features import *
from keras.preprocessing import image                  
from tqdm import tqdm
import numpy as np
import keras
import pickle

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def algorithm(img_path):
    model = keras.models.load_model('../models/model')
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    p_vector = model.predict(bottleneck_feature)
    dog_names = []
    with open ('../data/dog_names.txt', 'rb') as fp:
        dog_names = pickle.load(fp)
    return dog_names[np.argmax(p_vector)]








# from extract_bottleneck_features import extract_Resnet50
# from keras.models import Sequential
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.applications.resnet50 import ResNet50
# import numpy as np
# from keras.preprocessing import image                  
# from tqdm import tqdm
# import cv2
# from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
# import keras
# from glob import glob
# import pickle

# def ResNet50_predict_labels(img_path):
#     ResNet50_model = ResNet50(weights='imagenet')
#     # returns prediction vector for image located at img_path
#     img = preprocess_input(path_to_tensor(img_path))
#     return np.argmax(ResNet50_model.predict(img))

# def path_to_tensor(img_path):   
#     # loads RGB image as PIL.Image.Image type
#     img = image.load_img(img_path, target_size=(224, 224))
#     # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
#     x = image.img_to_array(img)
#     # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
#     return np.expand_dims(x, axis=0)

# def dog_detector(img_path):
#     prediction = ResNet50_predict_labels(img_path)
#     return ((prediction <= 268) & (prediction >= 151))

# def face_detector(img_path):
#     face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt.xml')
#     img = cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray)
#     return len(faces) > 0

# def predict_dog_breed(img_path):
#     # load list of dog names
#     dog_names = []
#     with open ('../data/dog_names.txt', 'rb') as fp:
#         dog_names = pickle.load(fp)

#     resnet_model = keras.models.load_model('../models/model')
#     resnet_model.load_weights('../models/weights.best.resnet50.hdf5')

#     # extract bottleneck features
#     bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
#     print(bottleneck_feature.shape)
    
#     # obtain predicted vector
#     p_vector = resnet_model.predict(bottleneck_feature)
    
#     # return predicted dog breed
#     return dog_names[np.argmax(p_vector)]

# def algorithm(img_path):
#     if dog_detector(img_path) == 1:
#         print("Looks like a dog, the breed might be : ")
#         return predict_dog_breed(img_path).partition('.')[-1]
    
#     elif face_detector(img_path) == 1:
#         print("Looks like a human, the spirit animal might be: ")
#         return predict_dog_breed(img_path).partition('.')[-1]
    
#     else:
#         return print("Confused! Please try again")        