import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v3
from tensorflow.keras.layers import Dense,GlobalMaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib

tf.test.is_built_with_cuda()
print("Num GPUs Available: ", len(device_lib.list_local_devices()))

IMG_SIZE=224

def split_train_test(image_folder_path:str,class_labels:array):
    ds_train=tf.keras.preprocessing.image_dataset_from_directory(image_folder_path,
                                                             labels='inferred',label_mode="int",
                                                             class_names=class_labels,color_mode='rgb',
                                                             image_size=(IMG_SIZE,IMG_SIZE), #reshapeauto
                                                             shuffle=True,seed=123,validation_split=0.4,subset="training")
    ds_validate=tf.keras.preprocessing.image_dataset_from_directory("D:\\ECG DB\\FourClasseswith 20percent",
                                                                labels='inferred',label_mode="int",
                                                                class_names=['AF','NSR','PAC','PVC'],color_mode='rgb',
                                                                image_size=(IMG_SIZE,IMG_SIZE), #reshapeauto
                                                                shuffle=True,seed=123,validation_split=0.4,subset="validation")