from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

TRAINING_DATA_PATH = '/Users/WenHsu/Documents/Self-Driving car/Term3 Path Planning, Concentrations and System/Project 4/traffic_light_images/training'
TEST_DATA_PATH = '/Users/WenHsu/Documents/Self-Driving car/Term3 Path Planning, Concentrations and System/Project 4/traffic_light_images/test'
SAVED_MODEL_PATH = '/Users/WenHsu/Documents/Self-Driving car/Term3 Path Planning, Concentrations and System/Project 4/model/tl_classifier_Mobile.h5'

# define function to load the train, test datasets
def load_dataset(path):
    data = load_files(path)
    X = np.array(data['filenames'])
    y = np_utils.to_categorical(np.array(data['target']))
    return X, y

# load the train, test dataset
from sklearn.model_selection import train_test_split
X, y = load_dataset(TRAINING_DATA_PATH)
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state =42, stratify = y)
X_test, y_test = load_dataset(TEST_DATA_PATH)

N_classes = y.shape[1]
print('There are %d total images' % X.shape[0])
print('There are %d kinds of signals:' % N_classes)


# transform the input data to tensors
from keras.preprocessing import image
from tqdm import tqdm
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
# to tensors and normalize it
train_tensors = paths_to_tensor(X_train).astype('float32')/255
valid_tensors = paths_to_tensor(X_val).astype('float32')/255
test_tensors = paths_to_tensor(X_test).astype('float32')/255

from keras.layers import Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint  
from keras import applications

model = applications.mobilenet.MobileNet(input_shape = (224,224,3))
last_layer = model.output
predictions = Dense(N_classes, activation='softmax')(last_layer)
model_final = Model(inputs=model.input, outputs=predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = 'rmsprop', metrics=["accuracy"])

epochs = 1
batch_size = 128

checkpointer = ModelCheckpoint(filepath= SAVED_MODEL_PATH, 
                               verbose=1, save_best_only=True)

model_final.fit(train_tensors, y_train, 
         validation_data=(valid_tensors, y_val),
         epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

# load the trained model
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
del model
with CustomObjectScope({'relu6': applications.mobilenet.relu6,'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
    model = load_model(SAVED_MODEL_PATH)

# get index of predicted signal sign for each image in test set
signal_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# print out test accuracy
test_accuracy = 100*np.sum(np.array(signal_predictions)==np.argmax(y_test, axis=1))/len(signal_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)




