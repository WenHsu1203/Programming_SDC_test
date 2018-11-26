from styx_msgs.msg import TrafficLight
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json
from styx_msgs.msg import TrafficLight
from keras.models import load_model
from keras.models import Model
from keras import applications
# load the trained model
from keras.utils.generic_utils import CustomObjectScope
import rospy

model_filepath = 'saved_models/model.MobileNet-3-classes.h5'
n_classes = 3

class TLClassifier(object):
    def __init__(self):
        # load keras libraies and load the MobileNet model
        with CustomObjectScope({'relu6': applications.mobilenet.relu6,'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
            self.model = load_model(model_filepath)
            self.model._make_predict_function() # Otherwise there is a "Tensor %s is not an element of this grap..." when predicting
        rospy.loginfo("TLClassifier: Model loaded - READY")
        # load the model
        # with open('model_architecture.json', 'r') as f:
        #     self.model = model_from_json(f.read())
        # self.model.load_weights('saved_models/weights.best.self_defined.h5')

        # rospy.loginfo("TLClassifier: Model loaded - READY")

    def img_to_tensor(self, img):
        # resize the image to (224, 224) to input into the model
        img = cv2.resize(img,(224,224))
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(img, axis=0)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light  

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        # signal should be a vector of size four [green, red, unknown, yellow]. One of the value is 1 others are 0
        # ex. [0 1 0 0], which indicates it's a red light
        # signal = self.model.predict(img_to_tensor(image).astype('float32')/255)
        # if (signal[0] == 1):
        #     return TrafficLight.GREEN
        # elif (signal[1] == 1):
        #     return TrafficLight.RED
        # elif (signal[3] == 1):
        #     return TrafficLight.YELLOW

        # return TrafficLight.UNKNOWN
        image = cv2.resize(image,(224,224))
        
        # to tensors and normalize it
        x = img_preprocessing.img_to_array(image)
        x = np.expand_dims(x, axis=0).astype('float32')/255
        
        # get index of predicted signal sign for the image
        signal_prediction = np.argmax(self.model.predict(x))

        return signal_prediction
