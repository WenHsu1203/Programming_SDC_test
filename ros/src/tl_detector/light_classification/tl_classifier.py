from styx_msgs.msg import TrafficLight
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

class TLClassifier(object):
    def __init__(self):
        # load classifier
        self.model.load_weights('saved_models/weights.test.self_defined.hdf5')

    def img_to_tensor(self, img):
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
        signal = self.model.predict(img_to_tensor(image))
        print(signal)
        return TrafficLight.UNKNOWN
