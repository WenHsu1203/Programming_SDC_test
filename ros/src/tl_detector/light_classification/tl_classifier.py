from styx_msgs.msg import TrafficLight
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from resizeimage import resizeimage
class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model = Sequential()
        # 224,224,16
        self.model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (224, 224, 3)))
        # 112,112,16
        self.model.add(MaxPooling2D(pool_size = 2))
        # Dropout
        self.model.add(Dropout(0.5))
        # 112,112,32
        self.model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
        # 56,56,32
        self.model.add(MaxPooling2D(pool_size = 2))
        # Dropout
        self.model.add(Dropout(0.5))
        # 56,56,64
        self.model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
        # 28,28,64
        self.model.add(MaxPooling2D(pool_size = 2))
        # Dropout
        self.model.add(Dropout(0.5))
        # 28,28,128
        self.model.add(Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
        # 14,14,128
        self.model.add(MaxPooling2D(pool_size = 2))
        # Dropout
        self.model.add(Dropout(0.5))
        # 14,14,256
        self.model.add(Conv2D(filters = 256, kernel_size = 2, padding = 'same', activation = 'relu'))
        # 7,7,256
        self.model.add(MaxPooling2D(pool_size = 2))
        # Dropout
        self.model.add(Dropout(0.5))
        # flatten
        self.model.add(Flatten())
        self.model.add(Dense(5000, activation = 'relu'))
        # Dropout
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1000, activation = 'relu'))
        # Dropout
        self.model.add(Dropout(0.5))
        self.model.add(Dense(500, activation = 'relu'))
        # Dropout
        self.model.add(Dropout(0.5))
        # Fully connected Layer to the number of signal categories
        self.model.add(Dense(4, activation = 'softmax'))

        self.model.load_weights('saved_models/weights.best.self_defined.hdf5')

    def img_to_tensor(self, img):
        img = resizeimage.resize_cover(img, [224, 224])
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

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
