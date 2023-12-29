from layers.fully_connected import FullyConnected
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.flatten import Flatten
from layers.activation import Elu, Softmax

from utilities.filereader import *
from utilities.model import Model

from loss.losses import CategoricalCrossEntropy
from sklearn.model_selection import train_test_split

import numpy as np
np.random.seed(0)


if __name__ == '__main__':
    
    img_dir="D:/Workspace/pythonProject1/numberplate/numpy-cnn-master/data/car-plate-detection/images/"
    path = 'D:/Workspace/pythonProject1/numberplate/numpy-cnn-master/data/car-plate-detection/annotations'
    text_path='D:/Workspace/pythonProject1/numberplate/numpy-cnn-master/data/car-plate-detection/annotations/'
    X=get_data(img_dir)
    y=get_y(path,text_path)
    
    X = X / 255
    y = y / 255
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.1, random_state=1)
    print("Train data shape: {}, {}".format(train_data.shape, train_labels.shape))
    print("Test data shape: {}, {}".format(test_data.shape, test_labels.shape))

    model = Model(
        Convolution(filters=5, padding='same'),
        Elu(),
        Pooling(mode='max', kernel_shape=(5, 5), stride=2),
        Flatten(),
        FullyConnected(units=10),
        Softmax(),
        name='cnn5'
    )

    model.set_loss(CategoricalCrossEntropy)

    model.train(train_data, train_labels.T, epochs=2) # set load_and_continue to True if you want to start from already trained weights
    # model.load_weights() # uncomment if loading previously trained weights and comment above line to skip training and only load trained weights.

    print('Testing accuracy = {}'.format(model.evaluate(test_data, test_labels)))
