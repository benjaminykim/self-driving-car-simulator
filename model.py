""" import libraries """
import pandas as pd
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.image as matplot
from sklearn.utils import shuffle
from data_process import process

# shape of input data
SHAPE = (66, 220, 3)


def model_architecture():
    """ model_architecture: creates model architecture
            1 : Lambda layer for preprocessing
            5 : Convolutional 2D Layers
            1 : Dropout Layer (0.5 dropout)
            1 : Flatten Layer
            4 : Dense Layers

        model architecture loosely based off of Nvidia's research paper:
            https://arxiv.org/pdf/1604.07316.pdf

        return: model (keras model)
    """

    model = Sequential()

    # IMAGE NORMALIZATION LAYER
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=SHAPE))

    # CONVOLUTIONAL LAYERS
    model.add(Conv2D(
            filters=24,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation='elu'))
    model.add(Conv2D(
            filters=36,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation='elu'))
    model.add(Conv2D(
            filters=48,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation='elu'))
    model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation='elu'))
    model.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation='elu'))

    # DROPTOUT LAYER
    model.add(Dropout(0.5))

    # FULLY CONNECTED LAYERS
    model.add(Flatten())                        # 1280 neurons
    model.add(Dense(100, activation='elu'))     # 100 neurons
    model.add(Dense(50, activation='elu'))      # 50 neurons
    model.add(Dense(10, activation='elu'))      # 10 neurons

    # OUTPUT LAYER
    model.add(Dense(2))                         # output steering angle, throttle

    # Inspect Model
    model.summary()
    return model


def data_preprocessing(
            load=False,
            save_image=False,
            save_data=True,
            clr=False,
            flip=False,
            shuffle_data=False,
            augment=None):
    """ data_preprocessing: preprocess data for training

        parameters:
            load: true to load previous data (bool)
            save_image: true to save input images (200 default) (bool)
            save_data: true to save current data (bool)
            clr: true to use all Center, Left, and Right images (bool)
            flip: true to include flipped images with steering angles (bool)
            shuffle_data: true to randomly shuffle all data (bool)
            augment: methodology of image manipulation for extra data (string)

        return: tuple of input_data and output_data (np array, np array)
    """

    if load:
        input_list = np.load('input_list.npy')
        output_list = np.load('output_list.npy')
    else:
        # filepath specific to Benjamin Kim's Macbook Pro
        data = pd.read_csv(
            ('/USERS/kernel-ares/Desktop/research/self_driving/data/driving_log.csv'),
            names=['C', 'L', 'R', 'STEERING', 'x', 'x', 'x'])

        scale = 1
        if clr:
            scale = 3
            input_data = data[['C', 'L', 'R']].values
        else:
            input_data = data[['C']].values
        if flip:
            scale *= 2

        batch_size = data.shape[0] * scale
        output_data = data[['STEERING']].values
        input_list = np.empty(shape=(batch_size, 66, 220, 3))
        output_list = np.empty(shape=(batch_size, 1))

        count = 0

        for training_instance in input_data:
            displacement = 0
            for image in training_instance:
                # IMAGE preprocessing
                # import image from path
                current_image = cv2.imread(str(image), 1)
                # crop image
                current_image = current_image[60:-25, :, :]
                # resize image to (220, 66)
                current_image = cv2.resize(current_image, (220, 66), cv2.INTER_AREA)
                # convert image from RGB to YUV
                current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2YUV)

                # 0 - C, 1 - L, 2 - R
                # configuration for CENTER, LEFT, RIGHT for steering adjustment
                if clr and displacement != 0:
                    # left camera view
                    if displacement == 1:
                        steering_adjustment = 0.1
                    # right camera view
                    elif displacement ==2:
                        steering_adjustment = -0.1
                else:
                    # center camera view
                    steering_adjustment = 0

                # create true steering angle
                steering_angle = output_data[int(count//scale)] + steering_adjustment

                # add normal image to input_list
                input_list[count] = current_image
                output_list[count] = steering_angle

                # add flipped image to input_list
                if flip:
                    flipped_image = cv2.flip(current_image, 1)
                    input_list[count + 1] = flipped_image
                    output_list[count + 1] = -steering_angle
                    count += 2
                else:
                    count += 1

                # adjust indices
                if clr:
                    displacement += 1
                    displacement %= 3

        if shuffle_data:
            # shuffle data randomly
            input_list, output_list = shuffle(input_list, output_list, random_state=0)
        if save_data:
            # save data for quicker reloading
            np.save('input_list', input_list)
            np.save('output_list', output_list)

    if save_image:
        # save image data for manual checking
        save_image_data(input_list, 200)

    return input_list, output_list


def save_image_data(data, count=1000):
    """
        save_image_data: save image data to system

        parameters:
            data: images to save to system (np array)
            count: number of images to save (int)

        return: none
    """
    for i in range(count):
        cv2.imwrite('image_preprocessed/' + str(i) + '.jpg', data[i])


def train(data, model, validation_percentage=0.5):
    """
        train: train the model

        parameters:
            data: input and output data (np array, np array)
            model: model to be fitted (keras model)
            validation_percentage: percent of data to be used for validation (float)

        return: none
    """

    # create checkpoint to save model after ever epoch
    checkpoint = ModelCheckpoint(
            'model-{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            mode='auto')

    # compile model
    # loss function:    mean_squared_error
    # optimizer:        Adam
    # learning rate:    0.0001
    model.compile(
            loss='mean_squared_error',
            optimizer=optimizers.Adam(lr=.0001))

    # fit model
    model.fit(
            x=data[0],
            y=data[1],
            epochs=10,
            initial_epoch=0,
            batch_size=32,
            validation_split=validation_percentage,
            shuffle=False,
            callbacks=[checkpoint],
            verbose=1)


if __name__ == "__main__":
    """ handler function
        creates a keras model that is fitted to the input data
    """

    # preprocess data
    # data = data_preprocessing(
    #         load=True,
    #         save_image=False,
    #         save_data=True,
    #         clr=True,
    #         flip=True,
    #         shuffle_data=True,
    #         augment='translate')

    data = process()

    # instantiate model architecture
    model = model_architecture()

    # train model
    train(
            data=data,
            model=model,
            validation_percentage=0.30)

    # save model for implementation/prediction
    model.save("model.h5")
