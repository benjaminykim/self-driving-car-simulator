# Convolutional Neural Network - Self Driving Car

## Introduction
With [Udacity's car simulator](https://github.com/udacity/self-driving-car-sim/), machine learning enthusiasts can easily use the embedded image feed from the virtual car to collect data to train a self driving car model. My implementation of this idea incorporates [research publications from Nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) to construct the neural network, Keras as the machine learning framework (with Tensorflow backend), and Python in order to drive a car around a map.

I trained a network with 280,000 parameters. Specifics on implementation can be found from inline comments or in the Nvidia link provided. My final model is stored in `models/best_model.h5`.

## Results

Here are short gifs that demonstrate the model's ability to maneuver the car around a race track. While there are no automated benchmarks for performance, I ran the network overnight and the car had never crashed.

![self driving car gif 1](https://github.com/benjaminykim/self-driving-car-simulator/blob/master/media%20assets/1.gif)

![self driving car gif 2](https://github.com/benjaminykim/self-driving-car-simulator/blob/master/media%20assets/2.gif)

## Shortcomings
1. Since the kind and number of maps within the simulator are limited, the model is not sufficiently robust as training data may not encapsulate edge cases.
2. The car's velocity was not considered in the model construction. Therefore, the I fixed the autonomous vehicles velocity to 10 mph. The disadvantages are straightforward.

## Usage
1. Install Udacity's self driving car simulator [here](https://github.com/udacity/self-driving-car-sim/).
2. Install Tensorflow and Keras
3. Data Generation
	1. Open the simulator
	2. Select "Training Mode"
	3. Hold "r" to record, save output data in /data
	4. Use the `data_process.py` script to auto-generate data
4. Model Training
	1. Activate tensorflow/keras
	2. Use the `model.py` script to train the model
5. Autonomous Driving
	1. Open the simulator
	2. Select "Autonomous mode"
	3. Activate tensorflow/keras
	4. Use the `drive.py` script to drive the car in the simulator

## Documentation
### `data_process.py`
This script is used to convert our image data to numpy arrays so that the Keras model can use the data properly. It also handles all data processing and generation.

Since the virtual car uses three digital cameras (left, center, and right), we can incorporate the left and right cameras in our model training by augmenting the steering wheel value by 0.25 and -0.25 respectively.

Additionally, we may reflect the image about the y-axis and multiply the steering angle by -1 in order to train our model on reflections. Since the default track is a circle that almost entirely uses left turns, this is valuable in the construction of our model in order to increase its robustness.

Some command-line arguments that are helpful in prototyping data augmentation or data processing configurations:
1. `-si`: saves processed images so that researchers may see what the final input to the model looks like
2. `-s`: shuffles the data in the dataset to create robustness in our training.
3. `-f`: reflects the data about the y axis

Some notes for data pre_processing:
1. Conversion of RGB to YUV is mandatory for more efficient color coding so that the CNN can easily differentiate gradients in the image.
2. Image cropping decreases the complexity of the model needed and training time. Items in the image such as the sky or sides of the road are not as important as the road itself for training a self-driving car in this simulated environment.

I ultimately used `python3 data_process.py -s -f` to create my dataset.
### `model.py`
This script trains the convolutional neural network model. It is designed by Nvidia and the actual research paper is found [here](https://arxiv.org/pdf/1604.07316.pdf).
###### Model Architecture:

	1 : Lambda layer for preprocessing
	5 : Convolutional 2D Layers
	1 : Dropout Layer (0.5 dropout)
	1 : Flatten Layer
	4 : Dense Layers

The model uses mean squared error for its loss function, the Adam optimizer, and a 0.0001 learning rate.

Run `python3 model.py` to create and train your model. The trained model is saved as a `.h5` file.
### `drive.py`
This script is provided by Udacity in order to operate the virtual vehicle. To run the model with parameters that I trained, use `python3 drive.py models/best_model.h5`
