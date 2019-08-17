# Convolutional Neural Network - Self Driving Car

## Introduction
With [Udacity's car simulator](https://github.com/udacity/self-driving-car-sim/), machine learning enthusiasts can easily use the embedded image feed from the virtual car to collect data to train a self driving car model. My implementation of this idea incorporates [research publications from Nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) to construct the neural network, Keras as the machine learning framework (with Tensorflow backend), and Python in order to drive a car around a map.

## Results

Here are short gifs that demonstrate the model's ability to maneuver the car around a race track. While there are no automated benchmarks for performance, I ran the network overnight and the car had never crashed.

![alt text](https://github.com/benjaminykim/self-driving-car-simulator/blob/master/media%20assets/1.gif)

![alt text](https://github.com/benjaminykim/self-driving-car-simulator/blob/master/media%20assets/2.gif)

### Shortcomings

1. Since the kind and number of maps within the simulator are limited, the model is not sufficiently robust as training data may not encapsualte edge cases.
2. The car's velocity was not considered in the model construction. Therefore, the I fixed the autonomous vehicles velocity to 10 mph. The disadvantages of this are straightfoward.

## Usage

1. Install Udacity's self driving car simulator [here](https://github.com/udacity/self-driving-car-sim/).
2. Install Tensorflow and Keras
3. Data Generation
	1. Open the simulator
	2. Select "Training Mode"
	3. Hold "r" to record, save output data in /data
	4. Use the data_process.py script to auto-generate data
4. Model Training
	1. Activate tensorflow/keras
	2. Use the model.py script to train the model
5. Autonomous Driving
	1. Open the simulator
	2. Select "Autonomous mode"
	3. Activate tensorflow/keras
	4. Use the drive.py script to drive the car in the simulator

		> python3 drive.py /models/best_model.h5

## Documentation
### data_process.py
tbd
### model.py
tbd
