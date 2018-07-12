import cv2
import pandas as pd
import numpy as np
import sys
from sklearn.utils import shuffle

def arguments(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", dest="data_dir", type=str, default="data/driving_log.csv", help="filepath of data directory")
    parser.add_argument("-si", dest="save_image", action="store_true", default=False, help="save processed images")
    parser.add_argument("-sd", dest="save_data", action="store_true", default=False, help="save numeric npy data")
    parser.add_argument("-s", dest="shuffle", action="store_false", default=True, help="shuffle data")
    parser.add_argument("-f", dest="flip", action="store_true", default=False, help="also flip data")
    parser.add_argument("-l", dest="load", action="store_true", default=False, help="load previous data")
    parser.add_argument("-i", dest="input_npy", type=str, default='data/input_npy.npy', help="filepath of input data")
    parser.add_argument("-o", dest="output_npy", type=str, default='data/output_npy.npy', help="filepath of output data")
    args = parser.parse_args(args=args)
    return args

def process_image(img):
    image = cv2.imread(str(img), 1)
    image = image[60:-25, :, :]                            # crop image
    image = cv2.resize(image, (220, 66), cv2.INTER_AREA)   # resize to (220, 66)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)         # convert RGB to YUV
    return image

def save_image_data(data):
    count = 1000
    for i in range(count):
        cv2.imwrite('image_preprocessed/' + str(i) + '.jpg', data[i])

def process(args=None):
    # if args.load:
    #     input_npy = np.load(args.input_npy)
    #     output_npy = np.load(args.output_npy)
    if True:
        input_npy = np.load("data/input_npy.npy")
        output_npy = np.load("data/output_npy.npy")
    else:
        data = pd.read_csv(
            args.data_dir,
            names=['C', 'L', 'R', 'STEERING', 'THROTTLE', 'x', 'x'])

        input_training_data = data[['C', 'L', 'R']].values
        output_training_data = data[['STEERING', 'THROTTLE']].values

        scale = 3
        #if args.flip:
        if True:
            scale *= 2

        batch_size = (data.shape[0] - 1) * scale
        input_npy = np.empty(shape=(batch_size, 66, 220, 3))
        output_npy = np.empty(shape=(batch_size, 2))
        adjustments = {0:0, 1:0.25, 2:-0.25}

        for index in range(batch_size):
            image = process_image(input_training_data[int(index//3)][index%3])
            steering_angle = output_training_data[int(index//3)] + adjustments[index%3]
            input_npy[index] = image
            output_npy[index] = steering_angle

            #if args.flip:
            if True:
                flipped_image = cv2.flip(image, 1)
                input_npy[-(index + 1)] = flipped_image
                output_npy[-(index + 1)] = -steering_angle
                if index == batch_size // 2:
                    break

        #if args.shuffle:
        if True:
            # shuffle data randomly
            input_npy, output_npy = shuffle(input_npy, output_npy, random_state=0)
        #if args.save_data:
        if True:
            # save data for future use
            np.save('data/input_npy', input_npy)
            np.save('data/output_npy', output_npy)

    #if args.save_image:
    if True:
        # save image data for manual checking
        save_image_data(input_npy)

    return input_npy, output_npy

def main(args=None):
    process(arguments(args))

if __name__ == "__main__":
        sys.exit(main())
