"""
Project 3: Behavioral Cloning
Keras model based on nVidia paper "End to End Learning for SDC"

Created on Mon Oct  2 01:45:22 2017
@author: Henry Yau

"""

import numpy as np
import cv2 
import csv
import sklearn
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Lambda, Flatten, Dense, Conv2D, Dropout, Cropping2D
from sklearn.model_selection import train_test_split

use_last_model_b = False

def generator(samples, batch_size=32):
    """creates a Python generator which produces batches of input data"""
    num_samples = len(samples)
    steering_offset = 0.04
    
    while 1: # Loop forever so the generator never terminates
        tf.random_shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            image_list = []
            steering_angle_list = []
            for batch_sample in batch_samples:
                stripped_name = batch_sample[0].split('\\')[-1] # try to strip windows dir path
                stripped_name = stripped_name.split('/')[-1] # try to strip linux dir path
                name = '../data/IMG/'+ stripped_name
                center_image = cv2.imread(name)
                
                # resize the image by 0.5                
                center_image = cv2.resize(center_image,(160, 80), interpolation = cv2.INTER_CUBIC)     
                center_angle = float(batch_sample[3])                
                
                image_list.append(center_image)
                steering_angle_list.append(center_angle)
                # include a flipped shot
                image_list.append(np.fliplr(center_image)) 
                steering_angle_list.append(center_angle*-1.0)
                
                # left camera
                stripped_name = batch_sample[1].split('\\')[-1] # try to strip windows dir path
                stripped_name = stripped_name.split('/')[-1] # try to strip linux dir path
                left_image = cv2.imread(name)
                left_image = cv2.resize(left_image,(160, 80), interpolation = cv2.INTER_CUBIC)
             
                image_list.append(left_image)
                steering_angle_list.append(center_angle+steering_offset)
          
                # flipped image
                image_list.append(np.fliplr(left_image))
                steering_angle_list.append((center_angle+steering_offset)*-1.0)

                # right camera
                stripped_name = batch_sample[2].split('\\')[-1] # try to strip windows dir path
                stripped_name = stripped_name.split('/')[-1] # try to strip linux dir path
                right_image = cv2.imread(name)
                right_image = cv2.resize(right_image,(160, 80), interpolation = cv2.INTER_CUBIC)
   
                image_list.append(right_image)
                steering_angle_list.append(center_angle-steering_offset)
        
                # flipped image
                image_list.append(np.fliplr(right_image))
                steering_angle_list.append((center_angle-steering_offset)*-1.0)

                
            X_train = np.array(image_list)
            y_train = np.array(steering_angle_list)
            yield sklearn.utils.shuffle(X_train, y_train) # return a subset for training
            
lines = []
with open('../data/driving_log_self.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip first line (header)
    for line in reader:
        lines.append(line)
        
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


if use_last_model_b != True:
    # Keras model based on nVidia paper "End to End Learning for SDC"
    model = Sequential()    
    model.add(Lambda(lambda x:(x/255.0)-0.5, input_shape = (80,160,3)))
    model.add(Cropping2D(cropping=((30,10), (0,0)))) # crop top of images, do cropping ops in the generator

    model.add(Conv2D(4,(5,5), activation='relu'))
    model.add(Conv2D(6,(5,5), activation='relu'))
    model.add(Conv2D(12,(3,3), activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1)) # just get steering input

else:
    model = load_model('../models/model.h5')
    
model.compile(loss = 'mse', optimizer = 'adam')

# use the generator instead
# should be running number of unique samples/batch size or thereabouts
batch_scale = batch_size/6 # for each line in csv we generate 6 samples, center,left, right and the flipped versions
model.fit_generator(train_generator, steps_per_epoch= (int)(len(train_samples)/batch_scale),\
                    validation_data=validation_generator,\
                    validation_steps=(int) (len(validation_samples)/batch_scale), epochs=4)

model.save('../models/model.h5')
