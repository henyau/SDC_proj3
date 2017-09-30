# Project 3: Behavioral Cloning
# Keras model based on nVidia paper "End to End Learning for SDC"
# Running on a car simulator, training with only video from cameras and steering wheel, throttle, and brake inputs
# 9/29/17: first go using a simple model, gets halfway around track 1
# 9/30/17: added a generator to create batches of training data
# todo: 
# 1.) Gather more training data. Driving on road backwards, getting back to center of lane and track 2
# 2.)(Done) Need to use generators, running out of memory . also might need to use AWS or googleCC if need to increase parameters
# 3.) Incorporate throttle and brake inputs
# 4.) Controller is just a simple PI controller, should probably modify that

import numpy as np
import cv2 
import os
import csv
import sklearn
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, Dropout, Cropping2D
from sklearn.model_selection import train_test_split

## use a generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    steering_offset = 0.05
    
    while 1: # Loop forever so the generator never terminates
        tf.random_shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            image_list = []
            steering_angle_list = []
            for batch_sample in batch_samples:
                name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                image_list.append(center_image)
                steering_angle_list.append(center_angle)
                
                #include a flipped shot
                image_list.append(np.fliplr(center_image)) 
                steering_angle_list.append(center_angle*-1.0)
                
                #left camera and right camera
                name = '../data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
             
                image_list.append(left_image)
                steering_angle_list.append(center_angle+steering_offset)
                
                name = '../data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
        
                image_list.append(right_image)
                steering_angle_list.append(center_angle-steering_offset)
                
                
            X_train = np.array(image_list)
            y_train = np.array(steering_angle_list)
            yield sklearn.utils.shuffle(X_train, y_train) #return a subset for training
            
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) #skip first line (header)
    for line in reader:
        lines.append(line)
        
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Keras model based on nVidia paper "End to End Learning for SDC"
model = Sequential()
model.add(Lambda(lambda x:(x/255.0)-0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((65,10), (0,0)))) #crop top of images
##model.add(Conv2D(6,(5,5), activation='relu'))
##model.add(Conv2D(6,(5,5), activation='relu'))
##model.add(Conv2D(12,(3,3), activation='relu'))
##model.add(Conv2D(12,(3,3), activation='relu'))
##
##model.add(Flatten())
##model.add(Dense(100))
##model.add(Dense(50))
##model.add(Dense(10))
##model.add(Dense(1)) #just get steering input

model.add(Conv2D(6,(5,5), activation='relu'))
model.add(Conv2D(8,(5,5), activation='relu'))

model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1)) #just get steering input


model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 5)

#use the generator instead
# takes about 50ms to generate each batch,should be running number of unique samples/batch size or thereabouts
# training samples is actually half of actual training set, since does not include the mirrors
# will soon be only 1/4 because also need to include the other two cameras

model.fit_generator(train_generator, samples_per_epoch= (int)(len(train_samples)/16), validation_data=validation_generator,nb_val_samples=(int) (len(validation_samples)/16), epochs=5)

model.save('../models/model.h5')