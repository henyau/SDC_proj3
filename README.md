# **Behavioral Cloning** 

## Henry Yau
## Oct 5, 2017


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn_model_structure.jpg "Model Structure"
[image2]: ./images/left_right_cameras.jpg "Three camera setup"



### Model Architecture and Training Strategy

#### 1.  Solution Design Approach/The Keras model

The model used in this project is based on the CNN presented in "End to End Learning for Self-Driving Cars" by Bojarski et. al. though the number of filters for each convolutional layer has been reduced significantly.That model has been tested to good sucess in the real world so it should serve as a more than adequate basis for a simple simulator.

The initial image is 160x320 but during preprocessing is scaled by 0.5 to 80x160 to simplify the model and save memory. It is reasoned that shape information of the road necessary for guidance is still maintained even at a much lower resolution. The input image data is normalized and zero mean centered using a Keras lambda layer. The image is then cropped using a Keras layer to remove the upper 30x160 and lower 10x160 pixels, which correspond to potential above the horizon line distractions and parts of the vehicle itself respectively. 

There are two sets of convolution layers with 5x5 kernels, first with four filters then with six filters. A third convolutional layer with a 3x3 kernel with 12 filters is then applied. The feature map is then flattened and is followed by three fully connected layers. After each layer, ReLu activation layers are used to introduce nonlinearities. The number of neurons in each fully connected layer are gradually reduced to a single output layer which corresponds to the predicted steering wheel angle.

The model structure is illustrated below:

![alt text][image1]

To reduce overfitting, a dropout layer is applied inbetween the fully connected layers after flattening. In addition, the entire dataset is shuffled and split into training and validation sets. 

The amount of training data is too much to fit into memory so a generator is used to produce batches of training and validation data. Keras has built in support for Python generators in the form of fit_generator(). For the number of steps per epoch, Keras documentation recommends using roughly the number of unique training samples divided by the batch size.

An Adam optimizer is used, so there is no need to tune the learning gains.

#### 2. Training process and training data

The training data comes from a set of three forward facing virtual cameras and the recorded steering angle. The vehicle was driven around the track once in each direction trying to remain in the center as much as possible. By driving in both directions, we avoid embedding an inherent left turn bias as the track is generally driven counter-clockwise. A few additional attempts moving from the side of the track to the center were also recorded which were intended to teach the model how to recover. The training images for the left, center, and right cameras are shown below:

![alt text][image2]

These images are from when the vehicle is roughly in the center of the road which is the desired behavior. The center of the road is shown with a green line. This line is in the middle of the center camera but is offset in the left and right cameras. The center of the left camera image is shown with a yellow line and that of the right camera is shown with a red line. Interpreting each image as if it were taken from a vehicle centered at the camera center, we can see how using the left and right images can be used to triple the amount of training data. For example, taking the left image, we see that to get to the true center of the lane shown in green, we must steer right. The amount required to turn can be computed by first training the model using only the center camera images, then feeding the left and right images as inputs to the trained model generating the predicted steering for each side. Subtracting the actual steering input we find the offset required for each side. This offset is approximatedly +0.04 for the left camera and -0.04 for the right camera. So we can triple the set of image-steering pairs from from 7005 to 21015.

Studying the images we can gather what information may be valuable to the model. The cyan lines connect the left and right side extents of the road in the image. For the center camera, we see that the line is practically horizontal, indicating the vehicle is centered. The lines for the left and right cameras are slightly tilted which corresponds to how far from the center the vehicle is. We can also see from these images how information above the horizon may be disruptive to learning and should be cropped. The vehicle itself should also be cropped as it exists in all training images and does not contribute additional information. 

We can also flip each image and multiply the steering input by -1 to double the number image-steering pairs. This results in a set of 42,030 unique image-steering pairs, 80% to be used in training and 20% to be used as validation. This flipping of images also contributes to avoiding the left turn bias.

The training set is used to train the model, while the validation set is used to see if the model is overfitting the data. After 3 epochs the validation error was 0.0037 and after 4 epochs the validation error was 0.0036. Though the error was still decreasing, the learning was stopped at this point because the computational cost outweighed the potential gains. In addition overfitting would likely occur soon after.

