#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[histogram]: ./images/histo1.jpg
[samples]: ./images/samples.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 158-168.). I tryied others models also. The Best result was obtained with a nvidia based model network.

The model includes RELU layers to introduce nonlinearity (code line 158-168). 
The data is normalized in the model using a Keras lambda layer (code line 158). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 172-184). 

The model was trained and validated on with udacity data sets and i introduce modifications on the images and dataset to ensure that the model was not overfitting (image flip, random image brightness modification, balance dataset, create recovery images with right and left camera). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was set to 0.00001 manually (model.py line 256).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, balance the dataset to avoid over sampling of just some steering angles, also added left and right camera images and simulate recovering correcting the steering angle.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different models approach. Please see modelA, modelB, and modelC to see the different models i try.

My first step was to use a convolution neural network model similar to the nvidia network that was in the nvidia self driving car pape.  I thought this model might be appropriate because it was already used with success by nvidia. On the first trys, i could not make the car stay on the road. So i tested two simpler models. ModelB have convolutional with relu activations and small dense layers, with the second best behavior. ModelC have convolutional with activation also, but the Dense layers are a little bigger, this one have the worst behavior in my test.

At first i split my data in training and validation set, 80/20. But, After reading in slack group that no validation was need it for predictions, i stoped using validation data. So i only checked the mse on the training set to see how well the training was doing. I this way i can also see if the model was overfitting if the mse error was droping to high from one epoch to the next.

To combat the overfitting, I modified the model adding dropout and also the dataset to upsample a balanced data set that i created. I also cropped the images to remove sky from the top and remove the car from the bottom of the image.

When i run the simulator to see how well the car was driving around track one, there were a few spots where the vehicle fell off the track, like the right curve after the bridge. To improve the driving behavior in these cases, I added more curves samples to the training dataset. After some trial and error, and testing a lot and not be able to complete the track, after a lot of models i try, i found out that in the drive.py i also have to implement some of the same modifications i made in the training dataset. So after i added cropping of the image and the propper resizing, i was able to complete the track 1.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     


lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 30, 30, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 30, 30, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 13, 13, 36)    21636       activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 13, 13, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 5, 48)      43248       activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 5, 48)      0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 3, 64)      27712       activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 3, 3, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 1, 64)      36928       activation_4[0][0]               
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 1, 64)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           16640       flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 256)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 256)           0           activation_6[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           25700       dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           activation_7[0][0]               
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 64)            6464        dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 64)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 64)            0           activation_8[0][0]               
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 16)            1040        dropout_3[0][0]                  
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 16)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             17          activation_9[0][0]               

Total params: 181209
____________________________________________________________________________________________________



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the udacity provided data sample. First, i balance the data using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then i simulated the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive on the center and recover when is going off the track. These images show how the left and right cameras images look like:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and adding random brightness to the images thinking that this would help improve the prediction. The flip because the udacity data the car is driving the other way than the simulator, the brightness is for the different light conditions on the road.  For example, here is an image that has then been flipped and randomized brightness, and also resized to 64x64:

![alt text][image6]


After the collection process, I had aproximately 22000 number of data points.

I randomly shuffled the data set and i used no data into a validation set. 

I used this training data for training the model. The ideal number of epochs was 16 as evidenced by trial and error, and testing in the simulator.

