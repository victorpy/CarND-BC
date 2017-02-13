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

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json model
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
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

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

