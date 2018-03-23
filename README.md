# Basic-Face-Classifier-KERAS

# ====== Description ======

These files let you create dataset, train and test your face classifier. It uses haarcascade for face detection and uses it for creating the dataset. Later it uses the haarcascade for detection and then classifier predicts the face. The name of classes is saved in a .yaml file.

The Network Model consists of two convolution NN layers followed by two Fully-connected layers. The network can be increased or pretrained networks can also be used instead.

Options:
1. Create Dataset
	> The dataset is stored in datasets/ folder
2. Classify (Won't work unless trained)
	> The Result of Face Classification is displayed on the LIVE Video Feed.
3. Train
	> If there are no existing weights, the network will be trained from scratch, otherwise the network training will continue using previous weights.

# ====== Dependencies ======
Python 3.x or 2.x

NumPy

OpenCV

tensorflow

Keras

fnmatch

yaml

termcolor

# ====== Steps ======
Download/Clone this repository in any folder and run.

Notes: 
1.You might need to change the value of 'vino' in case your webcam id is different.
2.You have to change the folder name in face_cascade. Locate the file where Python is installed