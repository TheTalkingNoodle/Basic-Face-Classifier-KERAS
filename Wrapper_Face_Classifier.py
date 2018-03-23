import os, sys
import numpy as np
from termcolor import colored
import cv2, fnmatch
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout, Input, merge, Lambda, Activation, Add
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from keras.optimizers import RMSprop, Adam, Adagrad, sgd
from keras.models import Sequential, Model, load_model
from keras.engine import Layer
from keras.regularizers import l2
from keras.utils import layer_utils
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
import yaml

vino=0

# ============================= GENERATE DATASETS =============================#
def gen_photos(dir, name):

    # Initialize Webcam
    cap = cv2.VideoCapture(vino)
    cap.set(3, 1920)
    cap.set(4, 1080)

    # Folder name for the dataset
    class_fold=dir+"datasets\\"+name

    # Get number of exisiting files
    posfiles = fnmatch.filter(os.listdir(class_fold), '*.jpg')
    posfiles = [os.path.splitext(each)[0] for each in posfiles]
    posfiles = np.asarray(posfiles)

    for j in range(0, posfiles.shape[0]):
        posfiles[j] = posfiles[j][-1:]

    posfiles = np.asarray(posfiles, np.int)

    if posfiles.shape == (0,):
        n = 1
    else:
        n = np.amax(posfiles) + 1

    # While Loop for dataset generation
    print ("Press 'c' to store files.")
    while (1):
        # Get Image
        ret, img = cap.read()
        roi=img.copy()

        # Convert To Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Setup the cascade classifier for frontal face detection
        face_cascade = cv2.CascadeClassifier('C:\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

        # Get faces from the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        key = cv2.waitKey(1) & 0xFF

        # Process for every face in the image
        for (x, y, w, h) in faces:
            top_left=(x,y)
            bottom_right=(x+w,y+h)

            roi = cv2.rectangle(roi, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (255,255,255), 2)
            roi = cv2.resize(roi,(640,480))
            cv2.imshow("ROI",roi)

        # press 'c' to store files
            if key == ord("c"):
                crop_img = img[y:y+h,x:x+w]
                cv2.imwrite(class_fold + "\\" + name + "_" + str(n) + ".jpg", crop_img)
                cv2.putText(crop_img, str(n), (10, int(crop_img.shape[0] * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
                print (class_fold + "\\" + name + "_" + str(n) + ".jpg")
                cv2.imshow("Saved Image", crop_img)

                n += 1

        # Image Display
        img = cv2.resize(img,(640,480))
        cv2.imshow('LIVE', img)


        if key == ord("q"):
            break

    for i in range(1, 50):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    cap.release()

# ============================= DATA PROCESSING =============================#
def load_data(w,h):

    # Folder where dataset is stored
    data_fold = curr_dir + "datasets/"
    classes = os.listdir(data_fold)
    load_data.classes=len(classes)
    classno=0
    x=[]
    y=[]
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]

    # Load files from the dataset folders and write their names in the .yaml file
    for i in classes:
        with open(curr_dir + "data.yaml", 'r') as stream:
            data_loaded = yaml.load(stream)
        if data_loaded:
            if classno not in data_loaded:
                    appenddata = {classno:
                        {
                            'Name': i
                        }
                    }
                    data_loaded.update(appenddata)
                    with open(curr_dir + 'data.yaml', 'w') as outfile:
                        yaml.dump(data_loaded, outfile, default_flow_style=False)
                    classno+=1
        else:
                appenddata = {classno:
                    {
                        'Name': i
                    }
                }
                with open(curr_dir + 'data.yaml', 'w') as outfile:
                    yaml.dump(appenddata, outfile, default_flow_style=False)
                classno+=1

    classno=0
    for i in classes:
        class_fold= curr_dir + "datasets/"+i+"/"
        images = os.listdir(class_fold)
        k=0
        #np.random.shuffle(images)
        for j in images:
            image=cv2.imread(class_fold+j)
            image=cv2.resize(image,(w,h))
            data=np.reshape(image,(w,h,3))
            x.append(data)
            y.append(classno)
            # Store every 5th Image in Test Set, Otherwise store in training set
            if k%5==0:
                x_test.append(data)
                y_test.append(classno)
            else:
                x_train.append(data)
                y_train.append(classno)
            k+=1
        classno+=1

    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    y_test=np.array(y_test)

    print ("Data Shapes (X_train, y_train, X_test, y_test)")
    print (x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return x_train,y_train,x_test,y_test

# ============================= TRAIN =============================#
def model_train(w,h):

    # the data split between train and test sets
    X_train, Y_train, X_test, Y_test = load_data(w,h)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # ==================== Convert 1-dimensional class arrays to 2-dimensional class matrices
    Y_train = np_utils.to_categorical(Y_train, load_data.classes)
    Y_test = np_utils.to_categorical(Y_test, load_data.classes)

    # ==================== Define Model
    model = Sequential()

    # 1st Convolution Layer
    model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(w, h, 3)))

    # 2nd Convolution Layer
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    # Flatten data to supply to FC Layer
    model.add(Flatten())

    # 1st FC Layer (Fully Connected)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # 2nd FC Layer
    model.add(Dense(load_data.classes, activation='softmax'))

    # ==================== Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # ==================== Fit model
    # Store weights only if validation loss has improved
    checkpoint_1 = ModelCheckpoint(curr_dir+"Weights/face_classifier_model.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    callbacks = [checkpoint_1]

    if not os.path.exists(curr_dir+"Weights/face_classifier_model.h5"):
        model.fit(X_train, Y_train,validation_data=(X_test, Y_test),
                  batch_size=10, epochs=50, shuffle=True, verbose=1, callbacks=callbacks)
    else:
        model.load_weights(curr_dir+"Weights/face_classifier_model.h5")
        model.fit(X_train, Y_train,validation_data=(X_test, Y_test),
                  batch_size=10, epochs=10, shuffle=True, verbose=1, callbacks=callbacks)

    # ==================== Train Accuracy
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("\nTEST ACCURACY (showing first 5 predictions):\n%s: %.2f%%\n" % (
    model.metrics_names[1], score[1] * 100))

def classify(w_model,h_model):

    with open(curr_dir + "data.yaml", 'r') as stream:
        data_loaded = yaml.load(stream)

    cap = cv2.VideoCapture(vino)
    cap.set(3, 1920)
    cap.set(4, 1080)

    while (1):
        ret, img = cap.read()
        roi=img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('C:\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )


        key = cv2.waitKey(1) & 0xFF

        for (x, y, w, h) in faces:
            top_left=(x,y)
            bottom_right=(x+w,y+h)
            roi = cv2.rectangle(roi, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (255,255,255), 2)
            roi = cv2.resize(roi,(640,480))


            crop_img = img[y:y+h,x:x+w]
            data=cv2.resize(crop_img,(w_model,h_model))
            data=np.reshape(data,(1,w_model,h_model,3))
            predict = model.predict(data)
            predict=predict[0].tolist()
            result=predict.index(np.max(predict))

            name = data_loaded[result]['Name']

            cv2.putText(roi, name+" "+str(result), (10, int(crop_img.shape[0] * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
            cv2.imshow("ROI", roi)

        # ================== IMAGE DISPLAY
        img = cv2.resize(img,(640,480))
        cv2.imshow('LIVE', img)

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break


# ============================= MAIN =============================#
if __name__== "__main__":

    # Define dimensions for the images to be trained (they will be shrinked to this size)
    w=100
    h=100

    # Get current directory.
    curr_dir=os.path.dirname(os.path.realpath(sys.argv[0]))

    # Check if the directory ends with /
    if curr_dir[-1] != '\\':
        curr_dir+="\\"
    print ("\nDirectory Name:")
    print (curr_dir)
    print ("\n")

    # Create Dataset folder
    dataset_fold = curr_dir + "datasets"
    if not os.path.exists(dataset_fold):
        os.makedirs(dataset_fold)

    # Create Weights folder
    weights_fold = curr_dir + "Weights"
    if not os.path.exists(weights_fold):
        os.makedirs(weights_fold)

    # Create .yaml file
    if not os.path.exists(curr_dir + "data.yaml"):
        file = open(curr_dir + "data.yaml", "w")
        file.close()

    while (1):

        # Get Object named from Dataset folder
        no_of_folders = os.listdir(curr_dir + "datasets")
        #sep = '_'
        #no_of_folders = np.unique([s.split(sep, 1)[0] for s in no_of_folders])

        print (colored("Available Commands:\n"
                      "1:Generate New Data\n"
                      "2:Classify\n"
                      "3:Train\n"
                      "q:Quit", 'green'))
        mode = input('Enter command:')

        if mode == '1':

            s=1
            for i in no_of_folders:
                print (str(s)+"."+i)
                s+=1
            name = input('Enter Object No or enter new object name:')
            if name.isdigit():
                if int(name)<s+1:
                    name=no_of_folders[int(name)-1]
                    gen_photos(curr_dir, name)
                else:
                    continue
            else:
                object_fold = curr_dir + "datasets\\"+name
                if not os.path.exists(object_fold):
                    os.makedirs(object_fold)
                gen_photos(curr_dir, name)


        elif mode == '2':
            model = load_model(curr_dir+"Weights/face_classifier_model.h5")
            classify(w,h)


        elif mode == '3':
            model_train(w,h) #or "vgg16" , "vgg19"


        elif mode == 'q':  # QUIT
            break


        else:
            print (colored('Command not found!', 'red'))