#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Import library

#%matplotlib inline
import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import math
import keras
from keras.layers import (Input, Activation, Conv3D, Dense, Dropout, Flatten, 
                          MaxPooling3D, BatchNormalization, AveragePooling3D, 
                          Reshape, Lambda, GlobalAveragePooling3D, Concatenate,
                          ReLU, Add)

from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from scipy.fft import fft
from models.slowfast import SlowFast_body, bottleneck


from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint

import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# from keras.backend import set_session
from tensorflow.python.keras.backend import set_session #tensorflow==2.4.1
from keras.callbacks import CSVLogger


# In[ ]:

def plot_history(history, result_dir):
    '''
    Plots the accuracy and loss graphs of train and val and saves them.
    '''

    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    # plt.show()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    # plt.show();


# In[ ]:


class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename):
        
        frames = []
        index = len(os.listdir(filename)) // self.depth
        images = os.listdir(filename)[::index]
        images = images[0:25]
        images.sort()

        for img in images:

            img_path = os.path.join(filename, img)
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, (self.height, self.width))
            frames.append(frame)

        return np.array(frames) / 255.0


# In[4]:


def preprocess(video_dir, result_dir, nb_classes = 14, img_size = 224, frames = 25):
    '''
    Preprocess the videos into X and Y and saves in npz format and 
    computes input shape
    '''

    img_rows, img_cols  = img_size, img_size

    channel = 3

    files = os.listdir(video_dir)
    files.sort()

    if '.ipynb_checkpoints' in files:
        files.remove('.ipynb_checkpoints')

    X = []
    labels = []
    labellist = []

    # Obtain labels and X
    for filename in files:

        name = os.path.join(video_dir, filename)
        
        for v_files in os.listdir(name):
            
            v_file_path = os.path.join(name, v_files)
            label = filename
            if label not in labellist:
                if len(labellist) >= nb_classes:
                    continue
                labellist.append(label)
            labels.append(label)
            X.append(v_file_path)

    if not os.path.isdir(result_dir):
        os.makedir(result_dir)
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{} {}\n'.format(i, labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
                
    Y = np_utils.to_categorical(labels, nb_classes)

    print('X_shape:{}\tY_shape:{}'.format(len(X), Y.shape))

    input_shape = (frames, img_rows, img_cols, channel)

    return X, Y, input_shape


# In[5]:


#%run models/slowfast.py
import models.slowfast
# from models.slowfast import SlowFast_body, bottleneck
def resnet50(inputs, **kwargs):
    model = SlowFast_body(inputs, [3, 4, 6, 3], bottleneck, **kwargs)
    return model


# In[6]:


#%run models/i3dinception.py
import models.i3dinception
def I3DModel(model, nb_classes):

    x = model.layers[-1].output
    x = Dropout(0.5)(x)
    x = conv3d_bn(x, nb_classes, 1, 1, 1, padding='same', 
            use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, nb_classes))(x)
    # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(x)
    x = Activation('softmax', name='prediction')(x)
    model = Model(model.inputs, x)
    
    return model


# In[7]:


class batchGenerator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, vid3d):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.vid3d = vid3d

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = []

        for video in self.x[idx * self.batch_size:(idx + 1) * self.batch_size]:
            batch_x.append(self.vid3d.video3d(video))

        batch_x = np.array(batch_x)
        batch_x = batch_x.astype('float32')
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y


# In[9]:


def main(video_dir, result_dir, nb_classes = 14, batch_size = 32, epochs = 100, img_size = 224, frames = 25):

    X, Y, input_shape = preprocess(video_dir, result_dir, nb_classes, img_size, 
                                   frames)

    print("Input Shape = ", input_shape)

    vid3d = Videoto3D(img_size, img_size, frames)

    ## For i3D Inception model ##
#     i3dmodel = Inception_Inflated3d(include_top=False,
#                 weights='rgb_imagenet_and_kinetics',
#                 input_tensor=None,
#                 input_shape=input_shape,
#                 dropout_prob=0.5,
#                 endpoint_logit=False)
    
#     model = I3DModel(i3dmodel, nb_classes)
    
    ## For Slowfast model ##
    x = Input(shape = input_shape)
    model = resnet50(x, num_classes=nb_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
                  metrics=['accuracy'])
    
#     model = load_model('/content/drive/Shared drives/Drive 13/Dataset/Slowfast/slowfast-77-0.73.hd5')
    
    model.summary()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.15, shuffle = True)
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.15, shuffle = True)

    print("X_train.shape = ", len(X_train))
    print("X_val.shape = ", len(X_val))
    print("X_test.shape = ", len(X_test))

    # MODEL CHECK POINTS #

    # csv_logger = CSVLogger("/content/drive/Shared drives/Drive 13/Dataset/Slowfast/slowfast_model_history_log.csv", append=True)

    # filepath="/content/drive/Shared drives/Drive 13/Dataset/Slowfast/slowfast-{epoch:02d}-{val_accuracy:.2f}.hd5"
    csv_logger = CSVLogger("Logges/slowfast_model_history_log.csv", append=True)

    filepath="Logges/slowfast-{epoch:02d}-{val_accuracy:.2f}.hd5"


    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, mode='max')
    callbacks_list = [checkpoint, csv_logger]

    history = model.fit(batchGenerator(X_train, Y_train, batch_size, vid3d), steps_per_epoch = math.ceil(len(X_train) / batch_size), 
                                  validation_data = batchGenerator(X_val, Y_val, batch_size, vid3d), validation_steps = math.ceil(len(X_val) / batch_size), 
                                  epochs = epochs, verbose = 1, callbacks=callbacks_list, initial_epoch = 0)

    model_json = model.to_json()
    
    if not os.path.isdir(result_dir):
        os.makedir(result_dir)
    with open(os.path.join(result_dir, 'slowfast.json'), 'w') as json_file:
        json_file.write(model_json)
    
    model.save_weights(os.path.join(result_dir, 'slowfast_finalweights.hd5'))

    model.save(os.path.join(result_dir,'slowfast_finalmodel.hd5'))

    ###### if you want use previous trained mode uncomment these two line and comment above 6 line (from history=)
    # model = load_model('output/slowfast_finalmodel_New.hd5')
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
    #               metrics=['accuracy'])



    loss, acc = model.evaluate(batchGenerator(X_test, Y_test, batch_size, vid3d),
                               steps = math.ceil(len(X_test) / batch_size), verbose = 1)
    
    plot_history(history, result_dir)

    print('Test loss:', loss)
    print('Test accuracy:', acc)

    return history, model


# In[ ]:


## TRAIN MODEL ##

# history, model = main(video_dir = 'DCSASS Dataset/', result_dir = 'output/', nb_classes = 3, batch_size = 1, epochs = 2, img_size = 224, frames = 25)

# In[27]:


def frames_from_video(video_dir, nb_frames = 25, img_size = 224):

    # Opens the Video file
    cap = cv2.VideoCapture(video_dir)
    i=0
    frames = []
    while(cap.isOpened() and i<nb_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames) / 255.0


# In[20]:


def predictions(video_dir, model, nb_frames = 25, img_size = 224):

    X = frames_from_video(video_dir, nb_frames, img_size)
    X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))
    
    predictions = model.predict(X)
    preds = predictions.argmax(axis = 1)

    classes = []
    with open(os.path.join('output', 'classes.txt'), 'r') as fp:
        for line in fp:
            classes.append(line.split()[1])

    for i in range(len(preds)):
        print('Prediction - {} -- {}'.format(preds[i], classes[preds[i]]))






def predictions_video(video_dir, model, nb_frames = 25, img_size = 224):

    # X = frames_from_video(video_dir, nb_frames, img_size)

    # Opens the Video file
    cap = cv2.VideoCapture(video_dir)
    len_Vidframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    j=0
    while(cap.isOpened() and j<len_Vidframes):
        vid_parts=[]
        k=0
        while (k<1*nb_frames and j<len_Vidframes):  ###change max frame processed per time
            i=0
            frames = []
            while(i<nb_frames):
                ret, frame = cap.read()
                if ret == False:
                    break
                frame = cv2.resize(frame, (img_size, img_size))
                frames.append(frame)
                j+=1
                k+=1
                i+=1
            vid_part=np.array(frames) / 255.0
            if(i==nb_frames):
                vid_parts.append(vid_part)
            elif(j<nb_frames or k<nb_frames): ## small video repeat frame 
                n_repeat=int(np.ceil(nb_frames/k))
                vid_part = np.tile(vid_part, (n_repeat,1,1, 1))
                vid_part=vid_part[0:nb_frames,:]
                vid_parts.append(vid_part)
        X =np.array(vid_parts)
        X = np.reshape(X, (len(vid_parts), nb_frames, img_size, img_size, 3))
        frames_forShow=np.reshape( X[:,int(np.math.ceil(nb_frames/2)),:,:,:],  (len(vid_parts), img_size, img_size, 3) )
        predictions = model.predict(X)
        preds = predictions.argmax(axis = 1)
        Prob=[]
        for i in range(len(preds)):
            Prob.append(predictions[i,preds[i]]) #

        classes = []
        with open(os.path.join('output', 'classes.txt'), 'r') as fp:
            for line in fp:
                classes.append(line.split()[1])

        for i in range(len(preds)):
            print('Prediction - {} -- {}--{}'.format(preds[i], classes[preds[i]],Prob[i]))
            frame=frames_forShow[i,:]
            gray = cv2.cvtColor(np.array(frame*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # org
            org = (50, 50)
            
            # fontScale
            fontScale = 1
            
            # Blue color in BGR
            color = (255, 0, 0)
            
            # Line thickness of 2 px
            thickness = 2
            
            # Using cv2.putText() method
            gray = cv2.putText(gray, classes[preds[i]], org, font, fontScale, color, thickness, cv2.LINE_AA)
            gray = cv2.putText(gray, ' proba ='+str(Prob[i]), (10,70), font, fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow('frame', gray)
            cv2.waitKey(20)

    cap.release()
    cv2.destroyAllWindows()
     





 


# In[8]:



#input_shape = (frames, img_rows, img_cols, channel)
# input_shape = (25, 224, 224, 3)
# x1 = Input(shape = input_shape)
# mm = resnet50(x1, num_classes=14)
# mm.load_weights('output/slowfast_finalmodel.hd5')
# mm.save('output/slowfast_finalmodel_Dozdi.hd5')
## LOAD MODEL ##

# model = load_model('output/slowfast_finalmodel_New.hd5') # for predict 13 anomally (you need change lables and classes.txt for 13)
model = load_model('output/slowfast_finalmodel.hd5')

# In[29]:


## MAKE PREDICTIONS ##

# predictions(video_dir = 'test/Shoplifting018_x264_19.mp4', model = model, nb_frames = 25, img_size = 224)

predictions_video(video_dir = 'test/3.mp4', model = model, nb_frames = 25, img_size = 224)

# In[ ]:




