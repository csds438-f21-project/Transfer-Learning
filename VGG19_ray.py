# -*- coding: utf-8 -*-

import os
import ray
import json
import math
import time
import argparse
import warnings
import numpy as np
import ray.train as train
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import timedelta
from ray.train import Trainer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Input,Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class VGG19:

    def __init__(self, args, verbose=True):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.label_num = args.label_num
        self.verbose = verbose
        self.include_top = args.include_top
        self.tr_dir = args.tr_dir
        self.va_dir = args.va_dir
        self.input_size = 224
    
    # load data 
    def data_loader(self, global_batch_size):
        # image data generator
        tr_gen = ImageDataGenerator(rescale=1./255, shear_range=0.18, zoom_range=0.18, horizontal_flip=True)
        va_gen = ImageDataGenerator(rescale=1./255)

        # load ImageNet data from directories
        tr_data = tr_gen.flow_from_directory(
                self.tr_dir,
                target_size=(self.input_size, self.input_size),
                batch_size=global_batch_size,
                class_mode='categorical')
        va_data = va_gen.flow_from_directory(
                self.va_dir,
                target_size=(self.input_size, self.input_size),
                batch_size=global_batch_size,
                class_mode='categorical')
        
        if self.verbose:
            print('Data loading finished')
        
        return tr_data, va_data

    # build VGG19 model
    def model(self):
        start = time.time()
        img_input = Input(shape=(self.input_size,self.input_size, 3))
        model = Sequential()

        # convolution layers & max pooling
        # Block 1
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(self.input_size,self.input_size, 3), name='conv1_1'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_4'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))
		
        if self.include_top:
        	# dense layers
        	model.add(Flatten(name='flatten1'))
        	model.add(Dense(4096, activation='relu', name='fc1'))
        	model.add(Dropout(0.2))
        	model.add(Dense(4096, activation='relu', name='fc2'))
        	model.add(Dropout(0.2))
        	model.add(Dense(1000, activation='softmax', name='fc3'))

        # load weights
        if self.include_top:
            model.load_weights('./vgg19_weights_tf_dim_ordering_tf_kernels.h5')
            print('pre-trained weights: {}'.format('./vgg19_weights_tf_dim_ordering_tf_kernels.h5'))
        else:
            model.load_weights('./vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
            print('pre-trained weights: {}'.format('./vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'))
        
        if self.verbose:
            print('Build model finished')

        return model

    def train_model(self):
        tf_config = json.loads(os.environ["TF_CONFIG"])
        num_workers = len(tf_config["cluster"]["worker"])

        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        # global batch size
        global_batch_size = self.batch_size * num_workers

        # load data
        tr_data, va_data = self.data_loader(global_batch_size)

        with strategy.scope():
            # train model
            vgg19_model = self.model()
            for layer in vgg19_model.layers:
                layer.trainable=False
            vgg19_output=vgg19_model.output
            x=Flatten(name='flatten2')(vgg19_output)
            x=Dense(self.label_num,activation='softmax')(x)
            model=Model(inputs=vgg19_model.input,outputs=x)

            # optimizers, Adam
            adam = Adam(lr=self.lr)

            model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        fit_his = model.fit(tr_data, steps_per_epoch=math.ceil(14034/global_batch_size), epochs=self.epochs, 
                                  verbose=self.verbose, validation_data=va_data, validation_steps=math.ceil(3000/global_batch_size))
        
        self.plot_acc_loss(fit_his)

    def multi_train(self):
        ray.init()
        trainer = Trainer(backend="tensorflow", num_workers=2, use_gpu=True)
        trainer.start()
        results = trainer.run(train_func=self.train_model)
        trainer.shutdown()

    

    def plot_acc_loss(self, fit_his):
        # save figure of acc per epoch
        plt.plot(fit_his.history['accuracy'])
        plt.plot(fit_his.history['val_accuracy'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'validation'])
        plt.savefig('./acc_ray.png')
        plt.close()

        # save figure of loss per epoch
        plt.plot(fit_his.history['loss'])
        plt.plot(fit_his.history['val_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['train', 'validation'])
        plt.savefig('./loss_ray.png')
        plt.close()


def initialize():
    parser = argparse.ArgumentParser(description='VGG19 Transfer Learning Image Classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tr-dir', default='./data_small/train/', help='training data path')
    parser.add_argument('--va-dir', default='./data_small/val/', help='validation data path')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--include-top', type=bool, default=False, help='include top')
    parser.add_argument('--label-num', type=int, default=6, help='numbers of labels')
    args = parser.parse_args()

    return args


if __name__=='__main__':
    args = initialize()
    tf.config.list_physical_devices(device_type='GPU')
    start_time = time.time()
    VGG19 = VGG19(args=args, verbose=True)
    VGG19.multi_train()
    print('Training time: {}'.format(str(timedelta(seconds=time.time() - start_time))))














































