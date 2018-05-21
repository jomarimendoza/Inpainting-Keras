"""
Implementation of Context-Conditional GAN using cifar10 images
From: https://github.com/eriklindernoren/Keras-GAN/blob/master/ccgan/ccgan.py
"""

from __future__ import print_function, division

from keras.datasets import cifar10
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate,Multiply,Add,Maximum,Average
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import scipy
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim as SSIM

import tensorflow as tf

from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def compute_mse(a,b):
    A = a.flatten()
    B = b.flatten()
    mse = mean_squared_error(A,B)
    return mse

def compute_ssim(a,b):
    ssim = []
    num_imgs = a.shape[0]
    for i in range(num_imgs):
        ssim.append( SSIM(a[i],b[i],
                      data_range=(1-(-1)),
                      multichannel=True))

    # dssim
    return ( 1 - np.array(ssim).mean() ) / 2

class CCGAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.mask_height = 64
        self.mask_width = 64
        self.channels = 3
        self.num_classes = 2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_LR_shape = (self.img_rows//4, self.img_cols//4, self.channels)
        self.LR_input = False # SET IF YOU WANT THE LOW RES CONDITION TO BE TRUE

        # Number of filters in first layer of generator and discriminator
        self.gf = 64
        self.df = 64

        lambda_recon = 0.999
        lambda_adv = 0.001
        optimizer_recon = Adam(lr=2e-3, beta_1=0.5,decay=1e-2)
        optimizer_adv =   Adam(lr=2e-3, beta_1=0.5,decay=1e-3)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer_adv,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        masked_img = Input(shape=self.img_shape)
        boolean_mask = Input(shape=self.img_shape)
        boolean_mask_reverse = Input(shape=self.img_shape)
        if self.LR_input:
            LR_img = Input(shape=self.img_LR_shape)
            _, gen_img = self.generator([masked_img,boolean_mask,boolean_mask_reverse,LR_img])
        else:
            _, gen_img = self.generator([masked_img,boolean_mask,boolean_mask_reverse])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The valid takes generated images as input and determines validity
        valid, _ = self.discriminator(gen_img)

        # The combined model  (stacked generator and discriminator) takes
        # masked_img as input => generates images => determines validity
        if self.LR_input:
            self.combined = Model([masked_img,boolean_mask,boolean_mask_reverse,LR_img], [gen_img, valid])
        else:
            self.combined = Model([masked_img,boolean_mask,boolean_mask_reverse], [gen_img, valid])
        self.combined.compile(loss=['mse','binary_crossentropy'],
            loss_weights=[lambda_recon, lambda_adv],
            optimizer=optimizer_recon,
            metrics=['accuracy'])

    def build_generator(self):
        """Create generator model"""
        def conv_bn_relu_pool(input_layer,filters,kernel_size=4,pool=True,pad='same',activation='relu',bn=True):
            if pool:
                y = Conv2D(filters,kernel_size=kernel_size,strides=2,padding=pad)(input_layer)
            else:
                y = Conv2D(filters,kernel_size=kernel_size,strides=1,padding=pad)(input_layer)
            #Activation
            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = LeakyReLU(alpha=0.2)(y)
            else:
                print("WARNING: no activation layer")
                pass
            #BatchNormalization
            if bn:
                y = BatchNormalization()(y)
                #y = InstanceNormalization()(y)
            else:
                pass
            output = y
            return output

        def deconv_bn_relu(input_layer,filters,kernel_size=4,stride=2,pad='same', activation='relu'):
            y = Conv2DTranspose(filters,kernel_size=kernel_size,
                                strides=stride,padding=pad)(input_layer)
            #Activation
            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = LeakyReLU(alpha=0.2)(y)
            else:
                input("WARNING: no activation layer <Enter to continue>")
                pass
            #BatchNormalization
            y = BatchNormalization()(y)
            output = y
            return output

        img = Input(shape=self.img_shape)
        mask = Input(shape=self.img_shape,name='boolean_mask')      # boolean tensor
        revmask = Input(shape=self.img_shape,name='boolean_inverted') # 0 at patch
        if self.LR_input:
            img_LR = Input(shape=self.img_LR_shape,name='low_res')

        # Downsampling
        y = conv_bn_relu_pool(img, filters=self.gf,activation='lrelu')
        y = conv_bn_relu_pool(y, filters=self.gf*2,activation='lrelu')
        if self.LR_input:
            y = Concatenate(axis=-1)([y,img_LR])
        y = conv_bn_relu_pool(y, filters=self.gf*4,activation='lrelu')
        y = conv_bn_relu_pool(y, filters=self.gf*8,activation='lrelu')
        y = deconv_bn_relu(y,filters=self.df*4,activation='relu')
        y = deconv_bn_relu(y,filters=self.df*2,activation='relu')
        y = deconv_bn_relu(y,filters=self.df,activation='relu')

        y = Conv2DTranspose(filters=self.channels, kernel_size=4, strides=2, padding='same')(y)
        output_img = Activation('tanh')(y)

        layer_patch =  Multiply()([img,revmask])
        layer_notPatch = Multiply()([output_img,mask])
        patched_img = Add()([layer_patch,layer_notPatch])

        if self.LR_input:
            model = Model([img, mask, revmask, img_LR], [output_img,patched_img])
        else:
            model = Model([img, mask, revmask], [output_img,patched_img])

        print("GENERATOR")
        model.summary()
        return model

    def build_discriminator(self):
        """Create discriminator model"""
        def conv_bn_relu_pool(input_layer,filters,kernel_size=3,stride=2,pool=True,pad='same',activation='relu',bn=True):
            if pool:
                y = Conv2D(filters,kernel_size=kernel_size,strides=2,padding=pad)(input_layer)
            else:
                y = Conv2D(filters,kernel_size=kernel_size,strides=1,padding=pad)(input_layer)
            #Activation
            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = LeakyReLU(alpha=0.2)(y)
            else:
                print("WARNING: no activation layer")
                pass
            #BatchNormalization
            if bn:
                y = BatchNormalization()(y)
                #y = InstanceNormalization()(y)
            else:
                pass
            output = y
            return output

        """ model """
        img = Input(shape=self.img_shape)
        #"""
        # VGG-A' model
        y = conv_bn_relu_pool(img,self.df,activation='lrelu')
        y = conv_bn_relu_pool(y,self.df*2,activation='lrelu')
        y = conv_bn_relu_pool(y,self.df*4,pool=False,activation='lrelu')
        y = conv_bn_relu_pool(y,self.df*4,activation='lrelu')
        y = conv_bn_relu_pool(y,self.df*8,pool=False,activation='lrelu')
        y = conv_bn_relu_pool(y,self.df*8,activation='lrelu')
        y = conv_bn_relu_pool(y,self.df*8,pool=False,activation='lrelu')
        features = conv_bn_relu_pool(y,self.df*8,activation='lrelu')
        #"""
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(features) # 1 output
        validity = Flatten()(validity)
        validity = Dense(1,activation='sigmoid',name='validity')(validity)

        label = Conv2D(1, kernel_size=4, strides=1, padding='same')(features) # 1 output
        label = Flatten()(label)
        label = Dense(self.num_classes, activation="softmax",name='label')(label)

        model = Model(img, [validity, label])
        print("DISCRIMINATOR")
        model.summary()
        return model

    """ Insert a random mask in the image """
    def mask_randomly(self, imgs, random=True, corner=None):
        if random:
            """create a random mask within the image"""
            y1 = np.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
            y2 = y1 + self.mask_height
            x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
            x2 = x1 + self.mask_width
        else:
            """Creates mask only at the 4 corners of the image"""
            y1 = np.random.randint(0, 2, imgs.shape[0])*(self.img_rows-self.mask_height) # stores 0 and 1
            y2 = y1 + self.mask_height
            x1 = np.random.randint(0, 2, imgs.shape[0])*(self.img_rows-self.mask_width) # stores 0 and 1
            x2 = x1 + self.mask_width

        # overrides the values on top
        if corner:
            y1 = corner[0]
            y2 = corner[0] + self.mask_height
            x1 = corner[1]
            x2 = corner[1] + self.mask_width

        mask_coordinates = np.array([y1,y2,x1,x2])

        #create array with random numbers with same shape as imgs
        masked_imgs = np.empty_like(imgs)
        boolean_masks = np.zeros_like(imgs).astype('bool')
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            boolean_mask_img = np.zeros_like(img).astype('bool')
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i],
            masked_img[_y1:_y2, _x1:_x2, :] = 0            # middle, normalized [-1,1]
            boolean_mask_img[_y1:_y2, _x1:_x2, :] = True   # True at patch

            # batch of images and masks
            masked_imgs[i] = masked_img
            boolean_masks[i] = boolean_mask_img

        return masked_imgs, boolean_masks, mask_coordinates

    def train(self, epochs, batch_size=32, sample_interval=50):
        # Load the dataset
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = np.vstack((X_train, X_test))
        y_train = np.vstack((y_train, y_test))

        # Extract dogs and cats
        X_cats = X_train[(y_train == 3).flatten()]
        X_dogs = X_train[(y_train == 5).flatten()]
        X_train = np.vstack((X_cats, X_dogs))
        y_cats = np.zeros(X_cats.shape[0])
        y_dogs = np.ones(X_dogs.shape[0])
        y_train = np.vstack((y_cats, y_dogs))

        # Rescale possible rescale
        X_train = np.array([scipy.misc.imresize(x, [self.img_rows, self.img_cols]) for x in X_train]) # resize to (128,128)

        # Rescale -1 to 1
        X_train = X_train.astype('float32') / 255.0
        X_train = (2 * X_train) - 1
        y_train = y_train.reshape(-1, 1)

        half_batch  = batch_size//2
        batch_count = X_train.shape[0]//batch_size
        print('Epochs:', epochs)
        print('Batch size:', batch_size)
        print('Batch count:', batch_count)
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            labels = y_train[idx]

            LR_imgs = np.array([scipy.misc.imresize(x, [self.img_rows//4, self.img_cols//4]) for x in imgs])

            masked_imgs, boolean_masks, mask_coordinates = self.mask_randomly(imgs,random=True)
            boolean_inverted = np.invert(boolean_masks)

            # Generate a half batch of new images (try loading generator weights here)
            if self.LR_input:
                _,gen_imgs = self.generator.predict([masked_imgs,boolean_masks,boolean_inverted,LR_imgs])
            else:
                _,gen_imgs = self.generator.predict([masked_imgs,boolean_masks,boolean_inverted])


            valid = np.ones((half_batch,1))
            fake = np.zeros((half_batch,1))

            comb_validity = np.concatenate([valid,fake],axis=0)

            # not yet included in combined
            labels = to_categorical(labels, num_classes=self.num_classes)
            comb_labels = np.concatenate([labels,labels],axis=0)

            # patch cropped part of image
            mask_coordinates = mask_coordinates
            comb_imgs = np.concatenate([imgs,gen_imgs],axis=0)

            # Shuffle data again here
            sh = np.arange(comb_validity.shape[0])
            np.random.shuffle(sh)

            comb_validity = comb_validity[sh]
            comb_labels = comb_labels[sh]
            comb_imgs = comb_imgs[sh]

            d_loss = self.discriminator.train_on_batch(comb_imgs,[comb_validity,comb_labels])

            # ---------------------
            #  Train Generator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            LR_imgs = np.array([scipy.misc.imresize(x, [self.img_rows//4, self.img_cols//4]) for x in imgs])
            masked_imgs, boolean_masks, mask_coordinates = self.mask_randomly(imgs,random=True)
            boolean_inverted = np.invert(boolean_masks)

            # Generator wants the discriminator to label the generated images as valid
            valid = np.ones((batch_size,1))

            # Train the generator
            if self.LR_input:
                g_loss = self.combined.train_on_batch([masked_imgs,boolean_masks,boolean_inverted,LR_imgs], [imgs,valid])
            else:
                g_loss = self.combined.train_on_batch([masked_imgs,boolean_masks,boolean_inverted], [imgs,valid])

            # Plot the progress
            print ("%d [D loss: %f, op_acc: %.2f%%] [gen acc: %.2f%%, mse: %f]" % ( epoch, d_loss[0], 100*d_loss[3], 100*g_loss[4], g_loss[1]))

            # If at save interval => save generated image samples
            if (epoch) % sample_interval == 0:
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)
                self.save_model(epoch)

    def test(self, batch_size=10, arch='ccgan.json', weights='ccgan.hdf5'):
        """ Produce same images (patch places) in data """
        pass

    def predict(self,arch='ccgan.json', weights='ccgan.hdf5'):
        pass

    def sample_images(self, epoch, imgs):
        """ saves sample images of the generator """
        r, c = 4, 6

        LR_imgs = np.array([scipy.misc.imresize(x, [self.img_rows//4, self.img_cols//4]) for x in imgs])
        masked_imgs,boolean_masks,mask_coordinates = self.mask_randomly(imgs,random=True)
        boolean_inverted = np.invert(boolean_masks)

        # orig_gen = no patch
        if self.LR_input:
            orig_gen_imgs,gen_imgs = self.generator.predict([masked_imgs,boolean_masks,boolean_inverted,LR_imgs])
        else:
            orig_gen_imgs,gen_imgs = self.generator.predict([masked_imgs,boolean_masks,boolean_inverted])

        imgs = (imgs + 1.0) * 0.5
        masked_imgs = (masked_imgs + 1.0) * 0.5
        orig_gen_imgs = (orig_gen_imgs + 1.0) * 0.5
        gen_imgs = (gen_imgs + 1.0) * 0.5

        orig_gen_imgs = np.clip(orig_gen_imgs,0,1)
        gen_imgs = np.clip(gen_imgs,0,1)

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0,i].imshow(imgs[i, :, :, :])
            axs[0,i].axis('off')
            axs[1,i].imshow(masked_imgs[i, :, :, :])
            axs[1,i].axis('off')
            axs[2,i].imshow(orig_gen_imgs[i, :, :, :])
            axs[2,i].axis('off')
            axs[3,i].imshow(gen_imgs[i, :, :, :])
            axs[3,i].axis('off')
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self,epoch):
        """ save the weights and architecture """
        def save(model, model_name,epoch):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.%s.hdf5" % (model_name,str(epoch))
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "ccgan_generator",epoch=epoch)
        save(self.discriminator, "ccgan_discriminator", epoch=0)

    def load_model(self,arch,weights):
        """ load the weights and architecture of model """
        json_file = open(arch, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(weights)


if __name__ == '__main__':
    ccgan = CCGAN()
    MODE = 'train'
    ARCH = './saved_model/ccgan_generator.json'
    WEIGHTS = './saved_model/ccgan_generator_weights.220.hdf5'
    mode = {'train': 0,
            'test': 1,
            'predict': 2}

    if mode[MODE] == 0:
        ccgan.train(epochs=50000, batch_size=32, sample_interval=50)

    elif mode[MODE] == 1:
        pass

    elif mode[MODE] == 2:
        pass
