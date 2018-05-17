"""
Implementation of Context Encoder using cifar10 images
"""

import keras
from keras import losses
from keras.datasets import cifar10
from keras.layers import Activation,BatchNormalization,Flatten,MaxPooling2D,ZeroPadding2D
from keras.layers import Input,Conv2D,Conv2DTranspose,Dense,Reshape,Dropout,UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential, model_from_json
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim as SSIM

import scipy
from scipy import misc
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MEAN_VIGNET_TRAIN = np.array([8.25, 7.66, 6.855])
MEAN_VIGNET_TRAIN_NORM = (MEAN_VIGNET_TRAIN) / 255.0
MEAN_VIGNET_TRAIN_NORM = (2 * MEAN_VIGNET_TRAIN_NORM) - 1

def compute_mse(a,b):
    A = a.astype('float32').flatten()
    B = b.astype('float32').flatten()
    mse = mean_squared_error(A,B)
    return mse

def compute_ssim(a,b):
    ssim = []
    num_imgs = a.shape[0]
    for i in range(num_imgs):
        ssim.append( SSIM(a[i].astype('float32'),
                          b[i].astype('float32'),
                          data_range=(1-(-1)),
                          multichannel=True))

    # dssim
    return ( 1 - np.array(ssim).mean() ) / 2

class CEncoder():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.mask_height = 64
        self.mask_width = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.patch_shape = (self.mask_height,self.mask_width,self.channels)
        self.overlap = 7
        SHOW_SUMMARY = False

        lambda_recon = 0.999
        lambda_adv = 0.001
        optimizer_recon = Adam(lr=2e-3, beta_1=0.5,decay=1e-2)
        optimizer_adv = Adam(lr=2e-3, beta_1=0.5,decay=1e-3)

        self.gf = 64
        self.df = 64 # masked patch size

        # Construct autoencoder
        self.autoencoder = self.build_autoencoder()
        # Construct discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_adv, metrics=['accuracy'])
        # Construct GAN
        masked_img = Input(shape=self.img_shape)
        #gen_missing = self.autoencoder(masked_img)
        gen_missing = self.autoencoder(masked_img)

        self.discriminator.trainable = False

        valid = self.discriminator(gen_missing)

        self.combined = Model(masked_img, [gen_missing, valid])
        self.combined.compile(loss=['mse','binary_crossentropy'],
            loss_weights=[lambda_recon,lambda_adv],
            optimizer=optimizer_recon,
            metrics=['accuracy','accuracy'])

        if SHOW_SUMMARY:
            print('DISCRIMINATOR')
            self.discriminator.summary()
            print('AUTOENCODER')
            self.autoencoder.summary()

    """ Building models """
    def build_autoencoder(self):
        def conv_bn_relu_pool(input_layer,filters,kernel_size=4,stride=2,pad='same',activation='relu'):
            y = Conv2D(filters,kernel_size=kernel_size,
                                strides=stride,padding=pad,kernel_initializer=RandomNormal(stddev=0.005))(input_layer)

            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = LeakyReLU(alpha=0.1)(y)
            else:
                print("WARNING: no activation layer")
                pass
            y = BatchNormalization(epsilon=1e-8)(y)
            output = y
            return output
        def deconv_bn_relu(input_layer,filters,kernel_size=4,stride=2,pad='same', activation='relu'):
            y = Conv2DTranspose(filters,kernel_size=kernel_size,
                                strides=stride,padding=pad,kernel_initializer=RandomNormal(stddev=0.005))(input_layer)
            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = LeakyReLU(alpha=0.1)(y)
            else:
                input("WARNING: no activation layer <Enter to continue>")
                pass
            y = BatchNormalization(epsilon=1e-8)(y)
            output = y
            return output


        # AlexNet
        input = Input(shape=self.img_shape)

        y = conv_bn_relu_pool(input,filters=self.gf,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf*2,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf*4,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf*8,activation='relu')
        latent = conv_bn_relu_pool(y,filters=4000,stride=1,pad='valid',activation='lrelu') # increase to 4000
        latent = Dropout(0.5)(latent)

        y = deconv_bn_relu(latent,filters=self.df*8,stride=1,pad='valid',activation='relu')
        y = deconv_bn_relu(y,filters=self.df*4,activation='relu')
        y = deconv_bn_relu(y,filters=self.df*2,activation='relu')
        y = deconv_bn_relu(y,filters=self.df,activation='relu')

        y = Conv2DTranspose(3,kernel_size=4,strides=2,padding='same')(y)
        output = Activation('tanh')(y)

        model = Model(input,output)

        return model

    def build_discriminator(self):
        def conv_bn_relu_pool(input_layer,filters,kernel_size=4,stride=2,pad='same',activation='relu'):
            y = Conv2D(filters,kernel_size=kernel_size,
                        strides=stride,padding=pad,kernel_initializer=RandomNormal(stddev=0.005))(input_layer)
            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = LeakyReLU(alpha=0.1)(y)
            else:
                print("WARNING: no activation layer")
                pass
            y = BatchNormalization(epsilon=1e-8)(y)
            output = y
            return output

        patch = Input(shape=self.patch_shape)
        y = conv_bn_relu_pool(patch,filters=self.df,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.df*2,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.df*4,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.df*8,activation='lrelu')

        y = Flatten()(y)
        validity = Dense(1,activation='sigmoid',
                            kernel_initializer=RandomNormal(stddev=0.005))(y) # review activation here
        model = Model(patch,validity)

        return model

    def mask_randomly(self, imgs,corner=None):
        y1 = np.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        # overrides the values on top
        if corner:
            y1 = corner[0]
            y2 = corner[0] + self.mask_height
            x1 = corner[1]
            x2 = corner[1] + self.mask_width

        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty((imgs.shape[0], self.mask_height, self.mask_width, self.channels))
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
            masked_img[_y1 + self.overlap:_y2 - self.overlap,
                       _x1 + self.overlap:_x2 - self.overlap, 0] = MEAN_VIGNET_TRAIN_NORM[0]
            masked_img[_y1 + self.overlap:_y2 - self.overlap,
                       _x1 + self.overlap:_x2 - self.overlap, 1] = MEAN_VIGNET_TRAIN_NORM[1]
            masked_img[_y1 + self.overlap:_y2 - self.overlap,
                       _x1 + self.overlap:_x2 - self.overlap, 2] = MEAN_VIGNET_TRAIN_NORM[2]
            masked_imgs[i] = masked_img

        return masked_imgs, missing_parts, (y1, y2, x1, x2)

    def train(self, epochs, batch_size=128, sample_interval=50):
        print("Loading data ...")
        PATH_TO_DATA = '/media/jomari/HDD/VIGNET/inpainting/train/npz/train.npz'
        vignet = np.load(PATH_TO_DATA)
        X_train = vignet['images']
        y_train = vignet['labels']

        # Rescale possible rescale
        X_train = np.array([misc.imresize(x, [self.img_rows, self.img_cols]) for x in X_train]) # resize to (128,128)

        # Rescale -1 to 1
        X_train = X_train.astype('float32') / 255.0
        X_train = (2 * X_train) - 1
        y_train = y_train.reshape(-1, 1)

        half_batch = int(batch_size / 2)

        batch_count = X_train.shape[0]//batch_size
        print('Epochs:', epochs)
        print('Batch size:', batch_size)
        print('Batch count:', batch_count)

        sample_interval_epoch = sample_interval*4
        prev_mse = 1e10
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            masked_imgs, missing, _ = self.mask_randomly(imgs)

            # Generate a half batch of new images
            gen_missing = self.autoencoder.predict(masked_imgs)

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(missing, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            masked_imgs, missing_parts, _ = self.mask_randomly(imgs)

            # Generator wants the discriminator to label the generated images as valid
            valid = np.ones((batch_size, 1))

            # Train the generator
            g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f, acc: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1], 100*g_loss[4]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)

            if epoch % sample_interval_epoch == 0:
                self.save_model(epoch)


    def test(self, batch_size=10,vehicle='car', arch='a.json', weights='a.hdf5'):
        """ Produce same images (patch places) in data """
        PATH_TO_DATA = './wpatch/' + vehicle + '_wpatch.test.npz'
        vignet = np.load(PATH_TO_DATA)
        X_test_gt = vignet['gt']
        X_test_crops = vignet['crops']
        xcoor_test = vignet['xs']
        ycoor_test = vignet['ys']

        # Rescale possible rescale
        X_test_gt = np.array([misc.imresize(x, [self.img_rows, self.img_cols]) for x in X_test_gt]) # resize to (128,128)

        # Rescale -1 to 1
        X_test_crops = X_test_crops.astype('float32') / 255.0
        X_test_crops = (2 * X_test_crops) - 1
        X_test_gt = X_test_gt.astype('float32') / 255.0
        X_test_gt = (2 * X_test_gt) - 1

        # load the model
        file_arch = './context/saved_model/context_generator.json'
        file_weights = './context/saved_model/context_generator_weights.29700.hdf5'
        self.load_model(arch, weights)

        num_images = X_test_gt.shape[0]
        batch_count = num_images//batch_size
        batches = np.array_split(np.arange(num_images),batch_count)
        mse_image = []
        ssim_image = []
        mse_patch = []
        ssim_patch = []

        for i,batch in enumerate(batches):
            percent_complete = (i/batch_count) * 100

            imgs_alpha = X_test_gt[batch]
            corners = (ycoor_test[batch], xcoor_test[batch])
            crops = X_test_crops[batch,:,:,:3]

            masked_imgs, _, _ = self.mask_randomly(imgs_alpha[:,:,:,:3],corner=corners)

            gen_missing = self.model.predict(masked_imgs)
            gen_imgs_alpha = imgs_alpha.copy()

            for j in range(len(batch)):
                gen_imgs_alpha[j,corners[0][j]:corners[0][j]+64,
                                 corners[1][j]:corners[1][j]+64,:3] = gen_missing[j]

            scipy.misc.imsave('images_test/sample-%s.png'%i,gen_imgs_alpha[0])

            mse_image.append(compute_mse(gen_imgs_alpha[:,:,:,:3], imgs_alpha[:,:,:,:3]))
            ssim_image.append(compute_ssim(gen_imgs_alpha[:,:,:,:3], imgs_alpha[:,:,:,:3]))
            mse_patch.append(compute_mse(gen_missing, crops))
            ssim_patch.append(compute_ssim(gen_missing, crops))

            #print(mse_image,mse_patch)
            #input('...')

            print(' Test completed: ', "{0:.5f}%".format(percent_complete), end="\r")

        print(' ')
        print( "MSE (image): ",np.array(mse_image).mean()  )
        print( "SSIM(image): ",np.array(ssim_image).mean() )
        print( "MSE (patch): ",np.array(mse_patch).mean()  )
        print( "SSIM(patch): ",np.array(ssim_patch).mean() )

    def predict(self,vehicle='car', arch='a.json', weights='a.hdf5',idx=1):
        PATH_TO_DATA = './wpatch/' + vehicle + '_wpatch.test.npz'
        vignet = np.load(PATH_TO_DATA)
        X_test_gt = vignet['gt']
        xcoor_test = vignet['xs']
        ycoor_test = vignet['ys']
        # Rescale possible rescale
        X_test_gt = np.array([scipy.misc.imresize(x, [self.img_rows, self.img_cols]) for x in X_test_gt]) # resize to (128,128)

        # Rescale -1 to 1
        X_test_gt = X_test_gt.astype('float32') / 255.0
        X_test_gt = (2 * X_test_gt) - 1

        # load the model
        self.load_model(arch, weights)

        sample_imgs_alpha = X_test_gt[idx]

        corners = (ycoor_test[idx], xcoor_test[idx])
        masked_imgs, _, _ = self.mask_randomly(sample_imgs_alpha[:,:,:,:3],
                                                corner=corners)

        gen_missing = self.model.predict(masked_imgs)

        gen_imgs_alpha = sample_imgs_alpha.copy()

        for j in range(len(gen_missing)):
            gen_imgs_alpha[j,corners[0][j]:corners[0][j]+64,
                             corners[1][j]:corners[1][j]+64,:3] = gen_missing[j]

        gen_imgs = gen_imgs_alpha[:,:,:,:3]

        WITH_ALPHA = False

        for i,index in enumerate(idx):
            print('Saved image: context-%s.png'%index)
            if WITH_ALPHA:
                scipy.misc.imsave('sample_output/context-%s.png'%index, gen_imgs_alpha[i])
            else:
                scipy.misc.imsave('sample_output/context-%s.png'%index, gen_imgs[i])

    def sample_images(self, epoch, imgs):
        r, c = 3, 6

        masked_imgs, missing_parts, (y1, y2, x1, x2) = self.mask_randomly(imgs)
        gen_missing = self.autoencoder.predict(masked_imgs)

        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_missing = 0.5 * gen_missing + 0.5

        gen_missing = np.clip(gen_missing,0,1)

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0,i].imshow(imgs[i, :,:])
            axs[0,i].axis('off')
            axs[1,i].imshow(masked_imgs[i, :,:])
            axs[1,i].axis('off')
            filled_in = imgs[i].copy()
            filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
            axs[2,i].imshow(filled_in)
            axs[2,i].axis('off')
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

        save(self.autoencoder, "context_generator",epoch=epoch)
        save(self.discriminator, "context_discriminator", epoch=0)

    def load_model(self,arch,weights):
        """ load the weights and architecture of model """
        json_file = open(arch, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(weights)

if __name__ == '__main__':
    ce = CEncoder()
    MODE = 'predict'
    ARCH = './context/saved_model/context_generator.json'
    WEIGHTS = './context/saved_model/context_generator_weights.20000.hdf5'
    mode = {'train': 0,
            'test': 1,
            'predict': 2}

    if mode[MODE] == 0:
        ce.train(epochs=30000, batch_size=32, sample_interval=50)
    elif mode[MODE] == 1:
        ce.test(batch_size=10,
                vehicle = 'car',
                arch = ARCH,
                weights = WEIGHTS)

    elif mode[MODE] == 2:
        car_indices = np.array([2,9,28,37,42,51,62,63,64,67,83,90,91,105,109,133,139,147,192,203])
        motor_indices = np.arange(20)
        ce.predict(vehicle='car',
                  arch=ARCH,
                  weights=WEIGHTS,
                  idx=car_indices)
