"""
Implementation of the autoencoder model of Context Encoder
"""

import keras
from keras.datasets import cifar10
from keras.layers import Activation,BatchNormalization,Flatten,MaxPooling2D,ZeroPadding2D
from keras.layers import Input,Conv2D,Conv2DTranspose,Dense,Reshape,Dropout,UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam

from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

#lr_scheduler
class CEncoder():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.mask_height = 16
        self.mask_width = 16
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.patch_shape = (self.mask_height,self.mask_width,self.channels)
        self.overlap = 3

        lambda_recon = 0.5
        lambda_adv = 0.5
        optimizer_recon = Adam(lr=2e-3, beta_1=0.5,decay=1e-2)
        optimizer_adv = Adam(lr=2e-3, beta_1=0.5,decay=1e-3)


        self.gf = 32
        self.df = 16 # masked patch size


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

    """ Building models """
    def build_autoencoder(self):
        def conv_bn_relu_pool(input_layer,filters,kernel_size=4,stride=2,pad='same',activation='relu'):
            y = Conv2D(filters,kernel_size=kernel_size,
                                strides=stride,padding=pad)(input_layer)

            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = LeakyReLU(alpha=0.2)(y)
            else:
                print("WARNING: no activation layer")
                pass
            y = BatchNormalization()(y)
            output = y
            return output
        def deconv_bn_relu(input_layer,filters,kernel_size=4,stride=2,pad='same', activation='relu'):
            y = Conv2DTranspose(filters,kernel_size=kernel_size,
                                strides=stride,padding=pad)(input_layer)
            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = LeakyReLU(alpha=0.2)(y)
            else:
                input("WARNING: no activation layer <Enter to continue>")
                pass
            y = BatchNormalization()(y)
            output = y
            return output


        # AlexNet
        input = Input(shape=self.img_shape)

        y = conv_bn_relu_pool(input,filters=self.gf,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf*2,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf*4,activation='lrelu')
        #y = conv_bn_relu_pool(y,filters=self.gf*8,activation='relu')
        #y = conv_bn_relu_pool(y,filters=self.gf*16,activation='relu')
        latent = conv_bn_relu_pool(y,filters=1000,stride=1,pad='valid',activation='lrelu') # increase to 4000
        latent = Dropout(0.5)(latent)

        y = deconv_bn_relu(latent,filters=self.df*4,stride=1,pad='valid',activation='relu')
        #y = deconv_bn_relu(y,filters=self.df*8,activation='relu')
        #y = deconv_bn_relu(latent,filters=self.df*4,activation='relu')
        y = deconv_bn_relu(y,filters=self.df*2,activation='relu')
        y = deconv_bn_relu(y,filters=self.df,activation='relu')

        output = Conv2DTranspose(3,kernel_size=4,padding='same')(y)
        ouptut = Activation('tanh')(output)

        model = Model(input,output)
        model.summary()
        return model

    def build_discriminator(self):
        def conv_bn_relu_pool(input_layer,filters,kernel_size=4,stride=2,pad='same',activation='relu'):
            y = Conv2D(filters,kernel_size=kernel_size,
                                strides=(stride,stride),padding=pad)(input_layer)
            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = LeakyReLU(alpha=0.2)(y)
            else:
                print("WARNING: no activation layer")
                pass
            y = BatchNormalization()(y)
            output = y
            return output

        patch = Input(shape=self.patch_shape)
        y = conv_bn_relu_pool(patch,filters=64,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=128,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=256,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=512,activation='lrelu')

        y = Flatten()(y)
        validity = Dense(1,activation='sigmoid')(y) # review activation here
        model = Model(patch,validity)

        model.summary()
        return model

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty((imgs.shape[0], self.mask_height, self.mask_width, self.channels))
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
            masked_img[_y1 + self.overlap:_y2 - self.overlap,
                       _x1 + self.overlap:_x2 - self.overlap, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs, missing_parts, (y1, y2, x1, x2)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = np.vstack((X_train, X_test))
        y_train = np.vstack((y_train, y_test))

        # Extract dogs and cats
        X_cats = X_train[(y_train == 3).flatten()]
        X_dogs = X_train[(y_train == 5).flatten()]
        X_train = np.vstack((X_cats, X_dogs))

        # Rescale -1 to 1
        X_train = X_train / 255
        X_train = 2 * X_train - 1
        y_train = y_train.reshape(-1, 1)

        half_batch = int(batch_size / 2)

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

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            masked_imgs, missing_parts, _ = self.mask_randomly(imgs)

            # Generator wants the discriminator to label the generated images as valid
            valid = np.ones((batch_size, 1))

            # Train the generator
            g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f, acc: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1], 100*g_loss[3]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)

    def sample_images(self, epoch, imgs):
        r, c = 3, 6

        masked_imgs, missing_parts, (y1, y2, x1, x2) = self.mask_randomly(imgs)
        gen_missing = self.autoencoder.predict(masked_imgs)

        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_missing = 0.5 * gen_missing + 0.5

        gen_missing = np.where(gen_missing < 0, 0, gen_missing)


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
        fig.savefig("images/cifar_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.autoencoder, "context_encoder_generator")
        save(self.discriminator, "context_encoder_discriminator")

if __name__ == '__main__':
    ce = CEncoder()
    ce.train(epochs=30000, batch_size=64, sample_interval=50)
