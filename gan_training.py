
import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf

h5_path = 'chest_xray.h5'

# Load images dataset using h5py
with h5py.File(h5_path, 'r') as h5_data:
    img_ds = h5_data['images'][:]
    img_ds = img_ds / 255
train_size = int(0.9 * len(img_ds))
x_train = img_ds[:train_size]
x_test = img_ds[train_size:]

from keras.models import Model,Sequential
from keras.layers import Dense,Flatten,Dropout,Activation,Lambda,Reshape
from keras.layers import Conv2D,ZeroPadding2D,UpSampling2D
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
import keras.backend as K
from keras.optimizers.legacy import Adam


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 32 * 32, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((32, 32, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("sigmoid"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50, last_model_point=0, summary_writer=None):

        # Rescale -1 to 1
        #X_train = X_train / 127.5 - 1.
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        #valid = np.ones((batch_size, 1))
        #fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            '''
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = np.array([X_train[i] for i in idx])

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)'''
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = np.array([x_train[i] for i in idx])
            gen_loss, disc_loss = self.train_step(imgs, batch_size)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, disc_loss, gen_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch, last_model_point, summary_writer['image'])
                self.evaluate(x_test, epoch, batch_size, summary_writer['evaluation'])
                self.generator.save('generated_models/Generator_model_{}'.format(epoch+last_model_point))
                self.discriminator.save('generated_models/Discriminator_model_{}'.format(epoch+last_model_point))

    def train_step(self, x_train, batch_size):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_loss, disc_loss = self.compute_loss(x_train, batch_size)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss 

    def compute_loss(self, real_images, batch_size):
        # Generate noise input for the generator
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images using the generator
            generated_images = self.generator(noise, training=True)

            # Get discriminator predictions for real and fake images
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate generator loss
            gen_loss = self.generator_loss(fake_output)

            # Calculate discriminator loss
            disc_loss = self.discriminator_loss(real_output, fake_output)

        return gen_loss, disc_loss

    def generator_loss(self, fake_output):
        # Generator loss: binary cross-entropy with target labels as valid (1)
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))

    def discriminator_loss(self, real_output, fake_output):
        # Discriminator loss: binary cross-entropy with target labels for real (1) and fake (0) images
        real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output))
        fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output))
        return real_loss + fake_loss
    
    def save_imgs(self, epoch, last_model_point, summary_writer=None):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='bone')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("generated_images/{}.png".format(epoch+last_model_point))
        plt.close()
        # Log selected images to TensorBoard
        if summary_writer:
            with summary_writer.as_default():
                tf.summary.image('Generated Image', gen_imgs, step=epoch+last_model_point, max_outputs=5)
            summary_writer.flush()
    
    def evaluate(self, x_test, epoch, batch_size, summary_writer=None):
        d_loss_real_values = []
        d_loss_fake_values = []
        g_loss_values = []
        d_accuracy_real_values = []
        d_accuracy_fake_values = []

        num_batches = len(x_test) // batch_size

        for i in range(num_batches):
            # Get the current batch
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_images = x_test[start_idx:end_idx]
            noise = np.random.normal(0, 1, (batch_images.shape[0], self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real, d_accuracy_real = self.discriminator.evaluate(batch_images, np.ones((batch_images.shape[0], 1)), verbose=0)
            d_loss_fake, d_accuracy_fake = self.discriminator.evaluate(gen_imgs, np.zeros((batch_images.shape[0], 1)), verbose=0)

            # Evaluate the combined model on noise and valid labels
            g_loss = self.combined.evaluate(noise, np.ones((batch_images.shape[0], 1)), verbose=0)

            d_loss_real_values.append(d_loss_real)
            d_loss_fake_values.append(d_loss_fake)
            g_loss_values.append(g_loss)
            d_accuracy_real_values.append(d_accuracy_real)
            d_accuracy_fake_values.append(d_accuracy_fake)

        #print(d_loss_real_values);print(d_loss_fake_values);print(g_loss_values);print(d_accuracy_real_values);print(d_accuracy_fake_values)
        
        # Log evaluation metrics to TensorBoard
        if summary_writer:
            with summary_writer.as_default():
                tf.summary.scalar('D Loss Real', np.mean(d_loss_real_values), step=epoch)
                tf.summary.scalar('D Loss Fake', np.mean(d_loss_fake_values), step=epoch)
                tf.summary.scalar('G Loss', np.mean(g_loss_values), step=epoch)
                tf.summary.scalar('D Accuracy Real', np.mean(d_accuracy_real_values), step=epoch)
                tf.summary.scalar('D Accuracy Fake', np.mean(d_accuracy_fake_values), step=epoch)
            summary_writer.flush()

from keras.models import load_model

def find_last_model_checkpoint():
    last_model_point=0
    for f in (glob('generated_models/Generator_model_*')):
        file = f.split('/')[-1]
        checkpoint_no = int(file.split('_')[-1])
        if checkpoint_no > last_model_point:
            last_model_point = checkpoint_no

    return int(last_model_point)

if __name__ == '__main__':
    dcgan = DCGAN()
    last_model_point=find_last_model_checkpoint()
    print ("Last checkpoint number = %d" %last_model_point)

    if os.path.exists('generated_models/Generator_model_{}'.format(last_model_point)):
        dcgan.generator = load_model('generated_models/Generator_model_{}'.format(last_model_point))
        dcgan.discriminator = load_model('generated_models/Discriminator_model_{}'.format(last_model_point))
        optimizer = Adam(0.0002, 0.5)
        dcgan.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        z = Input(shape=(100,))
        img = dcgan.generator(z)

        # For the combined model we will only train the generator
        dcgan.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = dcgan.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        dcgan.combined = Model(z, valid)
        dcgan.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    log_dir = './Tensorboard/dcgan/'
    summary_writer = {'image':tf.summary.create_file_writer(log_dir+'gen_images'),
                      'evaluation':tf.summary.create_file_writer(log_dir+'eval')}
    dcgan.train(epochs=2000, batch_size=128, save_interval=100,last_model_point=last_model_point, summary_writer=summary_writer)
    summary_writer.close()
