from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

from scipy.ndimage.filters import maximum_filter
from scipy import stats
from scipy.spatial.distance import sqeuclidean
import numpy as np
from gym import logger
from matplotlib import pyplot as plt

ENTITY_DIST_THRESHOLD = 0.25

class SymbolAutoencoder():
    '''Implements the DSRL paper section 3.1. Extract entities from raw image'''
    def __init__(self, input_shape):
        input_img = Input(shape=input_shape)
        x = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = UpSampling2D((2, 2))(encoded)
        decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

        self.encoder = Model(input_img, encoded)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_data, epochs=50, batch_size=128, shuffle=True,
                validation=None, tensorboard=False):
        if tensorboard:
            print('Make sure you started the Tensorboard server (tensorboard --logdir=/tmp/autoencoder)\n',
                  'Go to http://0.0.0.0:6006 to view Tensorboard')
        self.autoencoder.fit(train_data, train_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        validation_data=(validation, validation),
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')] if tensorboard else None,
                        )

    def encode(self, data):
        return self.encoder.predict(data)

    def predict(self, data):
        return self.autoencoder.predict(data)

    def extract_positions(self, encoded_image, return_map=False):
        '''Finds locations of pixels of entities in the image'''
        features = np.max(encoded_image, axis=2)
        background_value = stats.mode(features, axis=None)[0][0]
        features -= background_value
        #apply the local maximum filter; all pixel of maximal value
        #in their neighborhood are set to 1
        filtered = maximum_filter(features, size=(4,4))    #TODO: Abstract size
        filtered = np.asarray(filtered == features, dtype=int) - np.asarray(filtered == 0, dtype=int)
        filtered.reshape(encoded_image.shape[:-1])
        if return_map:
            #2d image of the positions
            return filtered
        else:
            #just the indices
            return np.transpose(np.nonzero(filtered))

    def visualize(self, images):
        '''Visualize autoencoder processing steps'''
        if len(images) > 20:
            raise Exception('Too many visualization images, please provide <20')
        logger.info('Visualizing...')

        encoded_imgs = self.encode(images)
        position_maps = [self.extract_positions(x, return_map=True) for x in encoded_imgs]
        decoded_imgs = self.predict(images)

        def flatten_to_img(array):
            '''Reshape (x, y, 1) array to (x, y)'''
            return np.reshape(array, array.shape[:2])

        n = len(images)
        plt.figure(figsize=(20, 4))
        for i in range(0, n):
            plt_i = i+1
            # display original
            ax = plt.subplot(4, n, plt_i)
            plt.imshow(flatten_to_img(images[i]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(4, n, plt_i + n)
            plt.imshow(flatten_to_img(decoded_imgs[i]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display encoded
            ax = plt.subplot(4, n, plt_i + 2*n)
            plt.imshow(np.sum(encoded_imgs[i], axis=2).reshape((images[0].shape[0]//2, images[0].shape[1]//2)))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display extracted entity centers
            ax = plt.subplot(4, n, plt_i + 3*n)
            plt.imshow(position_maps[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        print('\nPlot visible, close it to proceed')
        plt.show()

    def get_entities(self, image):
        '''
        Finds entities in image using autoencoder
        Returns {'type1': [(x_1, y_1), (x_2, y_2)... list of centers for all type_1],
                 'type2': [list of centers for all type_2],
                 etc.
                }
        '''

        encoded = self.encode(image.reshape((1,) + image.shape))[0]
        entities = self.extract_positions(encoded)

        repr_entity_activations = []
        entity_types = []
        # TODO: Enhancements: knn classifier instead of this caveman art
        for entity_coords in entities:
            activations = encoded[entity_coords[0], entity_coords[1], :]
            if not repr_entity_activations:
                repr_entity_activations.append(activations)
                entity_types.append([entity_coords])
                continue

            for i, e_activations in enumerate(repr_entity_activations):
                dist = sqeuclidean(activations, e_activations)
                if dist < ENTITY_DIST_THRESHOLD:
                    # Same type
                    entity_types[i].append(entity_coords)
                    break
            else:
                # No type match, make new type
                repr_entity_activations.append(activations)
                entity_types.append([entity_coords])


        # plt.figure()
        # pos_map = self.extract_positions(encoded, return_map=True)
        # # display original
        # ax = plt.subplot(2, 1, 1)
        # plt.imshow(pos_map)
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # ax = plt.subplot(2, 1, 2)
        # plt.imshow(image.reshape(image.shape[:-1]))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.show()
        # Images are compressed, multiply by 2 for real indices

        # sort arrays by sum of activations to ensure that same objects are
        # (probably) in the same type index every time
        sorted_perm = np.sum(repr_entity_activations, axis=1).argsort()
        entity_types = np.asarray(entity_types)[sorted_perm]
        # repr_entity_activations = repr_entity_activations[sorted_perm]
        return entity_types.tolist()

    @staticmethod
    def from_saved(filename, input_shape):
        ret = SymbolAutoencoder(input_shape)
        ret.autoencoder.load_weights(filename)
        return ret

    def save_weights(self, filename):
        self.autoencoder.save_weights(filename)
