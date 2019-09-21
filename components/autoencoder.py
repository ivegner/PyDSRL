'''Low-level entity extraction autoencoder'''
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard

from scipy.ndimage.filters import maximum_filter
from scipy import stats
from scipy.spatial.distance import sqeuclidean
import numpy as np
from gym import logger
from matplotlib import pyplot as plt

ENTITY_DIST_THRESHOLD = 0.25
POOL_SIZE = 2

class SymbolAutoencoder():
    '''Implements the DSRL paper section 3.1. Extract entities from raw image'''
    def __init__(self, input_shape, neighbor_radius=25):
        self.neighbor_radius = neighbor_radius

        input_img = Input(shape=input_shape)
        encoded = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
        encoded = MaxPooling2D((POOL_SIZE, POOL_SIZE), padding='same')(encoded)

        decoded = UpSampling2D((POOL_SIZE, POOL_SIZE))(encoded)
        # Get decoded number of channels, for now assume channels_last ordering
        decoded = Conv2D(input_shape[2], (5, 5), activation='sigmoid', padding='same')(decoded)

        self.encoder = Model(input_img, encoded)
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_data, epochs=50, batch_size=128, shuffle=True,
              validation=None, tensorboard=False):
        '''Train the autoencoder on provided images'''
        if tensorboard:
            print('''Make sure you started the Tensorboard server with
                     tensorboard --logdir=/tmp/autoencoder
                     Go to http://0.0.0.0:6006 to view Tensorboard''')
        self.autoencoder.fit(train_data, train_data,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(validation, validation),
                             callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
                             if tensorboard else None,
                            )

    def encode(self, data):
        '''Pass data through the encoder part of the autoencoder'''
        return self.encoder.predict(data)

    def predict(self, data):
        '''Do a full pass through the autoencoder'''
        return self.autoencoder.predict(data)

    def _extract_positions(self, encoded_image):
        '''Finds locations of pixels of entities in the image'''
        features = np.max(encoded_image, axis=2)
        background_value = stats.mode(features, axis=None)[0][0]
        features -= background_value
        #apply the local maximum filter; all pixel of maximal value
        #in their neighborhood are set to 1
        filtered = maximum_filter(features, size=(4, 4))    #TODO: Abstract size
        filtered = np.asarray(filtered == features, dtype=int) - np.asarray(filtered == 0,
                                                                            dtype=int)
        filtered.reshape(encoded_image.shape[:-1])
        filtered *= POOL_SIZE # Pooling = downsampling = everything is scaled down by POOL_SIZE
        #2d image of the positions, and just the indices
        return filtered, np.transpose(np.nonzero(filtered))

    def visualize(self, images):
        '''Visualize autoencoder processing steps'''
        if len(images) > 20:
            raise Exception('Too many visualization images, please provide <= 20')
        logger.info('Visualizing...')

        encoded_imgs = self.encode(images)
        position_maps = [self._extract_positions(x)[0] for x in encoded_imgs]
        decoded_imgs = self.predict(images)

        def flatten_to_img(array):
            '''Reshape (x, y, 1) array to (x, y)'''
            return np.reshape(array, array.shape[:2])

        n_plots = len(images)
        plt.figure(figsize=(20, 4))
        for i in range(0, n_plots):
            plt_i = i+1
            # display original
            axis = plt.subplot(4, n_plots, plt_i)
            plt.imshow(flatten_to_img(images[i]))
            plt.gray()
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

            # display reconstruction
            axis = plt.subplot(4, n_plots, plt_i + n_plots)
            plt.imshow(flatten_to_img(decoded_imgs[i]))
            plt.gray()
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

            # display encoded
            axis = plt.subplot(4, n_plots, plt_i + 2*n_plots)
            plt.imshow(np.sum(encoded_imgs[i], axis=2).reshape((images[0].shape[0]//2,
                                                                images[0].shape[1]//2)))
            plt.gray()
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

            # display extracted entity centers
            axis = plt.subplot(4, n_plots, plt_i + 3*n_plots)
            plt.imshow(position_maps[i])
            plt.gray()
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

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
        pos_map, entities = self._extract_positions(encoded)

        repr_entity_activations = []    # Representative depth slice for a certain type
        typed_entities = []   # Actual Entity() array
        found_types = []
        # TODO: Enhancements: knn classifier instead of this caveman shit
        for entity_coords in entities:
            activations = encoded[entity_coords[0], entity_coords[1], :]
            if not repr_entity_activations:
                repr_entity_activations.append(activations)
                e_type = 'type0'

            else:
                for i, e_activations in enumerate(repr_entity_activations):
                    dist = sqeuclidean(activations, e_activations)
                    if dist < ENTITY_DIST_THRESHOLD:    # Same type
                        repr_entity_activations[i] = (e_activations + activations) / 2
                        e_type = 'type' + str(i)
                        break
                else:
                    # No type match, make new type
                    repr_entity_activations.append(activations)
                    new_type_idx = len(repr_entity_activations) - 1
                    e_type = 'type' + str(new_type_idx)

            min_coords = entity_coords-self.neighbor_radius
            min_coords = np.maximum(min_coords, 0)
            max_coords = entity_coords+self.neighbor_radius
            max_coords = np.minimum(max_coords, pos_map.shape)
            n_neighbors = np.count_nonzero(pos_map[min_coords[0]:max_coords[0],
                                                   min_coords[1]:max_coords[1]])
            typed_entities.append(Entity(position=entity_coords,
                                         entity_type=e_type,
                                         n_neighbors=n_neighbors))
            if e_type not in found_types:
                found_types.append(e_type)

        return typed_entities, found_types

    @staticmethod
    def from_saved(filename, input_shape, neighbor_radius=None):
        '''Load autoencoder weights from filename, given input shape'''
        if neighbor_radius is not None:
            ret = SymbolAutoencoder(input_shape, neighbor_radius=neighbor_radius)
        else:
            ret = SymbolAutoencoder(input_shape)
        ret.autoencoder.load_weights(filename)
        return ret

    def save_weights(self, filename):
        '''Save autoencoder weights to file'''
        self.autoencoder.save_weights(filename)

class Entity():
    '''Class for an entity that has an id, attributes, etc.'''
    def __init__(self, position, entity_type, n_neighbors=None, tracking_id=None):
        self.position = position
        self.entity_type = entity_type
        self.id = tracking_id
        self.last_transition = None
        self.n_neighbors = n_neighbors
        # Used for calculating interactions
        self.prev_state = {'position': self.position, 'entity_type': self.entity_type}
        # Used for marking entities for deletion
        self.exists = 1

    def update_self(self, entity):
        '''Update own params based on new entity object'''
        self.prev_state = {'position': self.position, 'entity_type': self.entity_type}
        self.position = entity.position
        self._transition(self.entity_type, entity.entity_type)

    def appeared(self):
        '''Set object transition as having spawned'''
        self._transition('null', self.entity_type)

    def disappeared(self):
        '''Set object transition as despawned, and mark it for deletion'''
        self._transition(self.entity_type, 'null')
        self.exists = 0 # mark for deletion

    def _transition(self, from_type, to_type):
        self.last_transition = [from_type, to_type]
        self.entity_type = to_type
