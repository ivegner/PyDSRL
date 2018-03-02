from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

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

    #TODO: Visualize reconstructed and encoded data

    def get_entities(self, image):
        '''
        Finds entities in image using autoencoder
        Returns {'type1': [(x_1, y_1), (x_2, y_2)... list of centers for all type_1],
                 'type2': [list of centers for all type_2],
                 etc.
                }
        '''
        pass

    @staticmethod
    def from_saved(filename, input_shape):
        ret = SymbolAutoencoder(input_shape)
        ret.autoencoder.load_weights(filename)
        return ret

    def save_weights(self, filename):
        self.autoencoder.save_weights(filename)