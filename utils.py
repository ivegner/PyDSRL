import os

import gym
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from components.autoencoder import SymbolAutoencoder

def make_autoencoder_train_data(env_parameters, num, args, min_entities=1, max_entities=30):
    '''
    Make training images for the autoencoder

    env_parameters: (dict) dictionary that specifies the properties of the environment
    num: (int) number of samples the data should consist of
    min_entities, max_entities: (int) min/max number of entities that can appear \
                                in a single environment frame

    return: (np.array) BxWxHxC dataset of environment images
    '''

    temp_env = gym.make('CrossCircle-MixedRand-v0',**env_parameters)
    temp_env.seed(0)
    states = []
    for i in range(num):
        state = temp_env.make_random_state(min_entities, max_entities)
        if len(state)==0:
            continue
        states.append(state)
    args.logger.info(f'Final number of states collected in the current configuration {len(states)}')

    if (len(states)/num)<0.8:
        raise Exception('With the current environment configuration entities do /'
                        'not fit onto the grid without overlapping too much')
    return np.asarray(states)

def prepare_training(args):
    '''
    (1) Creates environment
    (2) Checks whether training images for the autoencoder exist, if not creates them
    (3) Creates the autoencoder
    (4) Trains or loads the weights of the autoencoder

    return: trained autoencoder, environment
    '''

    # Create the environment
    env_parameters = {'entity_size': args.entity_size,
                      'min_entities': args.n_entities,
                      'max_entities': args.n_entities,
                      'step_size': args.step_size,
                      'overlap_factor': args.overlap_factor}
    env = gym.make(args.env_id, **env_parameters)
    seed = env.seed(1)[0]

    # Load or create images
    if args.colour_state:
        GRAY = 'colour'
    else:
        GRAY = 'gray'

    TRAIN_IMAGES_FILE = f'train_images_{GRAY}.pkl'
    print(os.path.join(args.image_dir,TRAIN_IMAGES_FILE))
    if not os.path.exists(os.path.join(args.image_dir,TRAIN_IMAGES_FILE)) or args.new_images:
        args.logger.info('Making test images...')
        images = make_autoencoder_train_data(env_parameters, 5000, args, max_entities=20)
        with open(os.path.join(args.image_dir,TRAIN_IMAGES_FILE), 'wb') as f:
            pickle.dump(images, f)
    else:
        args.logger.info('Loading test images...')
        with open(os.path.join(args.image_dir,TRAIN_IMAGES_FILE), 'rb') as f:
            images = pickle.load(f)

    # Create the autoencoder
    input_shape = images[0].shape
    if args.load:
        autoencoder = SymbolAutoencoder.from_saved(args.load,
                                                   input_shape,
                                                   args.filter_size,
                                                   neighbor_radius=args.neighborhood_size)
    else:
        autoencoder = SymbolAutoencoder(input_shape, args.filter_size, neighbor_radius=args.neighborhood_size)


    # Train or load autoencoder
    if args.load_train or args.visualize or not args.load:
        args.logger.info('Splitting sets...')
        X_train, X_test = train_test_split(images, test_size=0.2, random_state=seed)
        X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=seed)

        if args.load_train or not args.load:
            args.logger.info('Training...')
            autoencoder.train(X_train, epochs=10, validation=X_val,tensorboard=args.tensorboard)

        if args.visualize:
            # Visualize autoencoder
            vis_imgs = X_test[:10]
            autoencoder.visualize(vis_imgs)

    if args.save:
        autoencoder.save_weights(os.path.join(args.save, f'{GRAY}_{args.entity_size}_model.h5'))


    # Visualize the results of the autoencoder
    if args.play:
        # Visualize your own moves for 10 steps
        state = env.reset()
        for i in range(20):
            state = np.reshape(state, (1,) + input_shape)
            autoencoder.visualize(state,show=True)
            action = int(input('Next action: '))
            state, reward, _, _ = env.step(action)
            print(f'The overall reward is {reward}')

    return autoencoder, env

