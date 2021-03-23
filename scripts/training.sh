#! /bin/bash

experiment_name='default'

load='../autoencoder_models/gray_10_model.h5'  # If pretrained autoencoder exist here is the file-path of the model
image_dir='../'
logdir='../logs'
log_level='info' # info, warn

evaluation_frequency=50


# Environment
n_entities=16
entity_size=10
neighborhood_size=10
step_size=2.0
overlap_factor=0.01

# Training parameters
alpha=0.01
epsilon_decay=0.99999


# Autoencdoer
filter_size=7

python ../main.py --experiment_name $experiment_name \
                  --load $load \
                  --logdir $logdir \
                  --image_dir $image_dir \
                  --log_level $log_level \
                  --evaluation_frequency $evaluation_frequency \
                  --n_entities $n_entities \
                  --entity_size $entity_size \
                  --neighborhood_size $neighborhood_size \
                  --step_size $step_size \
                  --overlap_factor $overlap_factor \
                  --alpha $alpha \
                  --epsilon_decay $epsilon_decay \
                  --filter_size $filter_size\


