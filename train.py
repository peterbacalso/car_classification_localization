import os
import time
import argparse
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPooling2D, 
    Flatten, Dropout, BatchNormalization,
    Activation
)
from tensorflow.keras.callbacks import (
        TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from functools import partial

from hyperopt import STATUS_OK, Trials, tpe
from hyperas.distributions import choice, uniform
from hyperas import optim

from tensorflow.keras.backend import clear_session
from data.data_loader import DataLoader
from models.custom_cnn import model
from models.three_cnn_layer import CNN_3

clear_session() # Clear models from previous sessions

# CONSTANTS

# =============================================================================
# BATCH_SIZE=32
# SEED=23
# CHANNELS=1
# =============================================================================

def data():
    output="label_bbox"
    channels=1
    BATCH_SIZE=32
    SEED=23
    CHANNELS=1
    
    data = DataLoader('./data/cars_train', 
                  './data/cars_test', 
                  './data/devkit', 
                  batch_size=BATCH_SIZE)
    #n_classes = len(data.df_train['label'].unique())
    #print(f'{n_classes} CLASSES, Random Chance: {1/n_classes}')
    
    train_gen = data.get_pipeline(type='train',
                                  output=output,
                                  apply_aug=True,
                                  channels=channels,
                                  seed=SEED)
    #steps_per_epoch = np.ceil(len(data.df_train)/data.batch_size)
    #steps_per_epoch = tf.cast(steps_per_epoch, tf.int16).numpy()
    
    valid_gen = data.get_pipeline(type='validation',
                                  output=output,
                                  channels=channels,
                                  apply_aug=False,
                                  seed=SEED)
    #validation_steps = np.ceil(len(data.df_valid)/data.batch_size)
    #validation_steps = tf.cast(validation_steps, tf.int16).numpy()
    
    #return n_classes, train_gen, steps_per_epoch, valid_gen, validation_steps
    return train_gen, valid_gen
        


# =============================================================================
# def my_args():
#    args = argparse.Namespace()
#    args.ouput = 1
#    return args
# =============================================================================


if __name__=="__main__":
    
# =============================================================================
#     n_classes, train_gen, steps_per_epoch, \
#     valid_gen, validation_steps = load_data(channels=CHANNELS)
#     
#     print('nclasses', n_classes)
#     print('steps_per_epoch', steps_per_epoch)
#     print('validation_steps', validation_steps)
# =============================================================================
    
# =============================================================================
#     args = argparse.Namespace()
#     args.ouput = "label_bbox"
#     args.n_classes = n_classes
#     args.steps_per_epoch = steps_per_epoch
# =============================================================================

    # Classification Only
    # model = CNN_3(n_classes, channels=CHANNELS, output="label")
    # model = CNN(n_classes, lr=lr, reg=reg, channels=CHANNELS, output="label")
    # Bounding Box Only
    # model = CNN(n_classes, lr=lr, reg=reg, channels=CHANNELS, output="bbox")
    # Classification and Bounding Box
    # model = CNN(n_classes, lr=lr, reg=reg, channels=CHANNELS)
    
    train_generator, validation_generator = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    print("Evaluation of best performing model:")

    print(best_model.evaluate(validation_generator))











# HYPERPARAMS

# =============================================================================
# params = {
#         "lr":  [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
#         #"reg": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#         "reg": [1e-3, 1e-2, 1e-1]
# }
# =============================================================================
     

# TESTING

# =============================================================================
# test_gen_clf = data.get_pipeline(type='test', output='label', apply_aug=False)
# # test_gen_clf_localize = data.get_pipeline(type='test', apply_aug=False)
# steps=tf.math.ceil(len(data.df_test)/data.batch_size)
# tf.cast(steps, tf.int16).numpy()
# 
# index_to_label = data.labels.to_dict()['labels']
# 
# # Display Test Predictions
# def index_to_class(index, classes):
#     return classes[index]
# 
# def decode_predictions(y_probs, classes, top=3):
#     top_probs = np.sort(y_probs, axis=1)[:, ::-1][:, :top] # desc
#     top_indeces = np.argsort(y_probs, axis=1)[:, ::-1][:, :top] # desc
#     vfunc = np.vectorize(index_to_class)
#     top_cat = vfunc(top_indeces, classes)
#     return [top_probs, top_cat]
# 
# for images in test_gen_clf.take(1):
#     test_batch = images.numpy()
#     results = clf_model.predict(test_batch)
#     top_k = 3
#     top_k_probs, top_k_labels = decode_predictions(results, index_to_label, top=top_k)
#     plt.figure(figsize=(3,3))
#     for i in range(BATCH_SIZE):
#         print(f"Image #{i+1}")
#         img = test_batch[i]
#         img.resize(224,224)
#         plt.imshow(img)
#         plt.show()
#         for j in range(top_k):
#             print("    {:12s} {:.2f}%".format(top_k_labels[i][j], top_k_probs[i][j] * 100))
#         print()
# =============================================================================

# =============================================================================
# for images in test_gen_clf_localize.take(1):
#     test_batch = images.numpy()
#     results = clf_localize_model.predict(test_batch)
#     top_k = 3
#     top_k_probs, top_k_labels = decode_predictions(results[0], index_to_label, top=top_k)
#     plt.figure(figsize=(3,3))
#     for i in range(BATCH_SIZE):
#         print(f"Image #{i+1}")
#         img = test_batch[i]
#         img.resize(224,224)
#         plt.imshow(img)
#         plt.show()
#         for j in range(top_k):
#             print("    {:12s} {:.2f}%".format(top_k_labels[i][j], top_k_probs[i][j] * 100))
#         print()
# =============================================================================
# =============================================================================
# model.evaluate(test_generator, steps=steps)
# =============================================================================
