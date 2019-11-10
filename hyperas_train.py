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
    Activation, GlobalAvgPool2D
)
from layers.Residual import Residual
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
from models.hyperas_cnn import model

clear_session() # Clear models from previous sessions

def data():
    output="label_bbox"
    channels=3
    BATCH_SIZE=32
    SEED=23
    
    data = DataLoader('./data/cars_train', 
                  './data/cars_test', 
                  './data/devkit', 
                  batch_size=BATCH_SIZE)

    train_gen = data.get_pipeline(type='train',
                                  output=output,
                                  apply_aug=True,
                                  channels=channels,
                                  seed=SEED)

    valid_gen = data.get_pipeline(type='validation',
                                  output=output,
                                  channels=channels,
                                  apply_aug=True,
                                  seed=SEED)

    return train_gen, valid_gen
        


# =============================================================================
# def my_args():
#    args = argparse.Namespace()
#    args.ouput = 1
#    return args
# =============================================================================


if __name__=="__main__":
    
    
    train_generator, validation_generator = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())

    print("Evaluation of best performing model:")
    print('loss, labels_loss, bbox_loss, labels_acc, bbox_acc')
    print(best_model.evaluate_generator(validation_generator, steps=51))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)




