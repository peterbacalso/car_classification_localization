import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb

from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.backend import clear_session

from data.data_loader import DataLoader
from models.custom_cnn import CNN
from models.three_cnn_layer import CNN_3
from models.transfer_learning import TL
from callbacks.stochastic_weighted_avg import SWA

clear_session() # Clear models from previous sessions


def load_data(output="label_bbox", batch_size=32, channels=1, 
              apply_tl_preprocess=False, model_type="resnet", seed=23):
    
    data = DataLoader('./data/cars_train', 
                  './data/cars_test', 
                  './data/devkit', 
                  batch_size=batch_size)
    n_classes = len(data.df_train['label'].unique())
    labels = data.labels
    print(f'{n_classes} CLASSES, Random Chance: {1/n_classes}')
    
    train_gen = data.get_pipeline(type='train',
                                  output=output,
                                  apply_aug=True,
                                  channels=channels,
                                  apply_tl_preprocess=apply_tl_preprocess,
                                  model_type=model_type,
                                  seed=seed)
    steps_per_epoch = np.ceil(len(data.df_train)/data.batch_size)
    
    valid_gen = data.get_pipeline(type='validation',
                                  output=output,
                                  channels=channels,
                                  apply_aug=False,
                                  apply_tl_preprocess=apply_tl_preprocess,
                                  model_type=model_type,
                                  seed=seed)
    validation_steps = np.ceil(len(data.df_valid)/data.batch_size)
    
    return ( labels, n_classes, train_gen, 
            steps_per_epoch, valid_gen, validation_steps)
        

def get_callbacks(wandb_labels,
                  wandb_gen,
                  early_stopping_patience, 
                  reduce_lr_on_plateau_factor,
                  reduce_lr_on_plateau_patience,
                  reduce_lr_on_plateau_min_lr):   
# =============================================================================
#     # Tensorboard
#     root_logdir = os.path.join(os.curdir, 'logs')
#     run_id = time.strftime(f"run_%Y_%m_%d-%H_%M_%S")
#     run_logdir = os.path.join(root_logdir, run_id)
#     tensorboard = TensorBoard(run_logdir)
# =============================================================================
    wandb_cb = WandbCallback(
            data_type="image",
            output_type="label",
            labels=wandb_labels,
            generator=wandb_gen
            )
    
    swa = SWA('./logs/swa.h5', 3)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, 
                                   restore_best_weights=True)
    
     # Model Checkpoints
    checkpoint = ModelCheckpoint(
        filepath=f'checkpoints/' + \
        'epoch.{epoch:02d}_val_loss.{val_loss:.6f}.h5', 
        verbose=1, save_best_only=True)
    
    #Reduce LR on Plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=reduce_lr_on_plateau_factor,
                                  patience=reduce_lr_on_plateau_patience, 
                                  min_lr=reduce_lr_on_plateau_min_lr, 
                                  verbose=1)
    
    return [wandb_cb, swa, early_stopping, checkpoint, reduce_lr]
    #return [wandb_cb, early_stopping, checkpoint, reduce_lr]
        
    


if __name__=="__main__":
    
    
    
    wandb.init(config={
            "epochs": 150,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "dropout_chance": 0.7,
            "reduce_lr_on_plateau_min_lr": 1e-6,
            "reduce_lr_on_plateau_factor": .33333,
            "reduce_lr_on_plateau_patience": 5,
            "l2_reg": 3e-3,
            "num_frozen_layers": 62, # 34, 62, 100
            "batch_size": 32,
            "channels": 3,
            "random_seed": 23,
            "early_stopping_patience": 10,
            "use_focal_loss": True,
            "transfer_learning": True,
            "transfer_learning_model": "efn_b3" #resnet | mobilenet
            })
    config = wandb.config
    
    labels, n_classes, train_gen, steps_per_epoch, \
    valid_gen, validation_steps = \
    load_data(batch_size=config.batch_size, 
              channels=config.channels,
              apply_tl_preprocess=config.transfer_learning,
              model_type=config.transfer_learning_model,
              seed=config.random_seed)
    
    callbacks = get_callbacks(
        wandb_labels=labels.labels,
        wandb_gen=valid_gen,
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_on_plateau_factor=config.reduce_lr_on_plateau_factor,
        reduce_lr_on_plateau_patience=config.reduce_lr_on_plateau_patience,
        reduce_lr_on_plateau_min_lr=config.reduce_lr_on_plateau_min_lr
        )
    
    config.num_classes = n_classes
    
    model = TL(n_classes, 
               optimizer_type=config.optimizer,
               lr=config.learning_rate, 
               reg=config.l2_reg,
               use_focal_loss=config.use_focal_loss,
               num_frozen_layers=config.num_frozen_layers,
               dropout_chance=config.dropout_chance,
               model_type=config.transfer_learning_model,
               channels=config.channels)

    history_clf = model.fit(
        train_gen,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1)
    
    #print(history_clf.history)
    validation_labels_loss = np.amin(history_clf.history['val_labels_loss']) 
    print('Best validation labels loss:', validation_labels_loss)
    labels_loss = np.amin(history_clf.history['labels_loss']) 
    print('Best labels loss:', labels_loss)

    del model
    clear_session()



