import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb

from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import (
        TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.backend import clear_session

from data.data_loader import DataLoader
from models.custom_cnn import CNN
from models.three_cnn_layer import CNN_3

clear_session() # Clear models from previous sessions

# CONSTANTS

BATCH_SIZE=32
SEED=23
CHANNELS=3

def load_data(output="label_bbox", channels=1):
    
    data = DataLoader('./data/cars_train', 
                  './data/cars_test', 
                  './data/devkit', 
                  batch_size=BATCH_SIZE)
    n_classes = len(data.df_train['label'].unique())
    print(f'{n_classes} CLASSES, Random Chance: {1/n_classes}')
    
    train_gen = data.get_pipeline(type='train',
                                  output=output,
                                  apply_aug=True,
                                  channels=channels,
                                  seed=SEED)
    steps_per_epoch = np.ceil(len(data.df_train)/data.batch_size)
    #steps_per_epoch = tf.cast(steps_per_epoch, tf.int16).numpy()
    
    valid_gen = data.get_pipeline(type='validation',
                                  output=output,
                                  channels=channels,
                                  apply_aug=False,
                                  seed=SEED)
    validation_steps = np.ceil(len(data.df_valid)/data.batch_size)
    #validation_steps = tf.cast(validation_steps, tf.int16).numpy()
    
    #return None, None, None, None, None
    return data.labels, n_classes, train_gen, steps_per_epoch, valid_gen, validation_steps
        

def get_callbacks():   
    # Tensorboard
    root_logdir = os.path.join(os.curdir, 'logs')
    run_id = time.strftime(f"run_%Y_%m_%d-%H_%M_%S")
    run_logdir = os.path.join(root_logdir, run_id)
    tensorboard = TensorBoard(run_logdir)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    
     # Model Checkpoints
    checkpoint = ModelCheckpoint(
        filepath=f'checkpoints/' + \
        'epoch.{epoch:02d}_val_loss.{val_loss:.6f}.h5', 
        verbose=1, save_best_only=True)
    
    #Reduce LR on Plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.3,
                                  patience=5, 
                                  min_lr=0.0001, 
                                  verbose=1)
    
    return []
    #return [early_stopping, reduce_lr]
    #return [tensorboard, early_stopping, checkpoint, reduce_lr]
        
    


if __name__=="__main__":
    
    labels, n_classes, train_gen, steps_per_epoch, \
    valid_gen, validation_steps = load_data(channels=CHANNELS)
    
    #wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
    
    run = wandb.init(project="test_wandb")
    config = run.config
    config.epochs = 1
    callbacks = get_callbacks()
    callbacks.append(WandbCallback(labels=labels))
    
    lr=1e-3
    reg=1e-5
    
    # Classification Only
    # model = CNN_3(n_classes, channels=CHANNELS, output="label")
    # model = CNN(n_classes, lr=lr, reg=reg, channels=CHANNELS, output="label")
    # Bounding Box Only
    # model = CNN(n_classes, lr=lr, reg=reg, channels=CHANNELS, output="bbox")
    # Classification and Bounding Box
    
    #for lr in [1e-3, 3e-3, 1e-2]:
    print(f'Learning Rate {lr}, L2 Reg: {reg}')
    
    model = CNN(n_classes, lr=lr, reg=reg, channels=CHANNELS)

    history_clf = model.fit(
        train_gen,
        epochs=config.epochs,
        #epochs=1,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1)
    
    print(history_clf.history)
    validation_labels_loss = np.amin(history_clf.history['val_labels_loss']) 
    print('Best validation labels loss:', validation_labels_loss)
    labels_loss = np.amin(history_clf.history['labels_loss']) 
    print('Best labels loss:', labels_loss)

    del model
    clear_session()











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