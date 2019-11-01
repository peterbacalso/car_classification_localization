import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.backend import clear_session

from data.data_loader import DataLoader
from models.one_hidden_layer import Simple_NN

clear_session() # Clear models from previous sessions

# Constants
BATCH_SIZE=32
SEED=23

# Initialize Pipeline
data = DataLoader('./data/cars_train', 
                  './data/cars_test', 
                  './data/devkit', 
                  batch_size=BATCH_SIZE)
n_classes = len(data.df_train['label'].unique())

train_gen_clf = data.get_pipeline(type='train', output='label', seed=SEED)
train_gen_localize = data.get_pipeline(type='train', output='bbox', seed=SEED)
train_gen_clf_localize = data.get_pipeline(type='train', seed=SEED)
steps_per_epoch=tf.math.ceil(len(data.df_train)/data.batch_size)
tf.cast(steps_per_epoch, tf.int16).numpy()

valid_gen_clf = data.get_pipeline(type='validation', output='label', 
                                  apply_aug=False, seed=SEED)
valid_gen_localize = data.get_pipeline(type='validation', output='bbox', 
                                       apply_aug=False, seed=SEED)
valid_gen_clf_localize = data.get_pipeline(type='validation', 
                                           apply_aug=False, seed=SEED)
validation_steps = tf.math.ceil(len(data.df_valid)/data.batch_size)
validation_steps = tf.cast(validation_steps, tf.int16).numpy()

# Callbacks
root_logdir = os.path.join(os.curdir, 'logs')
def get_run_logdir():
    run_id = time.strftime(f"run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = TensorBoard(run_logdir)

# Optimizer
optimizer = SGD(lr=.01, momentum=0.9, decay=0.01)

# Models

# Classification Only
clf_model = Simple_NN(n_classes, output="label")
clf_model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

history_clf = clf_model.fit(
        train_gen_clf,
        epochs = 1,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen_clf,
        validation_steps=validation_steps,
        verbose=1)

# =============================================================================
# # Bounding Box Only
# localize_model = Simple_NN(n_classes, output="bbox")
# localize_model.compile(loss="mse",
#               optimizer=optimizer,
#               metrics=["accuracy"])
# 
# history_localize = localize_model.fit(
#         train_gen_localize,
#         epochs = 1,
#         steps_per_epoch=steps_per_epoch,
#         validation_data=valid_gen_localize,
#         validation_steps=validation_steps,
#         verbose=1)
# =============================================================================

# =============================================================================
# # Classification and Bounding Box
# clf_localize_model = Simple_NN(n_classes)
# clf_localize_model.compile(loss=["categorical_crossentropy", "msle"],
#               loss_weights=[.8,.2],
#               optimizer=optimizer,
#               metrics=["accuracy"])
# 
# history_clf_localize = clf_localize_model.fit(
#         train_gen_clf_localize,
#         epochs = 20,
#         steps_per_epoch=steps_per_epoch,
#         validation_data=valid_gen_clf_localize,
#         validation_steps=validation_steps,
#         callbacks=[tensorboard_cb],
#         verbose=1)
# =============================================================================



