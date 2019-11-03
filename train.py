import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.backend import clear_session

from data.data_loader import DataLoader
from models.custom_cnn import CNN
from models.three_cnn_layer import CNN_3

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

train_gen_clf = data.get_pipeline(type='train',
                                  output='label',
                                  channels=3,
                                  seed=SEED)
train_gen_localize = data.get_pipeline(type='train', 
                                       output='bbox',
                                       channels=3,
                                       seed=SEED)
train_gen_clf_localize = data.get_pipeline(type='train',
                                           channels=3,
                                           apply_aug=False, 
                                           seed=SEED)
steps_per_epoch=tf.math.ceil(len(data.df_train)/data.batch_size)
tf.cast(steps_per_epoch, tf.int16).numpy()

valid_gen_clf = data.get_pipeline(type='validation',
                                  output='label',
                                  channels=3,
                                  apply_aug=False,
                                  seed=SEED)
valid_gen_localize = data.get_pipeline(type='validation',
                                       output='bbox',
                                       channels=3,
                                       apply_aug=False,
                                       seed=SEED)
valid_gen_clf_localize = data.get_pipeline(type='validation',
                                           channels=3,
                                           apply_aug=False,
                                           seed=SEED)
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
optimizer = SGD(lr=1e-2, momentum=0.9, decay=0.01)

# Models

# Classification Only
clf_model = CNN_3(n_classes, channels=3, output="label")
# clf_model = CNN(n_classes, channels=3, output="label")
clf_model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

history_clf = clf_model.fit(
        train_gen_clf,
        epochs = 20,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen_clf,
        validation_steps=validation_steps,
        verbose=1)

# =============================================================================
# # Bounding Box Only
# localize_model = CNN(n_classes, output="bbox")
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
# clf_localize_model = CNN(n_classes)
# clf_localize_model.compile(loss=["categorical_crossentropy", "msle"],
#               loss_weights=[.8,.2],
#               optimizer=optimizer,
#               metrics=["accuracy"])
# 
# history_clf_localize = clf_localize_model.fit(
#         train_gen_clf_localize,
#         epochs = 200,
#         steps_per_epoch=steps_per_epoch,
#         validation_data=valid_gen_clf_localize,
#         validation_steps=validation_steps,
#         callbacks=[tensorboard_cb],
#         verbose=1)
# =============================================================================



#  TESTING

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
