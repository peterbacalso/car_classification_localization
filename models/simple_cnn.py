import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPooling2D, 
    Flatten, Dropout, BatchNormalization,
    Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import clear_session
from functools import partial

from data.data_loader import DataLoader

clear_session()

BATCH_SIZE=16
SEED=23

# =============================================================================
# data = DataLoader('../data/cars_train', 
#                   '../data/cars_test', 
#                   '../data/devkit', 
#                   batch_size=BATCH_SIZE)
# n_classes = len(data.df_train['label'].unique())
# 
# train_gen_clf = data.get_pipeline(type='train', output='label', seed=SEED)
# train_gen_localize = data.get_pipeline(type='train', output='bbox', seed=SEED)
# train_gen_clf_localize = data.get_pipeline(type='train', seed=SEED)
# steps_per_epoch=tf.math.ceil(len(data.df_train)/data.batch_size)
# tf.cast(steps_per_epoch, tf.int16).numpy()
# 
# valid_gen_clf = data.get_pipeline(type='validation', output='label', seed=SEED)
# valid_gen_localize = data.get_pipeline(type='train', output='bbox', seed=SEED)
# valid_gen_clf_localize = data.get_pipeline(type='validation', seed=SEED)
# validation_steps = tf.math.ceil(len(data.df_valid)/data.batch_size)
# validation_steps = tf.cast(validation_steps, tf.int16).numpy()
# =============================================================================
        
optimizer = SGD(lr=1e-7, momentum=0.9, decay=0.01)

DefaultConv2D = partial(Conv2D,
                        kernel_size=3,
                        kernel_initializer="he_normal",
                        # kernel_regularizer=l2(.01),
                        # activation="relu",
                        padding="SAME")

# =============================================================================
# input = Input(shape=(224,224,1))
# conv_1 = DefaultConv2D(filters=64, kernel_size=7, strides=2)(input)
# max_pool_1 = MaxPooling2D(pool_size=2)(conv_1)
# flatten = Flatten()(max_pool_1)
# class_output = Dense(n_classes, activation="softmax")(flatten)
# bbox_output = Dense(4)(flatten)
# =============================================================================

input = Input(shape=(224,224,1))

conv_1 = DefaultConv2D(filters=64, kernel_size=7, strides=2)(input)
norm_1 = BatchNormalization()(conv_1)
relu_1 = Activation(activation="relu")(norm_1)
max_pool_1 = MaxPooling2D(pool_size=2)(relu_1)


conv_2 = DefaultConv2D(filters=128)(max_pool_1)
norm_2 = BatchNormalization()(conv_2)
relu_2 = Activation(activation="relu")(norm_2)
max_pool_2 = MaxPooling2D(pool_size=2)(relu_2)

conv_3 = DefaultConv2D(filters=256)(max_pool_2)
max_pool_3 = MaxPooling2D(pool_size=2)(conv_3)
#norm_3 = BatchNormalization()(max_pool_3)

flatten = Flatten()(max_pool_3)
dense_1 = Dense(units=128, activation="relu")(flatten)
drop_1 = Dropout(0.5)(dense_1)
dense_2 = Dense(units=64, activation="relu")(drop_1)
drop_2 = Dropout(0.5)(dense_2)

class_output = Dense(n_classes, activation="softmax")(drop_2)
bbox_output = Dense(4)(drop_2)


# =============================================================================
# clf_model = Model(inputs=input, outputs=class_output)
# clf_model.compile(loss="categorical_crossentropy",
#               optimizer=optimizer,
#               metrics=["accuracy"])
# 
# history_clf = clf_model.fit(
#         train_gen_clf,
#         epochs = 1,
#         steps_per_epoch=steps_per_epoch,
#         validation_data=valid_gen_clf,
#         validation_steps=validation_steps,
#         verbose=1)
# =============================================================================

# =============================================================================
# localize_model = Model(inputs=input, outputs=bbox_output)
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

clf_localize_model = Model(inputs=input, outputs=[class_output,bbox_output])
clf_localize_model.compile(loss=["categorical_crossentropy", "mse"],
              loss_weights=[.8,.2],
              optimizer=optimizer,
              metrics=["accuracy"])

history_clf_localize = clf_localize_model.fit(
        train_gen_clf_localize,
        epochs = 5,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen_clf_localize,
        validation_steps=validation_steps,
        verbose=1)

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
