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

data = DataLoader('../data/cars_train', 
                  '../data/cars_test', 
                  '../data/devkit', 
                  batch_size=BATCH_SIZE)
n_classes = len(data.df_train['label'].unique())

train_gen_clf = data.get_pipeline(type='train', output='label', seed=SEED)
train_gen_localize = data.get_pipeline(type='train', output='bbox', seed=SEED)
train_gen_clf_localize = data.get_pipeline(type='train', seed=SEED)
steps_per_epoch=tf.math.ceil(len(data.df_train)/data.batch_size)
tf.cast(steps_per_epoch, tf.int16).numpy()

valid_gen_clf = data.get_pipeline(type='validation', output='label', seed=SEED)
valid_gen_localize = data.get_pipeline(type='train', output='bbox', seed=SEED)
valid_gen_clf_localize = data.get_pipeline(type='validation', seed=SEED)
validation_steps = tf.math.ceil(len(data.df_valid)/data.batch_size)
validation_steps = tf.cast(validation_steps, tf.int16).numpy()
        
optimizer = SGD(lr=1e-1, momentum=0.9, decay=0.01)

DefaultConv2D = partial(Conv2D,
                        kernel_size=3,
                        kernel_initializer="he_normal",
                        # kernel_regularizer=l2(.01),
                        # activation="relu",
                        padding="SAME")

# =============================================================================
# # simple 1 hidden layer cnn
# input = Input(shape=(224,224,1))
# conv_1 = DefaultConv2D(filters=64, kernel_size=7, strides=2)(input)
# max_pool_1 = MaxPooling2D(pool_size=2)(conv_1)
# flatten = Flatten()(max_pool_1)
# class_output = Dense(n_classes, activation="softmax")(flatten)
# bbox_output = Dense(4)(flatten)
# =============================================================================

input = Input(shape=(224,224,1))

# =============================================================================
# conv_1 = DefaultConv2D(filters=64, kernel_size=7, strides=2)(input)
# norm_1 = BatchNormalization()(conv_1)
# relu_1 = Activation(activation="relu")(norm_1)
# =============================================================================

# 3 layers of 3x3 has same effective receptive field as 7x7 but less params and more activations

conv_1a = DefaultConv2D(filters=64, padding='same')(input)
norm_1a = BatchNormalization()(conv_1a)
relu_1a = Activation(activation="relu")(norm_1a)
conv_1b = DefaultConv2D(filters=64, padding='same')(relu_1a)
norm_1b = BatchNormalization()(conv_1b)
relu_1b = Activation(activation="relu")(norm_1b)
conv_1c = DefaultConv2D(filters=64, padding='same')(relu_1b)
norm_1c = BatchNormalization()(conv_1c)
relu_1c = Activation(activation="relu")(norm_1c)

max_pool_1 = MaxPooling2D(pool_size=2)(relu_1c)

conv_2a = DefaultConv2D(filters=128)(max_pool_1)
norm_2a = BatchNormalization()(conv_2a)
relu_2a = Activation(activation="relu")(norm_2a)
conv_2b = DefaultConv2D(filters=128)(relu_2a)
norm_2b = BatchNormalization()(conv_2b)
relu_2b = Activation(activation="relu")(norm_2b)

max_pool_2 = MaxPooling2D(pool_size=2)(relu_2b)

conv_3a = DefaultConv2D(filters=256)(max_pool_2)
norm_3a = BatchNormalization()(conv_3a)
relu_3a = Activation(activation="relu")(norm_3a)
conv_3b = DefaultConv2D(filters=256)(relu_3a)
norm_3b = BatchNormalization()(conv_3b)
relu_3b = Activation(activation="relu")(norm_3b)

max_pool_3 = MaxPooling2D(pool_size=2)(relu_3b)

conv_4a = DefaultConv2D(filters=256, kernel_size=1)(max_pool_3)
norm_4a = BatchNormalization()(conv_4a)
relu_4a = Activation(activation="relu")(norm_4a)
conv_4b = DefaultConv2D(filters=256)(relu_4a)
norm_4b = BatchNormalization()(conv_4b)
relu_4b = Activation(activation="relu")(norm_4b)

max_pool_4 = MaxPooling2D(pool_size=2)(relu_4b)

flatten = Flatten()(max_pool_3)

drop_1 = Dropout(0.5)(flatten)
dense_1 = Dense(units=128, activation="relu")(drop_1)
norm_4 = BatchNormalization()(dense_1)
relu_4 = Activation(activation="relu")(norm_4)

drop_2 = Dropout(0.5)(relu_4)
dense_2 = Dense(units=64, activation="relu")(drop_2)
norm_5 = BatchNormalization()(dense_2)
relu_5 = Activation(activation="relu")(norm_5)

drop_3 = Dropout(0.5)(relu_5)
class_output = Dense(n_classes, activation="softmax")(drop_3)
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
        epochs = 20,
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
