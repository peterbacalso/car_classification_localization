import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l2
from functools import partial

from data.data_loader import DataLoader


data = DataLoader('../data/cars_train', '../data/cars_test', '../data/devkit')
n_classes = len(data.labels)

train_generator = data.get_pipeline(type='train')
steps_per_epoch=tf.math.ceil(len(data.df_train)/data.batch_size)
tf.cast(steps_per_epoch, tf.int16).numpy()

valid_generator = data.get_pipeline(type='validation')
validation_steps = tf.math.ceil(len(data.df_valid)/data.batch_size)
validation_steps = tf.cast(validation_steps, tf.int16).numpy()
        
optimizer = SGD(lr=0.01, momentum=0.9, decay=0.01)

DefaultConv2D = partial(Conv2D,
                        kernel_size=3,
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(1e-1),
                        activation="relu",
                        padding="SAME")

input = Input(shape=(224,224,1))
hidden_1 = DefaultConv2D(filters=128, kernel_size=7, strides=2)(input)
max_pool = MaxPooling2D(pool_size=2)(hidden_1)
flatten = Flatten()(max_pool)
class_output = Dense(n_classes, activation="softmax")(flatten)
bbox_output = Dense(4)(flatten)

model = Model(inputs=input, outputs=[class_output,bbox_output])
model.compile(loss=["categorical_crossentropy", "mse"],
              loss_weights=[0.8,0.2],
              optimizer=optimizer,
              metrics=["accuracy"])

history = model.fit(
        train_generator,
        epochs = 1,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        verbose=1)
