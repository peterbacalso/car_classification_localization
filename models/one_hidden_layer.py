import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization,
    Activation
)
from tensorflow.keras.regularizers import l2

weight_initer = tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.0001)
W = tf.compat.v1.get_variable(name="Weight", dtype=tf.float32, shape=[50176, 64], initializer=weight_initer)

W = tf.constant_initializer(W.numpy())

def Simple_NN(n_classes, output="label_bbox"):
    input = Input(shape=(224,224,1))
    
    flatten = Flatten()(input)
    
    dense = Dense(units=64, 
                  kernel_initializer=W, 
                  bias_initializer="zeros",
                  #kernel_regularizer=l2(1e2),
                  activation="relu")(flatten)
    
    class_output = Dense(n_classes, 
                         activation="softmax", 
                         name="classifier")(dense)
    bbox_output = Dense(units=4, name="localizer")(dense)

    model=None
    if output=="bbox":
        model = Model(inputs=input, outputs=bbox_output)
    elif output=="label":
        model = Model(inputs=input, outputs=class_output)
    else:
        model = Model(inputs=input, outputs=[class_output,bbox_output])
    return model

