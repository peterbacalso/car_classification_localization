from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPooling2D, 
    Flatten, Dropout, BatchNormalization,
    Activation
)
from tensorflow.keras.regularizers import l2
from functools import partial

DefaultConv2D = partial(Conv2D,
                        kernel_size=3,
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(1e-6),
                        # activation="relu",
                        padding="same")

def CNN(n_classes, channels=1, output="label_bbox"):
    input = Input(shape=(224,224,channels))
    
    conv_1a = DefaultConv2D(filters=64, padding='same')(input)
    #norm_1a = BatchNormalization()(conv_1a)
    relu_1a = Activation(activation="relu")(conv_1a)
    conv_1b = DefaultConv2D(filters=64, padding='same')(relu_1a)
    #norm_1b = BatchNormalization()(conv_1b)
    relu_1b = Activation(activation="relu")(conv_1b)
    conv_1c = DefaultConv2D(filters=64, padding='same')(relu_1b)
    #norm_1c = BatchNormalization()(conv_1c)
    relu_1c = Activation(activation="relu")(conv_1c)
    
    max_pool_1 = MaxPooling2D(pool_size=2)(relu_1c)
    
    conv_2a = DefaultConv2D(filters=128)(max_pool_1)
    #norm_2a = BatchNormalization()(conv_2a)
    relu_2a = Activation(activation="relu")(conv_2a)
    conv_2b = DefaultConv2D(filters=128)(relu_2a)
    #norm_2b = BatchNormalization()(conv_2b)
    relu_2b = Activation(activation="relu")(conv_2b)
    
    max_pool_2 = MaxPooling2D(pool_size=2)(relu_2b)
    
    conv_3a = DefaultConv2D(filters=256)(max_pool_2)
    #norm_3a = BatchNormalization()(conv_3a)
    relu_3a = Activation(activation="relu")(conv_3a)
    conv_3b = DefaultConv2D(filters=256)(relu_3a)
    #norm_3b = BatchNormalization()(conv_3b)
    relu_3b = Activation(activation="relu")(conv_3b)
    
    max_pool_3 = MaxPooling2D(pool_size=2)(relu_3b)
    
    conv_4a = DefaultConv2D(filters=512)(max_pool_3) #1x1 filter
    #norm_4a = BatchNormalization()(conv_4a)
    relu_4a = Activation(activation="relu")(conv_4a)
    conv_4b = DefaultConv2D(filters=512)(relu_4a)
    #norm_4b = BatchNormalization()(conv_4b)
    relu_4b = Activation(activation="relu")(conv_4b)
    
    max_pool_4 = MaxPooling2D(pool_size=2, strides=2)(relu_4b)
    
    conv_5a = DefaultConv2D(filters=512)(max_pool_4)
    #norm_5a = BatchNormalization()(conv_5a)
    relu_5a = Activation(activation="relu")(conv_5a)
    conv_5b = DefaultConv2D(filters=512)(relu_5a)
    #norm_5b = BatchNormalization()(conv_5b)
    relu_5b = Activation(activation="relu")(conv_5b)
    
    max_pool_5 = MaxPooling2D(pool_size=2)(relu_5b)
    
    flatten = Flatten()(max_pool_5)

    drop_1 = Dropout(0.5)(flatten)
    dense_1 = Dense(units=256)(drop_1)
    #norm_4 = BatchNormalization()(dense_1)
    relu_4 = Activation(activation="relu")(dense_1)
    
    drop_2 = Dropout(0.5)(relu_4)
    dense_2 = Dense(units=128)(drop_2)
    #norm_5 = BatchNormalization()(dense_2)
    relu_5 = Activation(activation="relu")(dense_2)
    
    drop_3 = Dropout(0.5)(relu_5)
    class_output = Dense(n_classes, 
                         activation="softmax", 
                         name="classifier")(drop_3)
    bbox_output = Dense(units=4, name="localizer")(drop_3)

    model=None
    if output=="bbox":
        model = Model(inputs=input, outputs=bbox_output)
    elif output=="label":
        model = Model(inputs=input, outputs=class_output)
    else:
        model = Model(inputs=input, outputs=[class_output,bbox_output])
    return model

