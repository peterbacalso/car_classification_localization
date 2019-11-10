from functools import partial
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPooling2D, 
    Flatten, Dropout, BatchNormalization,
    Activation, GlobalAvgPool2D
)
import sys, os; 
sys.path.insert(0, os.path.abspath('..'));

from layers.Residual import Residual


def CNN(n_classes, lr=.001, reg=1e-6, channels=1, output="label_bbox"):
    
    DefaultConv2D = partial(Conv2D,
                            kernel_size=3,
                            kernel_initializer="he_normal",
                            kernel_regularizer=l2(reg),
                            # activation="relu",
                            padding="same")
    
    input = Input(shape=(224,224,channels))
    
# =============================================================================
#     conv = DefaultConv2D(filters=64)(input)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)    
#     conv = DefaultConv2D(filters=64)(relu)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)
#     conv = DefaultConv2D(filters=64)(relu)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)
#     
#     x = MaxPooling2D(pool_size=2)(relu)
#     
#     conv = DefaultConv2D(filters=128)(x)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)
#     conv = DefaultConv2D(filters=128)(relu)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)
#     
#     x = MaxPooling2D(pool_size=2)(relu)
# =============================================================================
    
    
# =============================================================================
#     conv = DefaultConv2D(filters=64, kernel_size=5, strides=2)(input)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)    
#     
#     x = MaxPooling2D(pool_size=2)(relu)
#     
#     conv = DefaultConv2D(filters=128)(relu)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)
#     conv = DefaultConv2D(filters=128)(relu)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)
#     
#     x = MaxPooling2D(pool_size=2)(relu)
#     
#     conv = DefaultConv2D(filters=256)(x)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)
#     conv = DefaultConv2D(filters=256)(relu)
#     norm = BatchNormalization()(conv)
#     relu = Activation(activation="relu")(norm)
#     
#     x = MaxPooling2D(pool_size=2)(relu)
# =============================================================================
    
    conv = DefaultConv2D(filters=64, strides=2)(input)
    norm = BatchNormalization()(conv)
    relu = Activation(activation="relu")(norm)
    conv = DefaultConv2D(filters=64)(relu)
    norm = BatchNormalization()(conv)
    relu = Activation(activation="relu")(norm)
    
    x = MaxPooling2D(pool_size=2)(relu)

    for filters in [64] * 3:
        x = Residual(filters)(x)
    
    x = Residual(128, strides=2)(x)
    for filters in [128] * 2:
        x = Residual(filters)(x)
    
    x = Residual(256, strides=2)(x)
    for filters in [256] * 2:
        x = Residual(filters)(x)
        
    x = Residual(512, strides=2)(x)
    for filters in [512] * 2:
        x = Residual(filters)(x)
        
    x = GlobalAvgPool2D()(x)

    #x = Flatten()(x)
    
    drop = Dropout(0.2)(x)
    dense = Dense(units=512)(drop)
    norm = BatchNormalization()(dense)
    relu = Activation(activation="relu")(norm)
    
    drop = Dropout(0.2)(relu)
    dense = Dense(units=256)(drop)
    norm = BatchNormalization()(dense)
    relu = Activation(activation="relu")(norm)
    
    drop = Dropout(0.2)(relu)
    class_output = Dense(n_classes, 
                         activation="softmax", 
                         name="labels")(drop)
    bbox_output = Dense(units=4, name="bbox")(drop)
    
    # Optimizer
    #optimizer = SGD(lr=.1, momentum=0.9, decay=0.01)
    optimizer = Adam(lr=lr)

    model=None
    if output=="bbox":
        model = Model(inputs=input, outputs=bbox_output)
        model.compile(loss="msle",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        
    elif output=="label":
        model = Model(inputs=input, outputs=class_output)
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
    else:
        model = Model(inputs=input, outputs=[class_output,bbox_output])
        model.compile(loss=["categorical_crossentropy", "msle"],
                      loss_weights=[.8,.2],
                      optimizer=optimizer,
                      metrics=["accuracy"])
        
    print(model.summary())
    
    return model
