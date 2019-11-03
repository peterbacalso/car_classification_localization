from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPooling2D, 
    Flatten, Dropout, AveragePooling2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from functools import partial

DefaultConv2D = partial(Conv2D,
                        kernel_size=3,
                        kernel_initializer="he_normal",
                        #kernel_regularizer=l2(1e-6),
                        activation="relu",
                        padding="same")

def CNN_3(n_classes, lr=.1, channels=1, output="label_bbox"):
    input = Input(shape=(224,224,channels))
    
    conv = DefaultConv2D(filters=64, kernel_size=7)(input)    
    pool = MaxPooling2D(pool_size=2)(conv)
    
    conv = DefaultConv2D(filters=128)(pool)
    pool = MaxPooling2D(pool_size=2)(conv)
    
    conv = DefaultConv2D(filters=256)(pool)
    pool = MaxPooling2D(pool_size=2)(conv)
    
    flatten = Flatten()(pool)
    
    dense = Dense(units=64, activation="relu")(flatten)  
    drop = Dropout(0.5)(dense)
    
    class_output = Dense(n_classes, 
                         activation="softmax", 
                         name="classifier")(drop)
    bbox_output = Dense(units=4, name="localizer")(drop)

    # Optimizer
    optimizer = SGD(lr=lr, momentum=0.9, decay=0.01)
    
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
    return model

