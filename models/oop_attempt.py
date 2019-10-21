import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from functools import partial

# Model built using functional API
    
# =============================================================================
# DefaultConv2D = partial(Conv2D,
#                         kernel_size=3,
#                         kernel_initializer="he_normal",
#                         kernel_regularizer=l2(1e-1),
#                         activation="relu",
#                         padding="SAME")
# 
# =============================================================================
# =============================================================================
# optimizer = SGD(lr=0.01, momentum=0.9, decay=0.01)
# 
# input = Input(shape=(224,224,1))
# hidden_1 = DefaultConv2D(filters=128, kernel_size=7, strides=2)(input)
# 
# class_output = Dense(n_classes, activation="softmax")(hidden_1)
# bbox_output = Dense(4)(hidden_1)
# 
# model = Model(inputs=input, outputs=[class_output,bbox_output])
# model.compile(loss=["categorical_crossentropy", "mse"],
#               loss_weights=[0.8,0.2],
#               optimizer=optimizer,
#               metrics=["accuracy"])
# =============================================================================


# =============================================================================
# optimizer = SGD(lr=0.01, momentum=0.9, decay=0.01)
# 
# input = Input(shape=(224,224,1))
# hidden_1 = DefaultConv2D(filters=128, kernel_size=7, strides=2)(input)
# max_pool = MaxPooling2D(pool_size=2)(hidden_1)
# flatten = Flatten()(max_pool)
# class_output = Dense(n_classes, activation="softmax")(flatten)
# bbox_output = Dense(4)(flatten)
# 
# model = Model(inputs=input, outputs=[class_output,bbox_output])
# model.compile(loss=["categorical_crossentropy", "mse"],
#               loss_weights=[0.8,0.2],
#               optimizer=optimizer,
#               metrics=["accuracy"])
# =============================================================================
            


# Model built using OOP apprach

# =============================================================================
# class SimpleConvNet(Model):
#     def __init__(self, 
#                  n_classes,
#                  filters=128, 
#                  activation='relu', 
#                  shape=(224,224,1), **kwargs):
#         super().__init__(**kwargs)
#         self.input = Input(shape=shape)
#         self.default_conv2D = Conv2D(filters=filters,
#                                     kernel_size=3,
#                                     kernel_initializer="he_normal",
#                                     kernel_regularizer=l2(1e-1),
#                                     activation=activation,
#                                     padding="SAME")
#         self.max_pool = MaxPooling2D(pool_size=2)
#         self.flatten = Flatten()
#         self.class_output = Dense(n_classes, activation="softmax")
#         self.bbox_output = Dense(4)
#         
#     def build()
#         
#     def call(self, inputs):
#         input = self.input(inputs)
#         hidden_1 = self.default_conv2D(input)
#         max_pool = self.max_pool(hidden_1)
#         flatten = self.flatten(max_pool)
#         class_output = self.class_output(flatten)
#         bbox_output = self.bbox_output(flatten)
#         return class_output, bbox_output
# =============================================================================
        
        