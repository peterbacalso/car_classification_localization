# Stanford Cars Dataset Image Classification and Localization

# Input Pipeline

For preprocessing, making the data zero-centered would be computationally expensive to load all images and calculate the mean. Therefore we will just scale the pixels to be in range [-0.5,0.5] and use batch normalization between layers.

To augment the image dataset, the library [imgaug](https://imgaug.readthedocs.io/en/latest/) provides a wide range of augmentation techniques of which the following are used:

* Flip Left and Right
* Crop 0-10% off the image
* Apply GaussianBlur to 50% of the images
* Adjust the contrast
* Adjust the brightness
* Scale/zoom and translate/move images
* Add per pixel noise on a randomly chosen channel

The data has a class imbalance so one way to handle this is through oversampling which is creating copies of our minority classes to match the majority ones. Fortunately we do not have to do this explicitly since we can get it for free by modifying the way we output images from the tf.data generator. Oversampling is achieved by splitting the data by their class labels and sampling from them uniformly. This preserves the underlying distribution of the minority classes but evens out the dataset without needing to collect more data!

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/data_pipeline.jpg)

The following is the imbalanced distribution of classes

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/class_imbalance.png)

The following is an example of the distribution after oversampling. The generator was tested for 300 iterations @ 32 batch size each

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/oversampled.png)

## Sanity Checks

This step was done to help monitor training and adjust hyperparameters to get good learning results.

1. When using softmax, the value of the loss when the weights are small and no regularization is used can be approximated by -ln(1/C) = ln(C) where C is the number of classes.

The entire dataset has 196 classes which means the softmax loss should be approximately ln(196)=5.278. After running one epoch on a neural net with 1 hidden layer, the loss did in fact match.
```
217/217 [==============================] - 17s 80ms/step - loss: 5.2780 - accuracy: 0.0049 - val_loss: 5.2947 - val_accuracy: 0.0032
T
he same process was repeated for a subset of the dataset using 2 labels. The loss should be ln(2)=0.693.
3/3 [==============================] - 1s 233ms/step - loss: 0.6933 - accuracy: 0.5625 - val_loss: 0.5985 - val_accuracy: 0.6875
2. A
dding regularization should make the loss go up. The following test adds l2 regularization of magnitude 1e2 which made the loss jump from 0.693 to 2.9.
3/3
 [==============================] - 1s 322ms/step - loss: 2.9040 - accuracy: 0.4375 - val_loss: 2.9195 - val_accuracy: 0.6875
```
## Mode
l Architecture

## Training Process

1. Train on a small subset of data (eg. 20 samples) which should be easy to overfit and get a high training accuracy. The subset size used for this step was 73 images over 2 classes and ran for 200 epochs that resulted in 100% classifier accuracy.
```
Epoch 
197/200
3/3 [==============================] - 3s 1s/step - loss: 0.0555 - classifier_loss: 0.0281 - localizer_loss: 0.1651 - classifier_accuracy: 1.0000 - localizer_accuracy: 0.4271 - val_loss: 15.2211 - val_classifier_loss: 11.0812 - val_localizer_loss: 31.7805 - val_classifier_accuracy: 0.3125 - val_localizer_accuracy: 0.8125
Epoch 198/200
3/3 [==============================] - 3s 1s/step - loss: 0.0425 - classifier_loss: 0.0163 - localizer_loss: 0.1473 - classifier_accuracy: 1.0000 - localizer_accuracy: 0.4479 - val_loss: 15.2499 - val_classifier_loss: 11.0812 - val_localizer_loss: 31.9246 - val_classifier_accuracy: 0.3125 - val_localizer_accuracy: 0.9062
Epoch 199/200
3/3 [==============================] - 3s 1s/step - loss: 0.0487 - classifier_loss: 0.0264 - localizer_loss: 0.1382 - classifier_accuracy: 1.0000 - localizer_accuracy: 0.3542 - val_loss: 15.2735 - val_classifier_loss: 11.0812 - val_localizer_loss: 32.0426 - val_classifier_accuracy: 0.3125 - val_localizer_accuracy: 0.8125
Epoch 200/200
3/3 [==============================] - 3s 1s/step - loss: 0.0572 - classifier_loss: 0.0329 - localizer_loss: 0.1546 - classifier_accuracy: 1.0000 - localizer_accuracy: 0.3958 - val_loss: 15.3056 - val_classifier_loss: 11.0812 - val_localizer_loss: 32.2035 - val_classifier_accuracy: 0.3125 - val_localizer_accuracy: 0.8125
```
2. Train using full dataset, start with small regularization and find the learning rate that makes the loss go down. The model is able to overfit with train accuracy of 1 implying that it has enough capacity to learn the image features.


## Dataset Citation

Krause, Jonathan, et al. "3d object representations for fine-grained categorization." Proceedings of the IEEE International Conference on Computer Vision Workshops. 2013.
