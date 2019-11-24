import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy as sp
import tensorflow as tf
import efficientnet.tfkeras as efn
from efficientnet.tfkeras import preprocess_input as preproc_efn
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import sys, os; 
sys.path.insert(0, os.path.abspath('..'));
from loss_functions.focal_loss import focal_loss

#@st.cache(allow_output_mutation=True)
def load_cnn():
    model = load_model('../old_checkpoints/efn50_62_frozen/epoch.130_val_loss.0.337101.h5', 
                       custom_objects={'focal_loss_fixed': focal_loss(alpha=1)})
    return model

@st.cache
def load_p_r():
    precision_recall = pd.read_csv('../data_tables/precision_recall.csv')
    return precision_recall

@st.cache
def load_train():
    df_train = pd.read_csv('../data_tables/train.csv')
    return df_train

@st.cache
def load_test():
    df_test = pd.read_csv('../data_tables/test.csv')
    return df_test

@st.cache
def load_labels():
    df_labels = pd.read_csv('../data_tables/labels.csv')
    return df_labels


def plot_activation(model, img, labels):
   
    pred = model.predict(img[np.newaxis,:,:,:], steps=1)[0]
    pred_class = np.argmax(pred)

    weights = model.layers[-2].get_weights()[0]
    class_weights = weights[:, pred_class]

    intermediate = Model(model.input,
                         model.get_layer("top_conv").output)
    conv_output = intermediate.predict(img[np.newaxis,:,:,:], steps=1)
    conv_output = np.squeeze(conv_output)

    h = int(img.shape[0]/conv_output.shape[0])
    w = int(img.shape[1]/conv_output.shape[1])

    act_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)

    out = np.dot(act_maps.reshape((img.shape[0]*img.shape[1],-1)), 
                 class_weights).reshape(img.shape[0],img.shape[1])

    img = img.astype('float32').reshape(img.shape[0], img.shape[1],3)
    
    #return img, out, labels[pred_class]
    plt.imshow(img)
    plt.imshow(out, cmap='jet', alpha=0.35)
    plt.title(labels[pred_class])
    st.pyplot()

def main():    
    df_train = load_train()
    df_test = load_test()
    df_labels = load_labels()
    precision_recall = load_p_r()
    
    '''
    # Car Classification and Localization
    '''
    option = st.sidebar.selectbox(
    'Select dataset',
     ['Train', 'Test'])
    
    if option == 'Train':
        labels = st.multiselect('Select Car Types', 
                                np.sort(df_train['labels'].unique()))
        labels_filter = df_train['labels'].isin(labels)
        df_train = df_train[labels_filter] if len(labels) else df_train
        '''
        # Training Data
        '''
        df_train
        if len(labels) and st.sidebar.checkbox("View Images"):
            images = []
            for i in range(len(df_train[labels_filter]['fname'])):
                filepath = df_train[labels_filter]['fname'].values[i]
                image = Image.open(f'../data/{filepath}')
                images.append(image)
            image_width = st.slider("Image Width", 100, 300, 100)
            st.image(images, 
                     caption=df_train[labels_filter]['labels'].values, 
                     width=image_width)
        if st.sidebar.checkbox("View Training Data Distribution"):
            hist_values = np.histogram(
                df_train['label'], bins=len(df_train['label'].unique()))
            '''Class Distribution'''
            st.bar_chart(hist_values[0])
    elif option == 'Test':
        labels = st.multiselect('Select Car Types', 
                                np.sort(df_test['labels'].unique()))
        labels_filter = df_test['labels'].isin(labels)
        df_test = df_test[labels_filter] if len(labels) else df_test
        '''
        # Testing Data
        '''
        df_test
        if len(labels) and st.sidebar.checkbox("View Images"):
            images = []
            for i in range(len(df_test[labels_filter]['fname'])):
                filepath = df_test[labels_filter]['fname'].values[i]
                image = Image.open(f'../data/{filepath}')
                images.append(image)
            image_width = st.slider("Image Width", 100, 300, 100)
            st.image(images, 
                     caption=df_test[labels_filter]['labels'].values, 
                     width=image_width)
        if st.sidebar.checkbox("View Testing Data Distribution"):
            hist_values = np.histogram(
                df_test['label'], bins=len(df_test['label'].unique()))
            '''Class Distribution'''
            st.bar_chart(hist_values[0])
    
        '''
        # Test Results
        Precision and Recall
        '''
        p_r_filter = precision_recall['labels'].isin(labels)
        precision_recall = precision_recall[p_r_filter] if len(labels) \
        else precision_recall
        r_tresh = st.slider("Recall Threshold", 0., 1., 1.)
        p_tresh = st.slider("Precision Threshold", 0., 1., 1.)
        precision_recall = precision_recall[precision_recall['Precision']<=p_tresh]
        precision_recall[precision_recall['Recall']<=r_tresh]
        
# =============================================================================
#         if len(labels) and st.sidebar.checkbox("View Saliency Map"):
#             model = load_cnn()
#             for i in range(len(df_test[labels_filter]['fname'])):
#                 filepath = df_test[labels_filter]['fname'].values[i]
#                 img = tf.keras.preprocessing\
#                 .image.load_img(f'../data/{filepath}', target_size=(224, 224))
#                 img = np.array(img)
#                 img = preproc_efn(img)
#                 plot_activation(model, img, df_labels['labels'])
# =============================================================================

if __name__ == "__main__":
    main()
