import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random


def page_visualizer_body():
    st.write('---')

    st.header('**Image study**')

    st.info('''Business requirement 1: Help clients understand how the model interprets
images and provide insights into distinctive features.''')

    version = 'v1'

    # Load class labels from the pickle file
   # with open('class_labels.pkl', 'rb') as f:
    #    class_labels = pickle.load(f)

    if st.checkbox('Display the mean and standard deviation from the average image study'):
    # Load each class's average variability image
        avg_airplane = plt.imread(f"outputs/{version}/avg_var_airplane.png")
        avg_auto = plt.imread(f"outputs/{version}/avg_var_automobile.png")
        avg_bird = plt.imread(f"outputs/{version}/avg_var_bird.png")
        avg_cat = plt.imread(f"outputs/{version}/avg_var_cat.png")
        avg_deer = plt.imread(f"outputs/{version}/avg_var_deer.png")
        avg_dog = plt.imread(f"outputs/{version}/avg_var_dog.png")
        avg_frog = plt.imread(f"outputs/{version}/avg_var_frog.png")
        avg_horse = plt.imread(f"outputs/{version}/avg_var_horse.png")
        avg_ship = plt.imread(f"outputs/{version}/avg_var_ship.png")
        avg_truck = plt.imread(f"outputs/{version}/avg_var_truck.png")

        # Display a success message
        st.success('There is too much variation in the images to be able to see\
            any distinctive features of the different classes in this image study.')

        # Display images with captions for each class
        st.image(avg_airplane, caption='Airplane - Average and Variability', use_column_width=True)
        st.image(avg_auto, caption='Automobile - Average and Variability', use_column_width=True)
        st.image(avg_bird, caption='Bird - Average and Variability', use_column_width=True)
        st.image(avg_cat, caption='Cat - Average and Variability', use_column_width=True)
        st.image(avg_deer, caption='Deer - Average and Variability', use_column_width=True)
        st.image(avg_dog, caption='Dog - Average and Variability', use_column_width=True)
        st.image(avg_frog, caption='Frog - Average and Variability', use_column_width=True)
        st.image(avg_horse, caption='Horse - Average and Variability', use_column_width=True)
        st.image(avg_ship, caption='Ship - Average and Variability', use_column_width=True)
        st.image(avg_truck, caption='Truck - Average and Variability', use_column_width=True)

    
    if st.checkbox('Show the average differences between similar classes'):
        # Predefined list of class comparisons
        diff_airplane_bird = plt.imread("outputs/v1/avg_diff_airplane_bird.png")
        diff_deer_horse = plt.imread("outputs/v1/avg_diff_deer_horse.png")
        diff_truck_automobile = plt.imread("outputs/v1/avg_diff_truck_automobile.png")

        st.warning('Images in the dataset are too similar\
            to see any clear average differences in this image study.')

        st.image(diff_airplane_bird, caption='Airplane vs Bird - Average Differences', use_column_width=True)
        st.image(diff_deer_horse, caption='Deer vs Horse - Average Differences', use_column_width=True)
        st.image(diff_truck_automobile, caption='Truck vs Automobile - Average Differences', use_column_width=True)


    

    if st.checkbox('Display a montage of images from the dataset'):
        st.error('To see the montage, select labels and click on the "Create Montage" button, then wait for it to load.')

        sample_image_dir = 'outputs/sample_images'
        labels = os.listdir(sample_image_dir)

        labels_to_display = st.multiselect(
            label='Select labels',
            options=labels,
            default=[labels[0]]  # Default to the first label
        )

        if st.button('Create Montage'):
            if labels_to_display:
                image_montage(sample_image_dir, labels_to_display, nrows=3, figsize=(20, len(labels_to_display) * 5))
            else:
                st.warning('Please select at least one label to create the montage.')

def image_montage(image_dir, labels, nrows, figsize):
    ncols = len(labels)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    if ncols == 1:
        axes = axes.reshape(nrows, 1)  # Ensure axes is always 2D
    
    for col, label in enumerate(labels):
        images_list = os.listdir(os.path.join(image_dir, label))
        img_idx = random.sample(images_list, min(3, len(images_list)))  # Ensure at least 3 images, if available

        for row, img_file in enumerate(img_idx):
            img = imread(os.path.join(image_dir, label, img_file))
            img_shape = img.shape
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{label}\n{img_shape[1]}px x {img_shape[0]}px")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    # Hide empty subplots if any
    for col in range(len(labels), axes.shape[1]):
        for row in range(nrows):
            axes[row, col].axis('off')

    plt.tight_layout()
    st.pyplot(fig=fig)

page_visualizer_body()