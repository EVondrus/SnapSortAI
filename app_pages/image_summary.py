import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import os
import pickle


def page_image_study_body():
    st.write('---')

    st.header('**Image study**')

    st.info('''Business requirement 1: Help clients understand how the model interprets
images and provide insights into distinctive features.''')

    version = 'v1'

    # Load class labels from the pickle file
    with open('class_labels.pkl', 'rb') as f:
        class_labels = pickle.load(f)

    if st.checkbox('Display the mean and standard deviation from the average image study'):
        # Display each class's average variability image
        for label in class_labels:
            image_path = f'outputs/{version}/avg_var_{label}.png'
            try:
                image = plt.imread(image_path)
                st.image(image, caption=f'Average and Variability for {label}')
            except FileNotFoundError:
                st.error(f"Image not found for class {label}.")

        st.success('There is too much variation in the images to be able to see\
             any distinctive features of the different classes in this image study.')

    if st.checkbox('Show the average differences between similar classes'):
        # Predefined list of class comparisons
        comparisons = [
            'deer vs horse',
            'bird vs plane',
            'truck vs automobile'
        ]

        # Allow the user to select a comparison from the list
        selected_comparison = st.selectbox('Select comparison', options=comparisons)

        if st.button('Show Comparison'):
            # Map the selected comparison to the correct file name format
            comparison_map = {
                'deer vs horse': 'avg_diff_deer_vs_horse.png',
                'bird vs plane': 'avg_diff_bird_vs_plane.png',
                'truck vs automobile': 'avg_diff_truck_vs_automobile.png'
            }

            comparison_image_path = f'outputs/{version}/{comparison_map[selected_comparison]}'

            try:
                avg_differences = plt.imread(comparison_image_path)
                st.image(
                    avg_differences,
                    caption=f'Average differences: {selected_comparison}'
                )
            except FileNotFoundError:
                st.error(f"Comparison image for {selected_comparison} not found.")

        st.warning('Images in the dataset are too similar\
            to see any clear average differences in this image study.')

    # Define function to plot image montage
    def plot_image_montage(image_list, title, ncols=3):
        n_images = len(image_list)
        n_rows = (n_images + ncols - 1) // ncols
        fig, axs = plt.subplots(n_rows, ncols, figsize=(ncols * 3, n_rows * 3))
        axs = axs.flatten()

        for i, img in enumerate(image_list):
            axs[i].imshow(img)
            axs[i].axis('off')

        # Hide any unused subplots
        for i in range(n_images, len(axs)):
            axs[i].axis('off')

        plt.suptitle(title, fontsize=16, y=0.95)
        plt.tight_layout()
        plt.show()

    sample_image_dir = 'outputs/sample_images'

    if st.checkbox('Display Image Montage from Dataset'):
        # Dynamically load the class labels (directories in sample_image_dir)
        labels = os.listdir(sample_image_dir)

        # Let user select multiple labels for the montage
        selected_labels = st.multiselect(
            'Select classes to display',
            options=labels,
            default=labels[:3]
        )

        if st.button('Create Montage'):
            if len(selected_labels) > 0:
                for label in selected_labels:
                    label_path = os.path.join(sample_image_dir, label)
                    image_files = os.listdir(label_path)
                    images = [
                        Image.open(os.path.join(label_path, image_file))
                        for image_file in image_files[:3]
                    ]

                    st.pyplot(plot_image_montage(images, title=label))
            else:
                st.warning('Please select at least one class label to create the montage.')


# Directly call the function in a Streamlit app
page_image_study_body()