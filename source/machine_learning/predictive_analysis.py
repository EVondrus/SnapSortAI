
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from PIL import Image
from source.data_management import load_pkl_file

from tensorflow.keras.models import load_model


def plot_predictions_and_probabilities(pred_proba, pred_class):
    """
    Plot the prediction and probability result
    """

    # Class labels for CIFAR-10 dataset
    class_labels = [
        'Airplane',
        'Automobile',
        'Bird',
        'Cat',
        'Deer',
        'Dog',
        'Frog',
        'Horse',
        'Ship',
        'Truck'
    ]

    # Create DataFrame with probabilities for all classes
    prob_per_class = pd.DataFrame(
        data=pred_proba,  # A (10,) array, probabilities for all 10 classes
        index=class_labels,
        columns=['Probability']
    )

    # Round probabilities for better readability
    prob_per_class = prob_per_class.round(3)

    # Add a 'Diagnostic' column for easier plotting
    prob_per_class['Diagnostic'] = prob_per_class.index

    # Create the bar chart with Plotly
    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        range_y=[0, 1],
        width=600, height=300, template='seaborn'
    )

    st.plotly_chart(fig)


def resize_input_image(img, version):
    """
    Reshape image to image determined in the data visualization notebook
    """
    # Convert to RGB before resizing
    img = img.convert('RGB')

    image_shape = load_pkl_file(file_path=f"outputs/v1/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    resized_image = np.expand_dims(img_resized, axis=0)/255

    return resized_image


def load_model_and_predict(my_image, version):
    """
    Load model and predict class on live images
    """

    model = load_model(f'outputs/{version}/snapsort_model.h5')

    pred_proba = model.predict(my_image)[0]

    target_map = {value: key for key, value in {
        'Airplane': 0,
        'Automobile': 1,
        'Bird': 2,
        'Cat': 3,
        'Deer': 4,
        'Dog': 5,
        'Frog': 6,
        'Horse': 7,
        'Ship': 8,
        'Truck': 9}.items()
    }

    # Get the index of the class with the highest probability
    pred_class_procent = np.argmax(pred_proba)
    pred_class = target_map[pred_class_procent]

    print("Predicted Probabilities:", pred_proba)
    print("Predicted Class:", pred_class)

    return pred_proba, pred_class
