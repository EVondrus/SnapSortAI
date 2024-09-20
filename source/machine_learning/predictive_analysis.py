
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from PIL import Image
from source.data_management import load_pkl_file

from tensorflow.keras.models import load_model


def plot_predictions_and_probabilities(predicted_probability, predicted_class):
    """
    Plot the prediction and probability result
    """

    probability_per_class = pd.DataFrame(
        data = [0, 0],
        index = {'Cat': 0, 'Dog': 1}.keys(),
        columns = ['Probability'],
        dtype=np.float64,
    )

    probability_per_class.loc[predicted_class] = predicted_probability

    for x in probability_per_class.index.to_list():
        if x not in predicted_class:
            probability_per_class.loc[x] = 1 - predicted_probability

    probability_per_class = probability_per_class.round(3)
    probability_per_class['Diagnostic'] = probability_per_class.index

    fig = px.bar(
        probability_per_class, x = 'Diagnostic',
        y = probability_per_class['Probability'], range_y = [0, 1],
        width = 600, height = 300, template = 'seaborn',
    )
    st.plotly_chart(fig)


def resize_input_image(img, version):
    """
    Reshape image to image determined in the data visualization notebook
    """
    # Convert to RGB before resizing
    img = img.convert('RGB')

    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)

    resized_image = np.expand_dims(img_resized, axis=0)/255

    return resized_image


def load_model_and_predict(image, version):
    """
    Load model and predict class on live images
    """

    model = load_model(f'outputs/{version}/cats_vs_dogs_model.h5')

    predicted_probability = model.predict(image)[0, 0]

    # Get the target classes as a tuple stored in target_map (maps targets)
    target_map = {value: key for key, value in {'Cat': 0, 'Dog': 1,}.items()}

    # Set the class that has higher than 0.5 probability as predicted class
    predicted_class = target_map[predicted_probability > 0.5]

    if predicted_class == target_map[0]:
        predicted_probability = 1 - predicted_probability
        st.success(f'The predictive analysis indicates that the image is\
                   of a {predicted_class.lower()}.')
    else:
        st.error(f'The predictive analysis indicates that the image is\
                   of a {predicted_class.lower()}.')

    return predicted_probability, predicted_class
