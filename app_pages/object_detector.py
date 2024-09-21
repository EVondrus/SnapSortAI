import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from source.data_management import download_dataframe_as_csv
from source.machine_learning.predictive_analysis import (
    resize_input_image,
    load_model_and_predict,
    plot_predictions_and_probabilities,
)


def page_object_detector_body():

    st.write('---')

    st.header('**Image classifier**')

    st.info('**Business Requirement 2:** This feature allows clients to upload\
    images from various classes and receive real-time predictions.\
    It automates the classification process, providing immediate feedback on\
    the models predictions and associated probabilities.')

    st.warning('''
    Click [**here**](
    https://github.com/EVondrus/SnapSortAI/blob/main/outputs/sample_images.zip
    ),
    to download a set of images from each class for live prediction.''')

    images_buffer = st.file_uploader(
        'Upload images from the different classes here, You can select\
        more than one at the time.', type=['png', 'jpg'],
        accept_multiple_files=True
    )

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f'Uploaded image: *{image.name}*')
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f'Image size: {img_array.shape[1]} px \
    width x {img_array.shape[0]} px height')

            version = 'v5'

            resized_img = resize_input_image(img=img_pil, version=version)
            predict_probability, predict_class = load_model_and_predict(
                resized_img, version=version)

            plot_predictions_and_probabilities(
                predict_probability, predict_class)

            df_report = pd.concat([df_report, pd.DataFrame(
                {'Name': [image.name], 'Result': [predict_class]}
                )], ignore_index=True)

        if not df_report.empty:
            st.warning('Analysis Report')
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report),
                        unsafe_allow_html=True)
