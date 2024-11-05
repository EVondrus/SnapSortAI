import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from source.machine_learning.evaluate_clf import load_test_evaluation
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from source.data_management import load_pkl_file
from tensorflow.keras.utils import image_dataset_from_directory


def page_ml_performance_body():

    st.write('---')

    st.header('**ML Performance Metrics**')

    version = 'v7'

    st.info("Labels Frequencies in the dataset")

    labels_distribution = plt.imread(
        f"outputs/v1/distribution_plot.png")
    st.image(labels_distribution,
             caption='Labels Distribution on Train, Validation and Test Sets')

    st.write("---")

    st.success("Model training history")
    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')

    st.write("---")

    st.warning("Generalised performance on the test set")
    st.dataframe(pd.DataFrame(
        load_test_evaluation(version), index=['Loss', 'Accuracy']))

    st.write("---")

    st.success("Confusion Matrix")
    confusion_matrix_image = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(confusion_matrix_image, caption="Confusion Matrix", use_column_width=True)

    st.subheader("Confusion Matrix Explanation")
    st.write(
        """
        **Strong Performances**: Our model shows strong accuracy for recognizing 
        'automobile' (81 out of 100 correct), 'frog' (89 out of 100 correct), 
        and 'ship' (84 out of 100 correct).

        **Challenges**: The model faces challenges with some classes, particularly:

        - **Airplane**: The model often misclassifies 'airplane' as 'automobile' 
        (5 out of 100) and 'ship' (24 out of 100).
        - **Bird**: There's a significant amount of confusion between birds and 
        horses (12 out of 100). Birds are also frequently misclassified as 
        ships (26 out of 100).
        - **Cat**: The model struggles to distinguish cats from dogs (26 out of 100).
        - **Deer**: Deer images are consistently misclassified as 'dog' (65 out 
        of 100), 'ship' (37 out of 100), and 'truck' (12 out of 100).
        - **Horse**: Horses are often confused with birds (10 out of 100) and 
        ships (35 out of 100).
        - **Truck**: The model has difficulty recognizing trucks, misclassifying 
        them as 'automobile' (7 out of 100) and 'ship' (11 out of 100).

        **Key Observations**:

        - **Visual Similarities**: Our model seems to struggle with categories 
        that share visual similarities, a common challenge in image classification. 
        The model often mistakes airplanes for automobiles, birds for horses, 
        and deer for dogs.
        - **Class-Specific Errors**: The model is particularly struggling with 
        deer images, which are being misclassified into various other categories.

        **Next Steps**:

        Based on these findings, we'll prioritize improving our model's accuracy 
        by focusing on these key areas:

        - **Data Augmentation**: To combat overfitting, we will implement data 
        augmentation techniques to generate variations of our training images. 
        This will help our model learn more robust features and generalize better.
        - **Model Complexity**: We will explore more complex model architectures. 
        This could involve using a pre-trained model or adding more convolutional 
        layers to our existing model.
        - **Class Weights**: We'll investigate the use of class weights during 
        training to give more importance to the loss for classes that are 
        challenging for the model, such as 'deer.'
        - **Hyperparameter Tuning**: We will conduct systematic hyperparameter 
        tuning to find the best combination of settings for our model 
        architecture and training data.
        """
    )


    st.write("---")



