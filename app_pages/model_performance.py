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
    # Load model
    model = load_model(f'outputs/{version}/snapsort_model.h5')

    image_shape = load_pkl_file(file_path=f"outputs/v1/image_shape.pkl")

    class_names = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]

    test_dataset = image_dataset_from_directory(
        f"inputs/cifar10_dataset_small/test",
        labels='inferred',  # Automatically infer labels from folder names
        label_mode='categorical',  # Output labels as one-hot encoded vectors
        image_size=image_shape[:2],
        interpolation='nearest',
        batch_size=32,
        shuffle=False
    )

    # Convert test_dataset to NumPy arrays
    test_images = np.concatenate([batch[0] for batch in test_dataset.as_numpy_iterator()])
    test_labels = np.concatenate([batch[1] for batch in test_dataset.as_numpy_iterator()])

    # Predict class labels for the test set
    predictions = np.argmax(model.predict(test_images), axis=1)

    # Create confusion matrix
    cm = confusion_matrix(np.argmax(test_labels, axis=1), predictions)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix for Test Set")
    plt.tight_layout()

    st.pyplot(fig=plt)

    st.subheader("Confusion Matrix Explanation")
    st.write(
        """
        This confusion matrix shows how well our model is doing at classifying different types of objects in the test set. 
        Each row represents the actual category of an image, and each column represents the category that our model predicted. 
        The numbers in each cell show the number of times the model predicted a specific class when the actual class was different.
        
        For example, the model correctly classified 91 out of 100 automobile images. 
        But, it often misclassified airplane images as automobiles.
        
        This matrix tells us that the model is doing well with some categories, 
        but it struggles to distinguish between others, especially those that share visual similarities (like birds and horses). 
        We need to focus on improving the model's accuracy in these areas.
        """
    )

    st.write("---")

    st.subheader("Classification Report")
    report = classification_report(
    np.argmax(test_labels, axis=1), 
    predictions, 
    target_names=class_names, 
    output_dict=True,
    zero_division=1
)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.write("---")


