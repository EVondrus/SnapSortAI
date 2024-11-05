import streamlit as st


def page_hypothesis_body():
    st.write('---')

    st.header('**Project Hypotheses**')

    st.write('---')

    st.info(
        '''  
        **Hypothesis 1:** Exploratory Data Analysis (EDA) on the CIFAR-10 dataset
        will reveal patterns and challenges, including class imbalances and image
        quality issues, which will guide preprocessing steps and model development.
        '''
    )

    st.warning(
        '''  
        **Validation Method:** Conduct EDA on the CIFAR-10 dataset,
        visualizing class distribution (e.g., with bar charts), exploring
        the range of image sizes, and examining sample images for quality.
        '''
    )

    st.success(
        '''  
        **Conclusion:** EDA effectively identified potential class imbalances
        and image quality issues. These insights informed preprocessing steps
        and helped guide the model's architecture and training strategy.
        **Hypothesis confirmed**.
        '''
    )

    st.write('---')

    st.info(
        '''  
        **Hypothesis 2:** A convolutional neural network (CNN) model will achieve
        at least 70% classification accuracy on the CIFAR-10 dataset.
        '''
    )

    st.warning(
        '''  
        **Validation Method:** Train a CNN model using a dataset split into
        training, validation, and test sets. Evaluate the model's
        performance on the test set using accuracy as the primary metric.
        '''
    )

    st.success(
        '''  
        **Conclusion:** The final model (v7) achieved a test accuracy of
        approximately 71.20%, exceeding the initial 70% target.
        **Hypothesis confirmed**.
        '''
    )

    st.write('---')

    st.info(
        '''  
        **Hypothesis 3:** Data augmentation techniques, such as rotation,
        flipping, zooming, and shearing, will significantly enhance
        the model's accuracy and generalization ability.
        '''
    )

    st.warning(
        '''  
        **Validation Method:** Compare the performance of models trained
        with and without data augmentation. Analyze the confusion matrix
        and classification report for each model to assess the impact
        of data augmentation.
        '''
    )

    st.success(
        '''  
        **Conclusion:** The final model (v7) incorporated data augmentation
        and achieved a notable improvement in test accuracy
        (e.g., from 62.60% in v5 to 71.20% in v7), confirming the hypothesis
        that data augmentation is crucial for improving model accuracy
        and generalization.
        '''
    )

    st.write('---')

    st.info(
        '''  
        **Hypothesis 4:** Optimizing the model architecture and
        hyperparameters will further enhance the model's performance.
        '''
    )

    st.warning(
        '''  
        **Validation Method:** Experiment with different model architectures
        (e.g., adding more layers, using pre-trained models) and hyperparameters
        (e.g., learning rate, batch size, regularization strength) to find the
        best combination for the CIFAR-10 dataset.
        '''
    )

    st.success(
        '''  
        **Conclusion:** The final model (v7) incorporated a more complex
        architecture with three convolutional layers, batch normalization,
        and max pooling. Hyperparameters were fine-tuned through multiple
        training iterations. This optimization process contributed to the
        model achieving over 70% accuracy. **Hypothesis confirmed**.
        '''
    )

    st.write('---')
