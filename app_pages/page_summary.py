import streamlit as st


def page_summary_body():
    st.write('---')
    st.write("### Project Summary")

    st.info(
        """
        This project is a data science and machine learning initiative aimed
        at automating image categorization for e-commerce.
        The core objective is to classify product images into
        one of 10 predefined categories using a machine learning model.
        The application is built with a Streamlit Dashboard, enabling users
        (such as e-commerce managers and product analysts) to upload images
        and receive instant categorizations with detailed reports.

        The dashboard features comprehensive data analysis,
        insights into model performance, and real-time classification results.
        It also provides an overview of the hypotheses tested
        and the performance metrics evaluated.

        The project is optimized for efficiency, with considerations for
        minimizing resource usage during model inference and data handling.
        """
    )

    # Dataset Content
    st.write("### Dataset")
    st.info(
        """
        - **Source:** [CIFAR-10 dataset from Kaggle](https://www.kaggle.com/competitions/cifar-10)
        - **Image Size:** 32x32 pixels in RGB format.
        - **Image Uniformity:** Images within each category are not
            perfectly uniform. 
            - Examples of variations include:
                * **Backgrounds:** Images have diverse backgrounds. 
                * **Angles:** Objects are captured at various angles. 
                * **Lighting:** Lighting conditions across images vary.
        - **Subset:** A subset of 5,000 images was selected for training,
            testing, and validation due to limitations in storage and processing
            time for efficient deployment.
            This subset is split as follows:
            - **Train set:** 70% of the images (350 per class).
            - **Validation set:** 10% of the images (50 per class).
            - **Test set:** 20% of the images (100 per class).
        - **Categories:** 10 classes (e.g., airplanes, cars, birds, frogs).

        This subset balances training efficiency and performance while adhering to GitHub's file size limits.
        """
    )

    # Link to README file for full project documentation
    st.warning(
        "For more detailed information, please visit the [Project README file]"
        "(https://github.com/EVondrus/SnapSortAI)."
    )

    # Business Requirements
    st.write("### Business Requirements")
    st.success(
        """
        1. **Dataset Analysis**:
        - Analyze the CIFAR-10 dataset to understand image distribution, patterns, and potential challenges, such as class imbalance.
        - This will inform preprocessing and model development to ensure accurate and robust image classification. 

        2. **Model Development**:
        - Develop a machine learning model to classify images into 10 categories, aiming to automate the categorization process and improve accuracy.
        - The model should be scalable and efficient for future use. 

        3. **Performance Evaluation**:
        - Evaluate the model’s accuracy and processing speed to ensure practical application and identify areas for further improvement.
        - The model should achieve a minimum of 70% accuracy on the test set. 
        """
    )

    st.write('---')
