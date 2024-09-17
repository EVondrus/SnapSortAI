import streamlit as st

def page_summary_body():
    st.write("### Project Summary")

    st.info(
        "This project is a data science and machine learning initiative aimed "
        "at automating image categorization for e-commerce.\n\n"
        
        "The core objective is to classify product images into one of 10 "
        "predefined categories using a machine learning model. The application "
        "is built with a Streamlit Dashboard, enabling users (such as "
        "e-commerce managers and product analysts) to upload images and "
        "receive instant categorizations with detailed reports.\n\n"
        
        "The dashboard features comprehensive data analysis, insights into "
        "model performance, and real-time classification results. It also "
        "provides an overview of the hypotheses tested and the performance "
        "metrics evaluated.\n\n"
        
        "The project is optimized for efficiency, with considerations for "
        "minimizing resource usage during model inference and data handling."
    )

    # Dataset Content
    st.info(
        "**Project Dataset**\n\n"
        "* **CIFAR-10 Dataset**: Contains 60,000 images across 10 categories, "
        "including airplanes, cars, birds, and more.\n"
        "* **Image Dimensions**: 32x32 pixels in RGB format.\n"
        "* **Subset Used**: A carefully selected subset of 5,000 images to "
        "balance training efficiency with performance while keeping the "
        "repository size manageable.\n\n"
        
        "This dataset provides a diverse representation of categories and is "
        "crucial for building and evaluating the image classification model."
    )

    # Link to README file for full project documentation
    st.warning(
        "For more detailed information, please visit the [Project README file]"
        "(https://github.com/EVondrus/SnapSortAI)."
    )

    # Business Requirements
    st.success(
        "**Business Requirements**\n\n"
        "* **1. Dataset Analysis**: Analyze the CIFAR-10 dataset to understand "
        "image distribution, patterns, and potential challenges to inform "
        "preprocessing and model development.\n"
        "* **2. Model Development**: Develop a machine learning model to "
        "classify images into 10 categories, aiming to automate the "
        "categorization process and improve accuracy.\n"
        "* **3. Performance Evaluation**: Evaluate the model’s accuracy and "
        "processing speed to ensure practical application and identify areas "
        "for further improvement."
    )
