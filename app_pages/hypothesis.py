import streamlit as st

def page_hypothesis_body():
    st.write("### Hypothesis Statement")

    st.success(
        "The hypotheses for this project are designed to guide the development "
        "and evaluation of the image classification model using the CIFAR-10 "
        "dataset.\n\n"
        
        "1. **Exploratory Data Analysis (EDA)**: We hypothesize that analyzing "
        "the CIFAR-10 dataset will uncover patterns and challenges that will "
        "inform the preprocessing steps and guide model development. "
        "Understanding image distribution across the 10 classes is expected to "
        "help optimize the model for these categories.\n\n"
        
        "2. **Convolutional Neural Network (CNN) Performance**: We anticipate "
        "that a well-architected CNN will achieve at least 80% classification "
        "accuracy on the CIFAR-10 dataset, which meets the client’s "
        "requirements for effective automation of the categorization process.\n\n"
        
        "3. **Impact of Data Augmentation**: Applying data augmentation "
        "techniques (e.g., rotations, flips, and zooms) is expected to improve "
        "the model's accuracy by at least 5%, aiming for a final accuracy of "
        "85% or more. These techniques will help the model generalize better to "
        "unseen data and reduce overfitting.\n\n"
        
        "4. **Generalization and Robustness**: The CNN should generalize well "
        "to unseen data, utilizing regularization techniques such as dropout "
        "and batch normalization. The performance will be validated with a "
        "separate set of images: 50 images per class for validation and 100 "
        "images per class for testing.\n\n"
        
        "Key elements of the CNN model include:\n\n"
        "1. **Feature Extraction**: The CNN will capture hierarchical features, "
        "from simple edges to more complex shapes and patterns.\n"
        "2. **Architecture**: The CNN’s layers will be tailored to process the "
        "32x32 RGB images, making it efficient for multi-class classification.\n"
        "3. **Regularization**: Techniques like dropout and batch normalization "
        "will be employed to prevent overfitting and improve model robustness."
    )

    st.write('---')

    st.warning(
        "### Model Limitations Notice:\n\n"
        "1. **Dataset Size**: Each class in the CIFAR-10 dataset contains 350 "
        "training images, 100 test images, and 50 validation images. While this "
        "distribution provides balance across classes, the limited number of "
        "images per class may impact the model's ability to generalize to more "
        "diverse real-world images.\n\n"
        
        "2. **Potential Class Imbalance**: Although each class is evenly "
        "distributed, the small dataset size could still limit the model’s "
        "ability to capture more complex patterns or variations within a class, "
        "especially for more visually complex categories."
    )
    
    st.write('---')

    # Conclusion section
    st.write("### Conclusions")


    st.write('---')

    # Validation steps section
    st.write("### Validation and Testing")