import streamlit as st


def page_hypothesis_body():

    st.write('---')

    st.header('**Project Hypotheses**')

    st.write('---')

    st.info('''
**Hypothesis 1:** Exploratory Data Analysis (EDA) will reveal patterns and\
challenges that will guide preprocessing and model development.
''')

    st.warning('''
**Validation Method:** Conduct Exploratory Data Analysis on the\
CIFAR-10 dataset, including visualizing class distribution and image quality.
''')

    st.success('''
**Conclusion:** EDA effectively identified class imbalances and image quality\
issues, guiding preprocessing steps to improve model training.\
**Hypothesis confirmed**.
''')

    st.write('---')

    st.info('''
**Hypothesis 2:** The CNN model will achieve at least\
70% classification accuracy.
''')

    st.warning('''
**Validation Method:** Train a Convolutional Neural Network (CNN) and evaluate\
its performance on the test set.
''')

    st.success('''
**Conclusion:** The best model, v7, achieved a test accuracy of approximately 71%.
''')

    st.write('---')

    st.info('''
**Hypothesis 3:** Data augmentation (e.g., rotating, flipping, zooming) will\
    enhance model accuracy by at least 5%.
''')

    st.warning('''
**Validation Method:** Compare model performance with and\
    without data augmentation.
''')

    st.success('''
**Conclusion:**Data augmentation resulted in a notable performance improvement,\
    with test accuracy increasing from 62.60% to 67.20% in model v6, confirming \
    the hypothesis with a gain of 5.60%.
''')

    st.write('---')

    st.info('''
**Hypothesis 4:** Fine-tuning hyperparameters will significantly\
enhance model accuracy.
''')

    st.warning('''
**Validation Method:** Compare model performance with default\
hyperparameters versus tuned hyperparameters.
''')

    st.success('''
**Conclusion:**Multiple adjustments were made and\
    the best model did exceed the 70% accuracy threshold.
''')

    st.write('---')
