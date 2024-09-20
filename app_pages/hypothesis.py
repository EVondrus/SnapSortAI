import streamlit as st

def page_hypothesis_body():

    st.write('---')

    st.header('**Project Hypotheses**')

    st.write('---')

    st.info('''
**Hypothesis 1:** Exploratory Data Analysis (EDA) will reveal patterns and challenges that will guide preprocessing and model development.
''')

    st.warning('''
**Validation Method:** Conduct Exploratory Data Analysis on the CIFAR-10 dataset, including visualizing class distribution and image quality.
''')

    st.success('''
**Conclusion:** EDA effectively identified class imbalances and image quality issues, guiding preprocessing steps to improve model training. **Hypothesis confirmed**.
''')

    st.write('---')

    st.info('''
**Hypothesis 2:** A basic CNN will achieve at least 70% classification accuracy.
''')

    st.warning('''
**Validation Method:** Train a Convolutional Neural Network (CNN) and evaluate its performance on the test set.
''')

    st.success('''
**Conclusion:** 
''')

    st.write('---')

    st.info('''
**Hypothesis 3:** Data augmentation (e.g., rotating, flipping, zooming) will enhance model accuracy by at least 5%.
''')

    st.warning('''
**Validation Method:** Compare model performance with and without data augmentation.
''')

    st.success('''
**Conclusion:** 
''')

    st.write('---')

    st.info('''
**Hypothesis 4:** Fine-tuning hyperparameters will significantly enhance model accuracy.
''')

    st.warning('''
**Validation Method:** Compare model performance with default hyperparameters versus tuned hyperparameters.
''')

    st.success('''
**Conclusion:** 
''')

    st.write('---')

    st.info('''
**Hypothesis 5:** Training on higher-resolution images will improve accuracy.
''')

    st.warning('''
**Validation Method:** Train and evaluate the model on datasets with varying image resolutions.
''')

    st.success('''
**Conclusion:** 
''')

    st.write('---')
