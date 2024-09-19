import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
#from src.machine_learning.evaluate_clf import load_test_evaluation

def page_ml_performance_body():

    st.write('---')

    st.header('**ML Performance Metrics**')

    version = 'v1'

    st.info("Labels Frequencies in the dataset")

    labels_distribution = plt.imread(
        f"outputs/{version}/distribution_plot.png")
    st.image(labels_distribution,
             caption='Labels Distribution on Train, Validation and Test Sets')

    st.write("---")