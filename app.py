import streamlit as st
from app_pages.multipage import MultiPage

st.set_page_config(  # First Streamlit command
    page_title="SnapSort AI",
    page_icon="👀",
)

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.visualizer import page_visualizer_body
from app_pages.model_performance import page_ml_performance_body
from app_pages.object_detector import page_object_detector_body
from app_pages.hypothesis import page_hypothesis_body

app = MultiPage(app_name="SnapSort AI") # Create an instance of the app 

# Add app pages here using .add_page()
app.add_page("Project Summary", page_summary_body)
app.add_page("Hypothesis Statement", page_hypothesis_body)
app.add_page("Data Visualizer", page_visualizer_body)
app.add_page("Model Performance", page_ml_performance_body)
app.add_page("Object Identification", page_object_detector_body)

app.run()