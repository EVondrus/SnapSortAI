# CIFAR-10 Image Classification Project

**A Data Science and Machine Learning project developed as part of a course in Predictive Analytics, my final project at Code Institute.**

---

## Project Overview

A client working in e-commerce wants to explore how machine learning can help automate the process of categorizing product images. Currently, this task is manual, time-consuming, and prone to errors.

The client is seeking a two-phase approach: 
1. **Dataset Analysis**: Understanding the CIFAR-10 dataset.
2. **Model Development**: Creating an image classification model for automation.

The CIFAR-10 dataset contains 60,000 images across 10 distinct categories (e.g., animals, vehicles, and objects). The goal is to provide valuable insights and build a basic model for automation, with future scalability in mind.

---

## Dataset Content

- **Source**: [CIFAR-10 dataset from Kaggle](https://www.kaggle.com/competitions/cifar-10).
- **Image Size**: 32x32 pixels in RGB format.
- **Total Images**: A subset of 5,000 images for efficient training and diverse category representation.
- **Categories**: 10 categories (e.g., airplanes, cars, birds, frogs, ships).

This subset balances training efficiency and performance while keeping the repository size manageable. 

**TODO**:
- Add example images with labels.
- Include total file size to confirm it’s under GitHub's limit.

---

## Business Requirements

1. **Dataset Analysis**:
   - Analyze the CIFAR-10 dataset to understand image distribution, patterns, and potential challenges.

2. **Model Development**:
   - Develop a machine learning model for classifying images into 10 categories to reduce manual effort and improve accuracy.

3. **Performance Evaluation**:
   - Evaluate the model’s performance focusing on accuracy and processing speed to ensure practical real-world application.

[Back to top](#cifar-10-image-classification-project)

---

## Hypothesis and Validation Process

- **Hypothesis 1**: 
  - Exploratory Data Analysis (EDA) will reveal patterns and challenges that will guide preprocessing and model development.

- **Hypothesis 2**: 
  - A basic Convolutional Neural Network (CNN) will achieve at least 70% classification accuracy, providing a baseline for automation.

- **Hypothesis 3**: 
  - Data augmentation (e.g., rotating, flipping, zooming) will enhance model accuracy by at least 5%.

### Validation Process

- **Accuracy**: Measures overall correctness of predictions.
- **Precision & Recall**: Precision indicates correct predictions, while recall shows the number of true positives.
- **F1-Score**: Combines precision and recall into a single metric for balanced evaluation.
- **Confusion Matrix**: Visualizes classification errors and misclassifications.
- **Comparison**: Models with and without data augmentation and regularization will be compared.

### Success Metrics

- **Target Accuracy**: At least 70% accuracy on the test set.
- **Data Augmentation Impact**: Expect at least a 5% improvement in accuracy or F1-score with augmentation.

### Implications

Validating these hypotheses will confirm the efficacy of combining CNNs with advanced techniques and guide future model improvements.

[Back to top](#cifar-10-image-classification-project)

---

## Rationale for Mapping Business Requirements to Data Visualizations and ML Tasks

### 1. Dataset Exploration
- **Requirement**: Analyze the CIFAR-10 dataset.
- **Rationale**: Effective preprocessing and model training rely on understanding data distribution and imbalances.

### 2. Model Development
- **Requirement**: Build and optimize a CNN.
- **Rationale**: A well-tuned CNN automates categorization efficiently and can be enhanced with data augmentation and regularization.

### 3. Performance Metrics
- **Requirement**: Achieve high accuracy.
- **Rationale**: Metrics like precision, recall, and F1 score will guide model evaluation and refinement.

### 4. Interpretability
- **Requirement**: Provide insights into model predictions.
- **Rationale**: Visualizations (e.g., confusion matrices) help users understand and trust the model’s decisions.

### 5. Efficiency
- **Requirement**: Ensure efficient training and inference.
- **Rationale**: Efficient models support real-time applications and scalability.

---

## ML Business Case

### 1. Dataset Exploration
- **Task**: Perform EDA and visualize class distribution.
- **Business Case**: Informs preprocessing and improves model performance.

### 2. Model Development
- **Task**: Develop and train a CNN with data augmentation.
- **Business Case**: Automates product categorization, benefiting client operations.

### 3. High Accuracy
- **Task**: Evaluate model performance.
- **Business Case**: Reduces manual errors and meets client expectations.

### 4. Interpretability
- **Task**: Generate visualizations for predictions.
- **Business Case**: Enhances understanding and trust in the model.

### 5. Efficiency
- **Task**: Optimize training and inference.
- **Business Case**: Ensures practical deployment and scalability.

---

## Model Development

### CNN Architecture
- **Input**: 32x32 RGB images.
- **Layers**: Convolutional layers with ReLU, max-pooling, dense layers with softmax.
- **Regularization**: Dropout and batch normalization.

### Image Preprocessing
- **Rescaling**: Normalize images to [0, 1].
- **Data Augmentation**: Apply rotations, flips, and zooms.

### Model Compilation
- **Optimizer**: Adam.
- **Loss Function**: Categorical cross-entropy.
- **Metrics**: Accuracy.

### Training and Validation
- **Approach**: Train with early stopping and checkpointing. Evaluate on a test set.

### Model Development Diagram
![Model Development Diagram](link-to-diagram-image)

### Technical Setup

#### CNN Architecture
- **Input Layer**: 32x32 RGB images.
- **Convolutional Layers**: Extract spatial features with ReLU activation, followed by max-pooling.
- **Dense Layers**: Fully connected layers with a softmax output.
- **Regularization**: Dropout and batch normalization.

#### Image Preprocessing
- **Rescaling**: Normalize images to 0-1.
- **Data Augmentation**: Enhance robustness with rotations, flips, and zooms.

#### Model Compilation
- **Optimizer**: Adam.
- **Loss Function**: Categorical cross-entropy.
- **Metrics**: Accuracy.

#### Training and Validation
- **Approach**: Use early stopping and checkpointing. Evaluate on a separate test set.

[Back to top](#cifar-10-image-classification-project)

---

## Dashboard Design

### Dashboard Pages:

1. **Project Summary**: Overview of the project and objectives.
2. **Data Visualizer**: Visual representation of the dataset and preprocessing steps.
3. **Model Performance**: Detailed metrics, including accuracy and confusion matrix.
4. **Image Classifier**: Upload images and receive real-time predictions.
5. **Hypothesis and Results**: Discussion of hypothesis, results, and future work.

The Streamlit Dashboard includes:
<details><summary>1st Page - Project Summary</summary>
<br><br>
This page provides a summary of the project, objectives, and business requirements:
<img src="" width="80%" height="80%"><br></details>
<br><br>
<details><summary>2nd Page - Data Visualizer</summary>
<br><br>
Shows the type of data and preprocessing steps:
<img src="" width="80%" height="80%"><br></details>
<br><br>
<details><summary>3rd Page - Model Performance</summary>
<br><br>
Details the model's performance metrics:
<img src="" width="80%" height="80%"><br></details>
<br><br>
<details><summary>4th Page - Image Classifier</summary>
<br><br>
Allows users to upload images and make live predictions:
<img src="" width="80%" height="80%"><br></details>
<br><br>
<details><summary>5th Page - Hypothesis and Results</summary>
<br><br>
Discusses the hypothesis and points out considerations for using the model:
<img src="" width="80%" height="80%"><br></details>

[Back to top](#cifar-10-image-classification-project)

---

## Kanban Board

### User Stories

The Kanban board outlines key user stories and deadlines to ensure the project meets client objectives.

#### 1. Interactive Dashboard Navigation
- **Story**: As a client, I want an interactive dashboard for easy data visualization.
- **Goal**: Ensure intuitive navigation and effective data analysis.

#### 2. Data Verification
- **Story**: As a client, I want to verify the accuracy of the data.
- **Goal**: Provide transparency and data integrity.

#### 3. Model Accuracy Demonstration
- **Story**: As a client, I want a clear demonstration of the model's accuracy.
- **Goal**: Showcase model effectiveness with technical details.

#### 4. Model Testing by Uploading Pictures
- **Story**: As a client, I want to upload pictures to test the model’s capabilities.
- **Goal**: Enable interaction with the model using client data.

#### 5. Understanding Technical Processes
- **Story**: As a client, I want to understand the technical processes involved in building the model.
- **Goal**: Provide insights into model development.

### Project Status

All user stories have been successfully implemented, and the project is complete.

[Back to top](#cifar-10-image-classification-project)


---

## Unfixed Bugs

There are no known unfixed bugs at this time.

[Back to top](#cifar-10-image-classification-project)

---

## Deployment

### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

---

## Main Data Analysis and Machine Learning Libraries

- **NumPy**: Essential for performing numerical operations on image data, including preprocessing and preparing data for model input.
- **Pandas**: Used to manage and analyze structured data related to images, such as filenames, labels, and image dimensions.
- **Matplotlib**: Utilized to create static visualizations, including plots of model accuracy and loss over epochs.
- **Seaborn**: Enhances Matplotlib with additional statistical plotting capabilities, helping to visualize data distributions and model performance.
- **Plotly**: Provides interactive visualizations within the Streamlit app, allowing users to explore predictions and data in more detail.
- **TensorFlow**: Core library for defining and training the convolutional neural network (CNN) model, including building and optimizing the architecture.
- **Keras**: A high-level API used alongside TensorFlow for constructing and evaluating the CNN model, simplifying the process of model development.
- **Joblib**: Facilitates saving and loading the trained machine learning model, enabling quick deployment without the need for retraining.
- **Streamlit**: Used to create an interactive dashboard for the client, providing a user-friendly interface for visualizing model performance and making predictions.
- **Kaggle**: The source of the CIFAR-10 dataset, which serves as the basis for training and evaluating the image classification model.

[Back to top](#cifar-10-image-classification-project)


---

## Credits 

Thank you to Code Institute for providing the deployment-ready template and a comprehensive README structure.

### Content 

- The dataset is sourced from [Kaggle](https://www.kaggle.com/competitions/cifar-10).

### Code

- Parts of the code from Code Institute's walkthrough project were utilized and adapted for this project.

---

## Acknowledgements

- Special thanks to those who provided support throughout this project.

