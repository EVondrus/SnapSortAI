# CIFAR-10 Image Classification Project
**A Data Science and Machine Learning project developed as part of an course in Predictive Analytics, my final project at Code Institute.**

## Project Overview
This project is a comprehensive exploration of image classification using Convolutional Neural Networks (CNNs) applied to the CIFAR-10 dataset. The goal is to achieve high accuracy in categorizing images across 10 distinct classes. The project explores various preprocessing techniques and model architectures to optimize performance. 

**Purpose:**
The primary purpose of this project is to create an educational tool that aids students, researchers, and AI enthusiasts in understanding and applying CNNs for image classification tasks. CHECK: (This project serves as a benchmark for performance) and provides a foundation for more advanced image recognition models.

**Target Audience:**
- **Students**: A resource for learning about CNNs and image classification.
- **Researchers**: A benchmark model for exploring and refining image classification techniques.
- **AI Enthusiasts**: An interactive tool for experimenting with deep learning models.

---

## Dataset Content
(* Describe your dataset. Choose a dataset of reasonable size to avoid exceeding the repository's maximum size and to have a shorter model training time. If you are doing an image recognition project, we suggest you consider using an image shape that is 100px × 100px or 50px × 50px, to ensure the model meets the performance requirement but is smaller than 100Mb for a smoother push to GitHub. A reasonably sized image set is ~5000 images, but you can choose ~10000 lines for numeric or textual data.)

The CIFAR-10 dataset contains 60,000 color images categorized into 10 distinct classes. The dataset is divided into 50,000 training images and 10,000 test images, offering a balanced and diverse sample for training and evaluating machine learning models.

- **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Dimensions**: 32x32 pixels with RGB color channels
- **Objective**: To classify an image into one of the 10 classes accurately.

TODO:
ADD EXAMPLE IMAGES WITH LABELS!
**Size:** Include total file size to confirm it’s under the GitHub limit.

---


## Business Requirements

The client requires the development of a robust image classification model using the CIFAR-10 dataset. This entails:

1. **Dataset Exploration**: Conduct a thorough analysis of image characteristics and distribution across classes.
2. **Model Development**: Create and optimize a CNN model to accurately classify images into one of the 10 CIFAR-10 categories.
3. **Educational Support**: Develop an interactive platform to aid students, researchers, and AI enthusiasts in understanding and applying deep learning techniques.

The project aims to provide:
- **High Accuracy**: Achieve over 90% accuracy on the CIFAR-10 test set.
- **Interpretability**: Offer clear visualizations and insights into model predictions.
- **Efficiency**: Ensure fast training and inference for practical use.
- **Usability**: Deliver an easy-to-use interface for deploying and experimenting with the model.

TODO:
Specify the baseline accuracy of typical models on the CIFAR-10 dataset for context.
Mention any specific targets or thresholds for training and inference times,

[Back to top](#cifar-10-image-classification-project)

---

## Hypothesis and Validation Process

### Problem Statement
This project hypothesizes that advanced data augmentation, batch normalization, and dropout techniques will significantly enhance the CNN's performance on the CIFAR-10 dataset, leading to high classification accuracy.

### Expected Model Behavior
The CNN is expected to:
- Leverage its architecture to extract complex features from images.
- Utilize regularization techniques to reduce overfitting and improve test set performance.

### Assumptions
- The dataset is balanced, with each class adequately represented.
- Data augmentation will enhance the model’s robustness and generalization.
- The CNN architecture will effectively capture spatial hierarchies in 32x32 images.

### Validation Process
- **Statistical Tests**: Evaluate model performance using accuracy, precision, recall, and F1-score. Compare models with and without augmentation, batch normalization, and dropout.
- **Model Outputs**: Analyze the model’s predictions using confusion matrices and ROC-AUC curves to understand misclassifications and class discrimination.

### Success Metrics
The model will be considered successful if it achieves at least 90% accuracy on the test set.

### Implications
Validating this hypothesis will confirm the efficacy of combining CNNs with advanced regularization techniques and guide future model designs.

TODO:
Expand on the types of statistical tests or techniques

[Back to top](#cifar-10-image-classification-project)

---

## Business Case

### Overview
The client seeks to develop a high-accuracy image classification model using the CIFAR-10 dataset. This project aims to leverage CNNs to achieve robust performance in categorizing images into one of the 10 classes, providing a valuable tool for educational and research purposes.

### Technical Approach
The project employs Convolutional Neural Networks due to their proven ability to capture spatial hierarchies in image data. Techniques such as data augmentation, dropout, and batch normalization are integrated to enhance model generalization and performance.

### Objectives and Expected Outcomes
1. **High Accuracy**: The goal is to exceed 90% accuracy on the CIFAR-10 test set.
2. **Model Interpretability**: Provide clear and understandable insights into model predictions through visualizations.
3. **Efficiency**: Optimize the model for fast training and inference, making it suitable for educational environments.

### Performance Metrics
Success will be measured using:
- **Accuracy**: The proportion of correct classifications among all predictions.
- **Precision and Recall**: Evaluate the model's ability to correctly identify each class.
- **F1 Score**: A harmonic mean of precision and recall, providing a single metric for model performance.

### Application of Results
This model can be deployed as a learning tool, a benchmark for further research, or a foundation for developing more sophisticated image recognition systems.

### ROI and Benefits
Achieving high accuracy and interpretability provides several benefits, which translate into tangible returns on investment:

#### Educational Value:

- **ROI:** Enhances the educational experience by providing students and researchers with a robust and accurate model to study. This can lead to improved learning outcomes and more effective training programs.
Benefit: A high-performing model helps users gain practical insights into advanced image classification techniques, potentially accelerating learning and research productivity.
Research and Development:

- **ROI:** The model serves as a benchmark for further research and development. By exceeding the 90% accuracy threshold, it provides a solid foundation for future improvements and innovations in image recognition technologies.
- **Benefit:** Researchers can use the model's insights to develop new methods or refine existing ones, contributing to advancements in the field and potentially leading to new research opportunities or publications.
Industry Applications:

- **ROI:** High accuracy and interpretability are crucial for deploying models in real-world applications, where they can improve decision-making processes and customer experiences. For example, a highly accurate image classification model can be used in quality control, autonomous vehicles, or medical imaging.
- **Benefit:** Accurate and interpretable models increase trust and reliability in automated systems, leading to better adoption and customer satisfaction. This can result in cost savings, increased efficiency, and competitive advantage in the market.
Operational Efficiency:

- **ROI:** By optimizing the model for efficiency, the project ensures that training and inference times are minimized, reducing computational costs and resource consumption.
- **Benefit:** Faster model performance translates into quicker insights and decisions, making the system more responsive and effective in dynamic environments.


[Back to top](#table-of-contents)

---

## The rationale to map the business requirements to the Data Visualizations and ML tasks

#### 1. Dataset Exploration

- **Requirement:** Conduct a thorough analysis of image characteristics and distribution across classes.
- **Rationale:** Understanding the dataset’s distribution and characteristics is crucial for designing effective preprocessing and model training strategies. Visualization tools can help reveal class imbalances, image quality issues, and other insights that inform data preprocessing and model development.

#### 2. Model Development

- **Requirement:** Create and optimize a CNN model to accurately classify images into one of the 10 CIFAR-10 categories.
- **Rationale:** Developing and optimizing the CNN model is central to meeting the project's goal of high accuracy. The ML tasks here include experimenting with different CNN architectures, tuning hyperparameters, and implementing regularization techniques. Performance metrics like accuracy, precision, recall, and F1 score will be used to evaluate and refine the model.

#### 3. Educational Support

- **Requirement:** Develop an interactive platform to aid students, researchers, and AI enthusiasts in understanding and applying deep learning techniques.
- **Rationale:** To support educational goals, data visualizations and interactive features are essential. The dashboard will include:
    - Data Visualizer: To illustrate the dataset’s characteristics.
    - Model Performance: To show the accuracy and other metrics of the trained model.
    - Image Classifier: To provide hands-on experience by allowing users to upload images and see real-time predictions.

#### 3. High Accuracy

- **Requirement:** Achieve over 90% accuracy on the CIFAR-10 test set.
- **Rationale:** High accuracy is a critical success criterion for the project. Visualizations like confusion matrices and ROC curves will help understand the model’s performance across different classes and identify areas for improvement.

#### 4. Interpretability

- **Requirement:** Offer clear visualizations and insights into model predictions.
- **Rationale:** Providing insights into how the model makes predictions will enhance its usability and trustworthiness. (Visualizations such as class activation maps or saliency maps will help users understand which parts of the images influence the model’s decisions.)

#### 5. Efficiency

- **Requirement:** Ensure fast training and inference for practical use.
- **Rationale:** Efficient models are essential for real-time applications and educational settings. Visualization of training and inference times can help monitor and optimize the model’s performance.



## ML Business Case

#### 1. Dataset Exploration

- **ML Task:** Perform exploratory data analysis (EDA) and visualize the distribution of image classes and characteristics.
- **Business Case:** Understanding the dataset’s distribution informs preprocessing and model training strategies, ensuring the model is trained on a representative sample.

#### 2. Model Development

- **ML Task:** Design, train, and optimize a CNN model using the CIFAR-10 dataset. Implement techniques such as data augmentation, dropout, and batch normalization.
- **Business Case:** Developing a high-performing CNN model meets the client’s requirement for accuracy and provides a robust educational tool. Performance metrics will guide the optimization process and ensure the model exceeds the accuracy threshold.

#### 3. Educational Support

- **ML Task:** Develop an interactive dashboard with features like data visualization, model performance tracking, and an image classifier.
- **Business Case:** The interactive platform supports educational goals by offering hands-on experiences and insights into the model’s behavior. It allows users to learn about CNNs and explore their applications in real-time.

#### 4. High Accuracy

- **ML Task:** Evaluate the model’s performance using accuracy, precision, recall, and F1 score. Implement iterative improvements based on performance metrics.
- **Business Case:** Achieving high accuracy ensures the model is effective for practical applications and meets the client’s standards. Detailed performance metrics provide transparency and help in further refinement.

#### 5. Interpretability

- **ML Task:** Generate visualizations such as confusion matrices and saliency maps to explain model predictions.
- **Business Case:** Clear visualizations enhance the model’s usability and trustworthiness, making it easier for users to understand how predictions are made and identify potential biases or errors.

#### 6. Efficiency

- **ML Task:** Optimize the model for faster training and inference times. Monitor performance metrics to ensure efficiency.
- **Business Case:** Ensuring the model is efficient is critical for real-time applications and educational settings. Efficient models facilitate quick experimentation and practical use, meeting the client’s requirements for a usable educational tool.


## Model Development

### Model Development Diagram
![Model Development Diagram](link-to-diagram-image)

### Technical Setup

#### CNN Architecture
- **Input Layer**: 32x32 RGB images are fed into the model.
- **Convolutional Layers**: Multiple layers with ReLU activation to extract spatial features, followed by max-pooling to reduce dimensionality.
- **Dense Layers**: Fully connected layers that consolidate the features and output a softmax probability distribution for classification.
- **Regularization**: Dropout layers are used to prevent overfitting by randomly omitting neurons during training.
- **Batch Normalization**: Applied to accelerate and stabilize the training process.

#### Image Preprocessing
- **Rescaling**: Images are scaled to a 0-1 range to improve model training efficiency.
- **Data Augmentation**: Techniques like random rotations, flips, and zooms are applied to enhance model robustness.

#### Model Compilation
- **Optimizer**: Adam optimizer is chosen for its adaptive learning rate capabilities.
- **Loss Function**: Categorical cross-entropy is used, given its suitability for multi-class classification tasks.
- **Metrics**: Accuracy is the primary evaluation metric, reflecting the model’s classification performance.

#### Training and Validation
The model undergoes training with early stopping and checkpointing to prevent overfitting. The best model is selected based on validation performance and is then tested on a separate test set to evaluate its generalization capability.

[Back to top](#table-of-contents)

---

## Dashboard Design

### Dashboard Pages:

1. **Project Summary**: Overview of the project and objectives.
2. **Data Visualizer**: Visual representation of the dataset and preprocessing steps.
3. **Model Performance**: Detailed performance metrics, including accuracy and confusion matrix.
4. **Image Classifier**: Allows users to upload images and receive real-time predictions.
5. **Hypothesis and Results**: Discussion of the hypothesis, results, and future work.

The Streamlit Dashboard serves as the user interface, allowing for easy interaction and access to the model’s capabilities. It has 5 pages:
    <details><summary>1st Page - Project Summary</summary>
    <br><br>
    This page offers a summary of the project, what to expect going to the next pages and presents the 2 business requirements:
    <img src="" width="80%"       
    height="80%"><br></details>
    <br><br>
    <details><summary>2st Page - Data Visualizer</summary>
    <br><br>
    This page shows the type of data that will be worked on:
    <img src="" width="80%"       
    height="80%"><br></details>
    <br><br>
    <details><summary>3st Page - Model Performance</summary>
    <br><br>
    This page goes into the details of the model's performance, like accuracy percentage:
    <img src="" width="80%"       
    height="80%"><br></details>
    <br><br>
    <details><summary>4st Page - Image Classifier</summary>
    <br><br>
    This page allows users to upload pictures and make live predictions:
    <img src="" width="80%"       
    height="80%"><br></details>
    <br><br>
    <details><summary>5st Page - Hypothesis and Inaccuracies</summary>
    <br><br>
    This page elaborates about the hypothesis of the project and also points out some things to take into consideration when using this 
    A.I model:<br>
    <img src="" width="80%"       
    height="80%"><br></details>


[Back to top](#cifar-10-image-classification-project)

---

## Kanban Board

### User Stories

- **Data Exploration**: Users can explore the dataset to understand class distributions and image characteristics.
- **Model Interaction**: Users can interact with the model by uploading images and viewing predictions.
- **Performance Monitoring**: Users can monitor model performance through detailed metrics and visualizations.
- **Hypothesis Testing**: Users can review the hypothesis and evaluate the model's success in meeting its goals.

### Project Status

As of the last update, all user stories have been successfully implemented, and the project is complete.

[Back to top](#table-of-contents)

---

## Unfixed Bugs

There are no known unfixed bugs at this time.

[Back to top](#table-of-contents)

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


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* Thank the people that provided support through this project.

