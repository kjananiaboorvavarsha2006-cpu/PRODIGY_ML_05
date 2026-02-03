# PRODIGY_ML_05
🍽️ Food Image Recognition & Calorie Estimation

Task-05 | Machine Learning Project

📌 Project Overview

This project focuses on building a Machine Learning model that can recognize food items from images and estimate their calorie content.
The system helps users track dietary intake and make informed food choices by combining image classification with nutritional estimation.

The model is trained on the Food-101 dataset, which contains 101 different food categories.

🎯 Objectives

Recognize food items from input images

Classify food into one of 101 categories

Estimate calorie content based on predicted food class

Apply supervised learning using CNN and transfer learning

🧠 Machine Learning Approach

Learning Type: Supervised Learning

Model Used: Convolutional Neural Network (CNN)

Technique: Transfer Learning (MobileNetV2)

Loss Function: Sparse Categorical Cross-Entropy

Evaluation Metric: Accuracy

📂 Dataset

Name: Food-101

Source: Kaggle

Link: https://www.kaggle.com/datasets/dansbecker/food-101

Details:

101 food categories

101,000 images

750 training images per class

250 testing images per class

🛠️ Technologies Used

Python

TensorFlow & Keras

TensorFlow Datasets

NumPy

Pandas

Matplotlib

OpenCV

⚙️ Project Workflow

Load and preprocess the Food-101 dataset

Resize and normalize images

Apply data batching and prefetching

Build CNN using transfer learning

Train and validate the model

Predict food category from image

Estimate calories using nutritional mapping

🔥 Calorie Estimation Method

The Food-101 dataset does not include calorie values.
After food classification, calorie estimation is done using a predefined calorie mapping dictionary based on standard nutritional references.

Example:

Pizza → 266 kcal
Burger → 295 kcal
Salad → 152 kcal

🧪 Sample Output

Predicted Food: Pizza

Estimated Calories: 266 kcal

The output displays the food name along with its estimated calorie content.

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/your-repo-name.git


Install required libraries

pip install tensorflow tensorflow-datasets numpy pandas matplotlib


Run the Jupyter Notebook

jupyter notebook

📈 Future Enhancements

Portion size estimation

Regression-based calorie prediction

Mobile/Web application integration

Real-time food detection

🏁 Conclusion

This project demonstrates the application of machine learning and deep learning techniques to solve a real-world problem in health and nutrition.
By combining image classification with calorie estimation, the system provides a practical solution for dietary tracking.
