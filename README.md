# Science Reach Project 2025: Testing the Effect of Various Machine Learning Algorithms on Resultant Fraud Detection Accuracy

Our experiment performs a comparative analysis of various machine learning models and their accuracy on fraud detection–as measured through recall and success rate- with the goal of improving fraud prevention capabilities in modern digital security systems. Millions of Americans are victims of financial fraud, and our paper aims to analyze existing fraud detection systems using various machine learning techniques and create a framework for our own system which will be available for public use. 

Procedure:
A synthetic dataset of financial transactions was obtained from Kaggle, a data science platform. This dataset was collected, cleaned, preprocessed, and then loaded into DataRobot, a modeling and testing software for machine learning model blueprints. Through DataRobot, we were able to classify the data to efficiently run a comparative analysis of various supervised ensemble models, deep learning models, neural networks, and more. Each model’s accuracy of fraud detection was calculated through confusion matrices–this valuable information provided us with a complete analysis of how the model learned and evaluated each transaction. Confidence interval analysis of the result was then performed, which allowed us to assess the reliability, validity, and robustness of each model.

Extension:

Upon receiving and understanding the results, we developed our own machine learning model using the excellent LightGBM gradient boosting framework and the Optuna optimization library. Our model performed at a 99.98% accuracy level. Due to the possible risk of overfitting, we used early stopping mechanisms to ensure that the model didn't rely too much on the training data. Further plans involve potential involvement of quantum machine learning methods to implement real-time detection and help uncover more subtle and unique patterns that classical computers do not have the capability to figure out yet.


