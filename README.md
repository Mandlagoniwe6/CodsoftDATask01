#Overview: 
This project analyzes the famous Titanic dataset to build predictive models that classify whether a passenger survived or not. Using various data science techniques, 
the project explores which factors influenced passenger survival and evaluates the performance of multiple machine learning algorithms to identify the best approach.

#Project Goals: 
Analyze the Titanic dataset for insights into survival patterns.
Pre-process data by handling missing values, encoding categorical data, and scaling numerical features.
Build and compare machine learning models (Logistic Regression and Random Forest) for predicting passenger survival.
Fine-tune model hyperparameters and evaluate each model's performance.

#Dataset: 
The Titanic dataset contains data on 891 passengers, including features such as:
Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
Sex: Gender of the passenger
Age: Age in years
SibSp: Number of siblings/spouses aboard
Parch: Number of parents/children aboard
Fare: Ticket fare price
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
The target variable is Survived, where 1 indicates survival and 0 indicates non-survival.


#Project Structure:
Titanic-Dataset.csv: The dataset file used in this project.
titanic_analysis.py: Main Python script with the entire analysis and modeling pipeline.
README.md: Project documentation and usage instructions.


#Methodology: 
Data Preprocessing:
Imputed missing values for Age and Embarked.
Dropped irrelevant columns (e.g., Cabin, Name, Ticket, PassengerId).
Encoded categorical features and scaled Age and Fare for model compatibility.


#Modeling:
Built Logistic Regression and Random Forest models.
Used GridSearchCV to tune hyperparameters and identify the optimal model configuration.
Compared models using accuracy, classification reports, ROC-AUC curves, and confusion matrices.


#Evaluation:
Analyzed feature importance for key predictors of survival.
Visualized results through plots.
Assessed models on both training and test datasets to ensure performance reliability.

#Results: 
Logistic Regression: Achieved an accuracy of approximately 0.81.
Random Forest: Achieved an accuracy of approximately 0.83 and provided additional insights into feature importance.
Random Forest generally performed better on cross-validation, showing its suitability for this dataset.

#Summary of Findings According To How I Analyzed The Data
Factors Influencing Survival:
Sex: Women had a significantly higher survival rate than men. Sex was one of the most influential factors, 
with females being more likely to survive.
Age: Younger passengers, especially children, had higher survival rates, as age played a role in survival chances.
Socio-Economic Class (Pclass): Passengers in higher classes, especially those in first class, had better survival rates. 
This aligns with the prioritization during evacuation.
Fare: Passengers who paid higher fares, often correlating with higher socio-economic classes, tended to survive at higher rates.
Siblings/Spouses (SibSp): Passengers with a small number of family members on board (like one or two siblings or spouses) 
showed a slightly higher survival rate.

#Visualizations: 
The analysis includes several visualizations to illustrate key findings:
Survival by Gender and Class: Displaying counts and proportions of survival.
Feature Importances: Bar chart showing the impact of each feature on model predictions.
ROC Curves: Comparing true positive and false positive rates for Logistic Regression and Random Forest models.

#Conclusion: 
This project successfully utilized Logistic Regression and Random Forest models to predict Titanic survival outcomes. 
The Random Forest model proved to be more effective, highlighting the importance of feature selection and tuning.

#Acknowledgments: 
Special thanks to Kaggle for the dataset and to CodSoft for the data science internship opportunity to explore practical machine learning applications.

