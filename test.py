import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

scaler = StandardScaler()

titanic_data = pd.read_csv("Titanic-Dataset.csv",header=0, sep =",")

titanic_data["Age"].fillna(titanic_data["Age"].median(),inplace= True)
titanic_data.drop(columns=["Cabin", "Name","Ticket","PassengerId"],inplace= True)
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)

titanic_data["Sex"] = titanic_data["Sex"].map({"male": 0, "female": 1})
titanic_data = pd.get_dummies(titanic_data, columns=["Embarked"], drop_first=True)
titanic_data = titanic_data.astype({col: 'int' for col in titanic_data.select_dtypes(include=['bool']).columns})

titanic_data[["Age", "Fare"]] = scaler.fit_transform(titanic_data[["Age", "Fare"]])

X = titanic_data.drop(columns=["Survived"])
y = titanic_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid_lr = {
    "C": [1],  
    "solver": ["lbfgs"],  
    "max_iter": [600]  
}
grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5)
grid_search_lr.fit(X_train, y_train)
print("Best Parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Cross-Validation Score for Logistic Regression:", grid_search_lr.best_score_)

best_logistic_reg = grid_search_lr.best_estimator_
y_pred_best_logistic = best_logistic_reg.predict(X_test)
print("Best Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_best_logistic))
print("Best Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_best_logistic))
print("Confusion Matrix for Best Logistic Regression")
print(confusion_matrix(y_test, y_pred_best_logistic))

param_grid_rf = {
    "n_estimators": [200],
    "max_depth": [None],
    "min_samples_split": [10]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best Cross-Validation Score for Random Forest:", grid_search_rf.best_score_)

best_random_forest = grid_search_rf.best_estimator_
y_pred_best_randomF = best_random_forest.predict(X_test)
print("Best Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_randomF))
print("Best Random Forest Classification Report")
print(classification_report(y_test, y_pred_best_randomF))
print("Confusion Matrix for Best Random Forest")
print(confusion_matrix(y_test, y_pred_best_randomF))

cv_scores_logistic = cross_val_score(best_logistic_reg, X, y, cv=10, scoring="accuracy")
print("Cross-validation Accuracy for Logistic Regression: %0.2f (+/- %0.2f)" % (cv_scores_logistic.mean(), cv_scores_logistic.std()))
cv_scores_random_forest = cross_val_score(best_random_forest, X, y, cv=10, scoring="accuracy")
print("Cross-validation Accuracy for Random Forest: %0.2f (+/- %0.2f)" % (cv_scores_random_forest.mean(), cv_scores_random_forest.std()))

feature_importances = pd.Series(best_random_forest.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind="bar", title="Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right", fontsize=10) 
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_best_logistic), annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix for Best Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_probs_lr = best_logistic_reg.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_probs_lr)
auc_score_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression (AUC = %0.2f)" % auc_score_lr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_best_randomF), annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix for Best Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_probs_rf = best_random_forest.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_probs_rf)
auc_score_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = %0.2f)" % auc_score_rf)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Random Forest")
plt.legend(loc="lower right")
plt.show()

sns.countplot(data=titanic_data, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.xlabel("Gender (0 = Male, 1 = Female)")
plt.ylabel("Count")
plt.legend(title="Survived", loc="upper right")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data, x="Fare", hue="Survived", kde=True, element="step", palette="magma", bins=30)
plt.title("Survival by Fare")
plt.xlabel("Fare")
plt.ylabel("Density")
plt.legend(title="Survived", labels=["Did Not Survive", "Survived"])
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data, x="Age", hue="Survived", kde=True, element="step", palette="viridis", bins=30)
plt.title("Survival by Age")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend(title="Survived", labels=["Did Not Survive", "Survived"])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_data, x="SibSp", hue="Survived", palette="viridis")
plt.title("Survival by Number of Siblings/Spouses Aboard")
plt.xlabel("Number of Siblings/Spouses Aboard (SibSp)")
plt.ylabel("Count")
plt.legend(title="Survived", loc="upper right")
plt.show()

sns.countplot(data=titanic_data, x="Pclass", hue="Survived")
plt.title("Survival by Socio-Economic Class")
plt.xlabel("Passenger Class (1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class)")
plt.ylabel("Count")
plt.legend(title="Survived", loc="upper right")
plt.show()

print("Summary of Findings:")
print("- Logistic Regression Best Parameters:", grid_search_lr.best_params_)
print("- Random Forest Best Parameters:", grid_search_rf.best_params_)
print("- Logistic Regression Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores_logistic.mean(), cv_scores_logistic.std()))
print("- Random Forest Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores_random_forest.mean(), cv_scores_random_forest.std()))
print("\nTop 5 Most Important Features (Random Forest):")
print(feature_importances.sort_values(ascending=False).head(5))

#Summary of Findings According To How I Analyzed The Data
#Factors Influencing Survival:
#Sex: Women had a significantly higher survival rate than men. Sex was one of the most influential factors, 
#with females being more likely to survive.
#Age: Younger passengers, especially children, had higher survival rates, as age played a role in survival chances.
#Socio-Economic Class (Pclass): Passengers in higher classes, especially those in first class, had better survival rates. 
#This aligns with the prioritization during evacuation.
#Fare: Passengers who paid higher fares, often correlating with higher socio-economic classes, tended to survive at higher rates.
#Siblings/Spouses (SibSp): Passengers with a small number of family members on board (like one or two siblings or spouses) 
#showed a slightly higher survival rate.

#Model Selection:
#After comparing Logistic Regression and Random Forest models, I chose Random Forest as the final model. 
#I based this decision on Random Forest's superior cross-validation accuracy and the additional insights it provided through feature importance.
#The Random Forest model effectively captured the non-linear relationships among features, particularly useful given 
# the categorical variables like gender and socio-economic class, and provided higher accuracy and AUC scores.
