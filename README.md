# Identifying Racial bias: A Machine Learning approach

A Machine Learning based approach in designing a predictive model to investigate the underlying relationship between the various crime indicators and the arrests made. The deployed model attempts to establish a mathematical relationship highlighting the racial profile in the NYPD's stop-and-frisk program

## TABLE OF CONTENTS

* [Motivation](#Motivation)
* [Objective](#objective)
* [Data](#data)
* [Technologies](#technologies)
* [Algorithms](#Algorithm)
* [Implementation](#implementation)
* [Results](#results)
* [References](#references)

## MOTIVATION
With the increased number of incidents reported across the United States revealing brutality shown by the police against minority groups, there lies a scope in conducting a thorough statistical study of the various factors leading to an arrest. With that intent, this project attempts to verify the existence of racial bias in the stop-and-frisk initiative from NYPD by harnessing the capabilities of various statistical techniques and Machine Learning algorithms to model the relationship between various key features leading to the arrest.

## OBJECTIVE
 - The project attempts to study the NYPD stop-and-frisk data from the year 2015-16 with the intent of identifying whether there exists a significant relationship between race and the decision to arrest a stopped individual or not. 
 - Desiginging an array of Machine Learning algorithms to model the mathematical relationship between various predictor variables to accurately predict arrests
 - Tune the model hyperparameters to minimize the number of False positives in the predictions

## DATA
The stop-and-frisk Data is sourced from the open-source NYPD government website for the year of 2015-16 (https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page).

- Data consists of 34,967 instances of stops resulting from the stop-and-frisk practice conducted by on-duty police officers in and around the limits of New York city. 
- The feature set houses a wide variety of feature subsets capturing data regarding various elements. Locality based information like time, street name, area code, etc., crime-related features include weapons carried, contraband found, summons issued, suspect frisked, etc. The data also contains data elements describing the physical appearance of the suspect like height, weight, build, hair color, age, and race.
- The target variable of interest is arrest made which follows a binary distribution of Yes or No values.

## TECHNOLOGIES
Python - pandas, numpy, matplotlib, seaborn, sklearn, scikit learn
R - dplyr, tidyverse, psych, caret
MS Excel, Tableau

## ALGORITHMS AND TECHNIQUES
- Data Preparation: Exploratory Data Analysis, t-tests, chi-square tests, Data Visualization
- Statistical Modeling: Logistic Regression, Bernoulli Naive-Bayes model, k-nearest neighbors, L2-regularized Logistic Regression
- Model Tuning: L2-regularization, Stochastic Gradient Descent, Cross validation
- Model Evaluation: Confusion Matrix, ROC curve, PR curve, Cross Validation

## IMPLEMENTATION

### Data Exploration
An initial descriptive analysis of the data reveals a few key insights about the nature of the practice. Figure below shows some initial insights derived from the data including the bar chart of the number of individuals across different races stopped because of the practice.

- Distribution of arrest made across races
![arstmade (1)](https://user-images.githubusercontent.com/9445072/105079089-8c263900-5a54-11eb-9bba-0d5a2ce5b44a.JPG)


- Frisking seen across races

![frisk](https://user-images.githubusercontent.com/9445072/105079130-9cd6af00-5a54-11eb-974b-0c314b716b89.JPG)

- Arrests across races

![arstmade](https://user-images.githubusercontent.com/9445072/105079153-a5c78080-5a54-11eb-8449-c82a34695d7a.JPG)

Refer here for code: [Data Exploration & Feature Selection](https://github.com/abhilashhn1993/racial-bias-in-stop-and-frisk-program/blob/main/Code/Feature_Selection_with_statTests.ipynb)

### Data Modeling

#### K-nearest Neighbor
- Deployed a k-nearest neighbor initially as a base classifier with an optimal k-value of 20 optimized with cross validation.
- The baseline model provided a test accuracy of 80% with a precision of 52% and AUC of 0.6

#### Bernoulli Naive-Bayes model
- A Bernoulli Naïve Bayes designed on the Bernoulli distribution of all features with prior probabilities calculated
- The model yielded an Accuracy of 86.50%, Precision of 64.30% on the test data with an AUC of 0.73 in the PR-curve
Refer to this link for code: [Naive-Bayes Model](https://github.com/abhilashhn1993/racial-bias-in-stop-and-frisk-program/blob/main/Code/Naive-Bayes%20Model.ipynb)

#### Logistic Regression
- Final classifier, Logistic Regression model performed better compared to the other two models. 
- An Accuracy of about 88.4%, Precision of 74% and a 0.73 AUC value in the PR curve was observed on test data.

The Logistic Regression classifier designed gave the better prediction with minimal false positives. This model is chosen to be the best classifier and further tuned to improve the results

### Parameter Tuning
- Hyper parameter tuning by using stochastic gradient descent as well as Ridge regularization with a learning rate of 0.01 obtained from grid search

```python
model = SGDClassifier(loss="log", penalty="l1", max_iter=80)
```
```python
model.fit(X_train,y_train)
```

```python
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import parfit.parfit as pf

grid = {
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0], # learning rate
    #'n_iter': [1000], # number of epochs
    'loss': ['log'], # logistic regression,
    'penalty': ['l1','l2'],
    'n_jobs': [-1]
}
paramGrid = ParameterGrid(grid)
best_model, best_score, all_models, all_scores = pf.bestFit(SGDClassifier, paramGrid, 
     X_train, y_train, X_test, y_test, 
     metric=roc_auc_score, scoreLabel='AUC')
print(best_model)
```
- The model yielded an Accuracy of 89%, a Precision of 77% and an AUC of 0.78 in the PR curve
- Further modified the prediction thresholds of the model to minimize the False positives in predictions. On setting the threshold at 0.5 initially, an Accuracy of 89.96%, Precision of 78% and AUC of 0.79 was observed in the PR-curve
- With a further tuning of threshold at 0.6 which improved Precision to 85.10, with an Accuracy of 89.55 and an AUC-PR of 0.79

## RESULTS
Determining a list of key predictors of arrest with the aid of our best classifier was key in establishing the insignificance of Race as a key factor in decision to arrest. However, from the initial exploration of the data, we found that the people from Black community were stopped more often than individuals of other race raising the question about racial profiling. Our analysis revealed that there is no reason to stop and frisk individuals based on their race as the determining factors of arrest were related to presence of suspicious weapons, contrabands, etc.

The complete description and representation of results can be seen in the detailed report here: [Identification of Racial Bias: A Machine Learning approach](https://github.com/abhilashhn1993/racial-bias-in-stop-and-frisk-program/blob/main/Report/Racial-bias-in-arrests.pdf)

## REFERENCES
- JohnM. (2020, November 02). Police Violence & Racial Equity - Part 2 of 3. Retrieved December 08, 2020, from https://www.kaggle.com/jpmiller/police-violence-racial-equity
- Spitzer, E. (1999), “The New York City Police Department’s “Stop and Frisk” Practices,” Office of the New York State Attorney General; available at www.oag.state.ny.us/press/reports/stop_frisk/stop_frisk.html
- Kutner, D. (2020, March 17). Stop and Frisk - data visualization and analysis. Retrieved December 08, 2020, from https://medium.com/@daniellekutner/stop-and-frisk-data-visualization-and-analysis-504c9a41ab6c
- Brownlee, J. (2019, December 11). How To Implement Logistic Regression From Scratch in Python. Retrieved December 08, 2020, from https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python
- Epp, Charles R., Steven Maynard‐Moody and Donald P. Haider‐Markel. 2014. Pulled Over: How Police Stops Define Race and Citizenship. Chicago: University Chicago Press

