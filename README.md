# Bank_Customer_Categorization
Historical data were gathered from bank customers to determine whether a customer is a good or bad credit risk for a home equity loan. Bad risk customers are more likely to default on the loan.

**Packages:** pandas, numpy, sklearn, matplotlib, seaborn.

## Preprocess 
* First, I converted the target variable type to an object.
* I detected missing values

![alt text](https://github.com/tanerant/Bank_Customer_Categorization/blob/main/misssing.png "Missing values")

* I filled the missing values with average and most frequent values.

## EDA
* I looked at the relationship of categorical variables with target variable.
* I grouped the target value according to the mean of the numerical values.Then I separated the different ones.
* Shapiro-wilks test was performed to check normality assumption.(All Non normal distributed).
* Then I grouped them and translated them into a categorical variable. Thus, I was able to test variables that we know to affect the target variable.
* I paid attention to the homogeneous distribution of the data and the meaning of the variable.
* I did a chi-square test on the variables I separated and all of them turned out to be related to the target variable.

If the number of delaying monthly payments is greater than 5, he / she is put in the bad risk group. The number of delaying payments per month is 5 and the value of properties of customers who are identified as good risk is different from others.

![alt text](https://github.com/tanerant/Bank_Customer_Categorization/blob/main/data_viz.png "data_viz")

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using Accuracy and F1 score.  

## Only debtinc_score model

* I achieved 87% accuracy in the logistic regression model that I created with only debt-income ratio.

* So if we know only the debt-income ratio of a new customer, we can find out that this customer is either a good risk or a bad risk with 87% accuracy.

![alt text](https://github.com/tanerant/Bank_Customer_Categorization/blob/main/debtinc_score.PNG "debtinc_score")

* Deptinc_score stands out in the feature importance chart.

![alt text](https://github.com/tanerant/Bank_Customer_Categorization/blob/main/rf_feature_imp.png "Feature importance")

## Model performance
XGB model  outperformed the other approaches. 
![alt text](https://github.com/tanerant/Bank_Customer_Categorization/blob/main/models_performances.PNG "Model Performances")

## Summary
* When examining the model performance, f-score (1) (bad risk) was considered. We want to find the bad risk client.

