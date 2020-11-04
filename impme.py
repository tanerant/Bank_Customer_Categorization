#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import time
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.svm import SVR




class Basic_Information():

  
    def __init__(self, data):
        self.data=data
    
    def Info(self):
        
        print("\nHead\n")
        print(self.data.head())
       
        print("\nDescribe\n")
        print(self.data.describe().T)
        
        print("\nIndex, Columns\n")
        print(self.data.shape)
        
        print("\nData Info\n")
        print(self.data.info())
        
        print("\nMissing Value Sum per Variables\n")
        print(self.data.isnull().sum()) 
        
        print("\nMissing Value Sum\n")
        print(self.data.isnull().sum().sum())
        
        print("\nMissing Value Heatmap\n")
        
        
        import seaborn as sns
        
        sns.heatmap(self.data.isnull(),yticklabels = False,cbar = False)


class Missing():

    def __init__(self, data):
        self.data=data
        
    def meann(self,col):
        
        self.data[col].fillna(value=self.data[col].mean(), inplace=True)
        print(self.data.isna().sum())
        
    def zero(self,col):
        
        self.data[col].fillna(value = 0 , inplace=True)
        print(self.data.isna().sum())
        
    def cat(self,col):   
        
        self.data[col] = self.data[col].replace(np.nan, "Other")
        print(self.data.isna().sum())
    

    
class PreProcess():

    def __init__(self, data):
        self.data=data
        
        
        
    def crosstab(self,target,col):
        import scipy.stats 
        cross=pd.crosstab(index=self.data[target], columns=self.data[col],margins=True)
        print(cross)
        
        chi2,p,dof,expected= scipy.stats.chi2_contingency(cross)
        results = [
        ['Chi-Square test', chi2],
        ['P - value', p]
                ]
        print(results)  
    
    def groupbi(self,target,col1):
        group = self.data.groupby(target)[col1].mean()
        print(group)
        
       
    def shap(self,col):
        
        from scipy.stats import shapiro
        stat, p = shapiro(self.data[col])
        print(col)
        print(stat, p)
        
        alpha = 0.05
        if p > alpha:
            print('Sample comes from normal (Gaussian) distribution (Fail to Reject H0)')
        else:
       
            print('Sample does not come from normal (Gaussian) distribution (reject H0)')   


class Vis():
    
    def __init__(self, data):
            self.data=data
        
    def hist(self):
        
        import matplotlib.pyplot as plt
        

        for i in self.data.columns:
            fig=self.data[i].plot.hist()
            fig.set_title(i)
            plt.show()
            
    def joinplot(self,target):
        
        import seaborn as sns
        
        for i in self.data.columns:
            
            sns.jointplot(self.data[i],self.data[target],kind='reg', data =self.data)
        
    def corrheat(self,target,corr_value):
        
        corr = self.data.corr()

        kot = corr[corr>=corr_value]
        plt.figure(figsize=(12,8))
        sns.heatmap(kot, cmap="Greens")
        
        
        print(corr[target].sort_values(ascending=False))
        
    def scatter(self,target):
        
        
        for x in self.data.columns:
            
            fig, ax = plt.subplots()
    
            ax.scatter(self.data[x], self.data[target])
            
            ax.set_title(target)
            ax.set_xlabel(x)
            ax.set_ylabel(target)
                
        
class Model():
    
    def __init__(self, data):
            self.data=data
    
    
    
    
    def pcaa(self,X_train= None,X_test= None):
        
        
        
            pca = PCA()
        
            X_reduced_train = pca.fit_transform(scale(X_train))
            X_reduced_test = pca.fit_transform(scale(X_test))
            
            print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100))
        
            import matplotlib.pyplot as plt
            
            features = range(pca.n_components_)
            plt.bar(features, pca.explained_variance_ratio_, color='black')
            plt.xlabel('PCA features')
            plt.ylabel('variance %')
            plt.xticks(features);
            
            return X_reduced_train,X_reduced_test
        
    def catmodel(self,X_train= None,X_test= None,y_train= None,y_test= None,cat_model=None):
             
            start = time.process_time()
                      
            cart_model = cat_model.fit(X_train,y_train)
           
            print("Time  :" ,time.process_time() - start)
            
            y_pred = cart_model.predict(X_test)
            
            print("Confusion Matrix  :") 
            
            print(confusion_matrix(y_test, y_pred))
            
            print("Accuracy  :" , accuracy_score(y_test,y_pred))
              
            print("CV Score  :" , cross_val_score(cart_model,X_test,y_test,cv=10).mean())
           
            print("Model Report : ")
            
            print(classification_report(y_test,y_pred))
        
        
            from sklearn.metrics import roc_auc_score,roc_curve
            import matplotlib.pyplot as plt
        
            logit_roc_auc =  roc_auc_score(y_test, cart_model.predict(X_test))
            
            fpr, tpr, thresholds = roc_curve(y_test, cart_model.predict_proba(X_test)[:,1])
            
            plt.figure()
            plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Oranı')
            plt.ylabel('True Positive Oranı')
            plt.title('ROC')
            plt.show()
      
            print("AUC :", logit_roc_auc)
    
    def Grid_CV(self,X_train= None,y_train = None,params= None,model=None):
            
            
            model_cv = GridSearchCV(model, params, cv=10, n_jobs = -1, verbose = 2)
            model_tuned = model_cv.fit(X_train, y_train)
            
            print(model_tuned.best_params_)
            
    
    def regmodel(self,X_train= None,X_test= None,y_train= None,y_test= None,reg_model=None):
        
            start = time.process_time()
            
            re_model =reg_model.fit(X_train, y_train)
            
            print("Time  :" ,time.process_time() - start)
            
            y_pred = re_model.predict(X_test)
            
            print( "RMSE  :", np.sqrt(mean_squared_error(y_test, y_pred)))
            
            print("r2_score : " , r2_score(y_test, y_pred))
            
        
        
        


