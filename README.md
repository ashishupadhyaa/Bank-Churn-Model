# <ins> Bank Churn Model</ins>

> Using past bank churn data predicting future bank churn rate.

I used [kaggle data](https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers "Data for bank churn") to make *LogisticRegression*, *RandomForest* and *XGBClassifier* model and predicting the bank churn rate.

## EDA

- The distribution of data between *Exited=0* and *Exited=1* is:

![alt text](/image/data_dist.png)

- The *hist plots* between *Continuous variables* and *Categorical Variables* are as follows:

![alt text](/image/cat_cont_plot.png)

- The *box plots* between *Continuous variables* are as follows:
![alt text](/image/cont_box.png)

- The plot between *Geography* and *Age* with *hue=Exited* is as follows:

![alt text](/image/swarm.png)

- *Correlation* between *Continuous variables* area s follows:

![alt text](/image/cor.png)

### Logistic Regression Model

 - I used **GridSearchCV** to decide best ***estimator*** for *LogisticRegression* model and the <ins>best_estimator</ins> was:

```python
LogisticRegression(C=50, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=300, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
          tol=0.0001, verbose=0, warm_start=False)
```

The ***Classification_report*** of the model was:

![alt text](/image/lr_report.png)


### Random Forest Classifier Model

 - I used **GridSearchCV** to decide best ***estimator*** for *RandomForestClassifier* model and the <ins>best_estimator</ins> was:

```python
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=8, max_features=7, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
```

The ***Classification_report*** of the model was:

![alt text](/image/rf_report.png)

### XGBClassifier Model

 - I used **GridSearchCV** to decide best ***estimator*** for *XGBClassifier* model and the <ins>best_estimator</ins> was:

```python
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0.01,
       learning_rate=0.05, max_delta_step=0, max_depth=6,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
```

The ***Classification_report*** of the model was:

![alt text](/image/xgb_report.png)

### The ROC_Curve of the models:

![alt text](/image/roc_curve.png)

**<ins>The End</ins>**
