import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None

credit_risk=pd.read_csv("C:/Users/li/Desktop/credit risk/CreditRiskModelingInPython/UCI_Credit_Card.csv")#download the data
df= credit_risk.copy()

# As we seen Column ID has no meaning here so, we will remove it
df.drop(["ID"], axis=1, inplace= True) #axis=1 -- column removal and inplcae= True --means change in the original data

# Independnet features
X = df.drop(['default.payment.next.month'], axis=1)
# Dependent feature
y = df['default.payment.next.month']
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X= scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LogisticRegression
logit= LogisticRegression()
logit.fit(X_train, y_train)
# Predicting the model
pred_logit= logit.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, plot_confusion_matrix, plot_precision_recall_curve

print("The accuracy of logit model is:", accuracy_score(y_test, pred_logit))
print(classification_report(y_test, pred_logit))
# Plot confusion metrics
plot_confusion_matrix(logit, X_test, y_test, cmap="Blues_r")
plt.show()

# plot roc_auc curve
plot_precision_recall_curve(logit,X_test,y_test)
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
# Fitting the model
rf.fit(X_train,y_train)
# Predicting the model
pred_rf= rf.predict(X_test)
print("The accuracy of Random Forest Classifier is:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test,pred_rf ))

from sklearn.model_selection import RandomizedSearchCV
random_search=RandomizedSearchCV(xgb_clf,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
print("fitting the RandomizedSearchCV")
random_search.fit(X_train,y_train)
# Finding the best estimators
print(random_search.best_estimator_)
# Finding the best param
print(random_search.best_params_)
classifierXGB=xgb.XGBClassifier(objective='binary:logistic',
                                  gamma=random_search.best_params_['gamma'],
                                  learning_rate=random_search.best_params_['learning_rate'],
                                  max_depth=random_search.best_params_['max_depth'],
                                  reg_lambda=random_search.best_params_['reg_lambda'],
                                  min_child_weight=random_search.best_params_['min_child_weight'],
                                  subsample=random_search.best_params_['subsample'], 
                                  colsample_bytree=random_search.best_params_['colsample_bytree'],
                                  use_label_encoder=False)
# Fitting the model
classifierXGB.fit(X_train,y_train)
# Predicting model
y_pred= classifierXGB.predict(X_test)

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifierXGB,X,y,cv=10)
print(f"\n\nCross-Validation Scores : {score}")
print(f"Mean of the scores:{score.mean()}")


      

