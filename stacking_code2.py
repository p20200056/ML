import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure
from matplotlib.figure import Figure
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import time

df = pd.read_csv('Features_top3.csv', header=None)
X = df.iloc[:, :-2]
y = df.iloc[:,-1]
scaler = StandardScaler()
sensitivity=0
X=scaler.fit_transform(X)
epoch = 50
accuracy_list=[None]*epoch
fn_list=[None]*epoch
fp_list=[None]*epoch
tn_list=[None]*epoch
tp_list=[None]*epoch
accuracy_avg=0
for i in range(epoch):
    estimator = [('rf',RandomForestClassifier(n_estimators=170)),('svm',SVC(C=10, random_state=0, probability=True)),('ada',AdaBoostClassifier()),('xgb',xgb(n_neighbors=9))]
    clf = StackingClassifier(estimators=estimator, final_estimator=LogisticRegression())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, train_size=.7,random_state=None)
    sm = SMOTE(random_state = 0)
    X_train,y_train = sm.fit_resample(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred_clf = clf.predict(X_test)
    accuracy_list[i]=100*accuracy_score(y_test, y_pred_clf)
    accuracy_avg=accuracy_avg + accuracy_list[i]
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred_clf).ravel()
    accuracy_list[i]=100*accuracy_score(y_test, y_pred_clf)
    tn_list[i]=tn
    fp_list[i]=fp
    fn_list[i]=fn
    tp_list[i]=tp
    sensitivity=sensitivity+(tp/(tp+fn))*100 
    if i%10==0 :
        print(100*accuracy_score(y_test, y_pred_clf))
        print(100*(tp/(tp+fn)))
accuracy_avg = accuracy_avg/epoch
sensitivity_avg = sensitivity/epoch
dict = {'accuracy':accuracy_list,'tn':tn_list,'fp':fp_list,'fn':fn_list,'tp':tp_list}
df2 = pd.DataFrame(dict)
df2.to_csv('trial_smote_new.csv') 
print(accuracy_avg)
print(sensitivity_avg)
print(100-sensitivity_avg)