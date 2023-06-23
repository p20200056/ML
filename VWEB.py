import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure
from matplotlib.figure import Figure    
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import time 

epoch = 500

knn = []
svm = []
lr = []
rf = []
ens = []
ada = []
xgbm = []
knn_t = []
svm_t = []
lr_t = []
rf_t = []
ens_t = []
ada_t = []
xgbm_t =[]
files = [ "Fea"]
#files = ["Features"]



for file in files:
    print(file)
    df = pd.read_csv(f'{file}.csv', header=None)
    X = df.iloc[:, :-2]
    y = df.iloc[:,-1]
    X = scaler.fit_transform(X)
    X=scaler.fit_transform(X)
    accuracy_knn=0
    accuracy_rf=0
    accuracy_nb=0
    accuracy_lr=0
    accuracy_svm=0
    accuracy_knn=0
    accuracy_ens=[0]*(512)
    accuracy_ada=0
    accuracy_xgbm=0
    
    time_knn_t=0
    time_svm_t=0
    time_lr_t=0
    time_nb_t=0
    time_rf_t=0
    time_ens_t=0
    time_ada_t=0
    accuracy_list=[None]*epoch
    Latency_list=[None]*epoch
    sensitivity_list=[None]*epoch
    specificity_list=[None]*epoch
    fn_list=[None]*epoch
    fp_list=[None]*epoch
    tn_list=[None]*epoch
    tp_list=[None]*epoch
    accuracy_avg=0
    time_xgbm_t=0
    sm = SMOTE(random_state = 0)
    #X,y = sm.fit_resample(X, y)
    for m in range(epoch):
        X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X, y, test_size=0.2, random_state=None)
        
        y_test = y_test_5.to_numpy()
        accuracy_sum = [None]*(y_test.size)
        iteration=0

        classifier_knn = KNeighborsClassifier(n_neighbors=9)
        classifier_knn.fit(X_train_5, y_train_5)
        time_knn_s = time.time()
        y_pred_knn = classifier_knn.predict(X_test_5)
        time_knn_e=time.time()
        time_knn=(time_knn_e-time_knn_s)
        time_knn_t=time_knn_t+time_knn
        accuracy_knn = accuracy_knn + (100*accuracy_score(y_test_5, y_pred_knn))

        classifier_svm = SVC(C=10, random_state=0, probability=True)
        classifier_svm.fit(X_train_5, y_train_5)
        time_svm_s = time.time()
        y_pred_svm = classifier_svm.predict(X_test_5)
        time_svm_e = time.time()
        time_svm = time_svm_e - time_svm_s
        time_svm_t += time_svm
        accuracy_svm = accuracy_svm + 100*accuracy_score(y_test_5, y_pred_svm)

        classifier_lr = LogisticRegression()
        classifier_lr.fit(X_train_5, y_train_5)
        time_lr_s=time.time()
        y_pred_lr = classifier_lr.predict(X_test_5)
        time_lr_e=time.time()
        time_lr=(time_lr_e-time_lr_s)
        time_lr_t=time_lr_t+time_lr
        accuracy_lr = accuracy_lr + 100*accuracy_score(y_test_5, y_pred_lr)

        classifier_rf = RandomForestClassifier(n_estimators=170)
        classifier_rf.fit(X_train_5, y_train_5)
        time_rf_s=time.time()
        y_pred_rf = classifier_rf.predict(X_test_5)
        time_rf_e=time.time()
        time_rf=(time_rf_e-time_rf_s)
        time_rf_t=time_rf_t+time_rf
        accuracy_rf = accuracy_rf + 100*accuracy_score(y_test_5, y_pred_rf)
        
        classifier_ada = AdaBoostClassifier(n_estimators=170, random_state=0)
        classifier_ada.fit(X_train_5, y_train_5)
        time_ada_s=time.time()
        y_pred_ada = classifier_ada.predict(X_test_5)
        time_ada_e=time.time()
        time_ada=(time_ada_e-time_ada_s)
        time_ada_t=time_ada_t+time_ada
        accuracy_ada = accuracy_ada + 100*accuracy_score(y_test_5, y_pred_ada)

        classifier_xgbm = xgb.XGBClassifier(n_estimators=170, max_depth=8, learning_rate=0.1, subsample=0.5)
        classifier_xgbm.fit(X_train_5, y_train_5)
        time_xgbm_s=time.time()
        y_pred_xgbm = classifier_xgbm.predict(X_test_5)
        time_xgbm_e=time.time()
        time_xgbm=(time_xgbm_e-time_xgbm_s)
        time_xgbm_t=time_xgbm_t+time_xgbm        
        accuracy_xgbm = accuracy_xgbm + 100*accuracy_score(y_test_5, y_pred_xgbm)
        
        
        time_ens_s=time.time()
        accuracy_sum = [0]*(y_test.size)
        for l in range(y_test_5.size):        
            checker = ((30*y_pred_xgbm[l])/10) + ((25*y_pred_rf[l])/10) + ((10*y_pred_ada[l])/10)
            if (checker >= 3.5):
                 accuracy_sum[l] = 1     
            else :
                 accuracy_sum[l] = 0
                 
            checker =0   
        time_ens_e=time.time()    
        Latency_list[m]=(time_ens_e-time_ens_s)+time_xgbm+time_rf+time_ada            
        accuracy_list[m]=100*accuracy_score(y_test_5, accuracy_sum)
        tn, fp, fn, tp = confusion_matrix(y_test_5,accuracy_sum).ravel()
        tn_list[m]=tn
        fp_list[m]=fp
        fn_list[m]=fn
        tp_list[m]=tp
        sensitivity_list[m]=tp/(tp+fn)
        specificity_list[m]=tn/(tn+fp)
        print(m)
 
dict = {'accuracy':accuracy_list,'tn':tn_list,'fp':fp_list,'fn':fn_list,'tp':tp_list,'Latency':Latency_list,'Sensativity':sensitivity_list,'specificity':specificity_list}
df2 = pd.DataFrame(dict)
df2.to_csv('VWEB_Top3_cosupi_80-20.csv') 
