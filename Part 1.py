import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, auc ,confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


path = r'C:\Users\udit sharma\Downloads\Forest.xlsx'  # should change the path accordingly
rawdata = pd.read_excel(path)  # pip install xlrd
# print("data summary")
# print(rawdata.describe())
class_target=rawdata.iloc[:,0]
nrow, ncol = rawdata.shape
# print(nrow, ncol)
# print(rawdata['class'].unique())
label_encoder = preprocessing.LabelEncoder()
rawdata['class']= label_encoder.fit_transform(rawdata['class'])
# print(rawdata)
# ohe = ColumnTransformer([('anyname', OneHotEncoder(), [0])], remainder='passthrough')
# target=ohe.fit_transform(rawdata[['class']].values)


predictors=rawdata.iloc[:,1:ncol]
# print(predictors)
#print(predictors)
target=rawdata.iloc[:,0]
# print(target)
pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors,target,test_size=.3,shuffle=False)

split_threshold=3
fpr = dict()                  # store false positive rate in a dictionary object
tpr = dict()# likewise, store the true positive rate
roc_auc = dict()
colors=['blue','green','red']
#########################################################Decision Tree Classifier##########################################
print('Below is the output for decision tree classifier')
for i in range(2, split_threshold):
 classifier = DecisionTreeClassifier()  # configure the classifier
 classifier = classifier.fit(pred_train, tar_train)  # train a decision tree model
 predictions = classifier.predict(pred_test)  # deploy model and make predictions on test set
 probDT = classifier.predict_proba(pred_test)  # obtain probability scores for each sample in test set
 print("Accuracy score of our model with Decision Tree:", i, accuracy_score(tar_test, predictions))
 precision = precision_score(y_true=tar_test, y_pred=predictions, average='micro')
 print("Precision score of our model with Decision Tree :", precision)
 recall = recall_score(y_true=tar_test, y_pred=predictions, average='micro')
 print("Recall score of our model with Decision Tree :", recall)
cm=confusion_matrix(predictions,tar_test)
print(cm)
# for x in range(3):
#     fpr[x], tpr[x], _ = roc_curve(tar_test[:], probDT[:, x], pos_label=x)
#     roc_auc[x] = auc(fpr[x], tpr[x])
#     print("AUC values of the decision tree", roc_auc[x])
#     plt.plot(fpr[x], tpr[x], color=colors[x], label='ROC curve (area = %0.2f)' % roc_auc[x])
#
# print("Accuracy score of our model with DTC :", accuracy_score(tar_test, predictions))
#plt.show()


#######################Multilayer Preceptron###################################
clf = MLPClassifier(learning_rate='constant',activation='logistic',solver='sgd',max_iter=4000,learning_rate_init=0.001)
clf.fit(pred_train,np.ravel(tar_train,order='C'))
predictions1 = clf.predict(pred_test)
probMLP=clf.predict_proba(pred_test)
print("Accuracy score of our model with MLP :", accuracy_score(tar_test, predictions1))
scores = cross_val_score(clf, predictors, target, cv=10)
print("Accuracy score of our model with MLP under cross validation :", scores.mean())

y=sklearn.metrics.confusion_matrix(tar_test,predictions1)
print(y)

#######################Q2#############################################

# count_d=0
# count_h=0
# print(class_target)
# for x in range(class_target.len()):
# for i in range(len(pred_test)):
#  print(probDT[i],probMLP[i])

########################04###############################################3

max_avg_pred=0

def avgClassifier(pred_train,tar_train,pred_test):

 classifier = DecisionTreeClassifier()  # configure the classifier
 classifier = classifier.fit(pred_train, tar_train)  # train a decision tree model
 # predictions = classifier.predict(pred_test)  # deploy model and make predictions on test set
 probDT = classifier.predict_proba(pred_test)

 clf = MLPClassifier(learning_rate='constant', activation='logistic', solver='sgd', max_iter=4000,
                     learning_rate_init=0.001)
 clf.fit(pred_train, np.ravel(tar_train, order='C'))
 # predictions1 = clf.predict(pred_test)
 probMLP = clf.predict_proba(pred_test)
 avg_predictions=[]
 avg_prob = (probDT + probMLP) / 2
 max_avg_pred=0
 for i in range(len(pred_test)):
  l=[avg_prob[i][0],avg_prob[i][1],avg_prob[i][2],avg_prob[i][3]]
  max_avg_pred=l.index(max(l))
  avg_predictions.append(max_avg_pred)
  # print(max_avg_pred,avg_prob[i])
 return avg_predictions

avg_predictions=avgClassifier(pred_train,tar_train,pred_test)
print(avg_predictions)
print('The accuracy of the Average Aggregate Classifier is:',accuracy_score(avg_predictions,tar_test))

##############################Q5#####################
# avg_prob = (probDT + probMLP) / 2
#
# def conditionalClassifier(probDT,probMLP):
#  P1=[]
#  P2=[]
#  for i in range(len(pred_test)):
#   maxDTProb=max(probDT[i][0],probDT[i][1],probDT[i][2],probDT[i][3])
#   maxMlpProb=max(probMLP[i][0],probMLP[i][1],probMLP[i][2],probMLP[i][3])
#   P1