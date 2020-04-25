import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
# path = r'C:\Users\udit sharma\Downloads\Diabe.csv'
path=r'C:\Users\udit sharma\Desktop\Aut\Data Mining and Machine Learning\Diabe.csv'
rawdata = pd.read_csv(path)
print(rawdata)

nrow, ncol = rawdata.shape
##################################### 1 ######################################



predictors=rawdata.iloc[:,0:ncol-1]
# print(predictors)
target=rawdata.iloc[:,ncol-1]
# print(target)
pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors,target,test_size=.3,shuffle=False)

clf = MLPClassifier(hidden_layer_sizes=(20,), max_iter=150, alpha=1e-4,
                    solver='sgd', verbose=0, tol=1e-8, random_state=1,
                    learning_rate_init=.01)


clf.fit(pred_train,np.ravel(tar_train,order='C'))
predictions1 = clf.predict(pred_test)
probMLP=clf.predict_proba(pred_test)
print("Accuracy score of our model with MLP :", accuracy_score(tar_test, predictions1))
scores = cross_val_score(clf, predictors, target, cv=10)
print("Accuracy score of our model with MLP under cross validation :", scores.mean())


# ######################################## 2############################
#
#
#
#
# """ Example based on sklearn's docs """
# # mnist = fetch_openml('mnist_784')
# # # rescale the data, use the traditional train/test split
# # X, y = mnist.data / 255., mnist.target
# X_train, X_test = pred_train,pred_test
# y_train, y_test =  tar_train,tar_test
#
# mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=150, alpha=1e-4,
#                     solver='sgd', verbose=0, tol=1e-8, random_state=1,
#                     learning_rate_init=.01)
#
# N_TRAIN_SAMPLES = X_train.shape[0]
# N_EPOCHS = 25
# N_BATCH = 64
# N_CLASSES = np.unique(y_train)
#
# scores_train = []
# scores_test = []
# mlploss = []
#
# # EPOCH
# epoch = 0
# while epoch < N_EPOCHS:
#     print('epoch: ', epoch)
#     # SHUFFLING
#     random_perm = np.random.permutation(X_train.shape[0])
#     mini_batch_index = 0
#     while True:
#         # MINI-BATCH
#         indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
#
#         mlp.partial_fit(X_train.iloc[indices], y_train[indices], classes=N_CLASSES)
#         mini_batch_index += N_BATCH
#
#         if mini_batch_index >= N_TRAIN_SAMPLES:
#             break
#
#     # SCORE TRAIN
#     scores_train.append(1 - mlp.score(X_train, y_train))
#
#     # SCORE TEST
#     scores_test.append(1 - mlp.score(X_test, y_test))
#
#     # compute loss
#
#     mlploss.append(mlp.loss_)
#     epoch += 1
#
# """ Plot """
# fig, ax = plt.subplots(3, sharex=True)
# ax[0].plot(scores_train)
# ax[0].set_title('Train Error')
# ax[1].plot(mlploss)
# ax[1].set_title('Train Loss')
# ax[2].plot(scores_test)
# ax[2].set_title('Test Error')
# fig.suptitle("Error vs Loss over epochs", fontsize=14)
# # fig.savefig('C:/Users/rpear/OneDrive/Apps/Documents/LossCurve.png')
# plt.show()

########################################## 3 ##########################################
for i in range(20):

 mlp = MLPClassifier(hidden_layer_sizes=(20-i-1,i+1), max_iter=150, alpha=1e-4,
                    solver='sgd', verbose=0, tol=1e-8, random_state=1,
                    learning_rate_init=.01)
 clf.fit(pred_train, np.ravel(tar_train, order='C'))
 predictions1 = clf.predict(pred_test)
 # probMLP = clf.predict_proba(pred_test)
 print("Accuracy score of our model with MLP for",i+1," iteration :", accuracy_score(tar_test, predictions1))