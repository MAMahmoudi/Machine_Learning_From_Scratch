from src.KNN import KNN
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.utility import *
from src.Regression import Linear_Regression
from src.Regression import Logistic_Regression
from src.SGD_Regression import SGD_Regression
from src.Polynomial_Regression import Polynomial_Regression
from src.Ridge_Regression import Ridge_Regression
from src.Naive_Bayes import Naive_Bayes
from src.Perceptron import Perceptron
from src.SVM import SVM
from src.Decision_Tree import Decision_Tree
from src.Random_Forest import Random_Forest
from src.PCA import PCA
# cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target
# X, y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
# X, y = datasets.make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=123)
# X, y = datasets.make_blobs(n_samples=150,n_features=2,centers=2,cluster_std=1.05,random_state=2)
# X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
# BC = datasets.load_breast_cancer()
# X, y = BC.data, BC.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


################ PCA #######################
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)
print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
################ Random Forest #######################

# clf = Random_Forest(n_trees=3, max_depth=10)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("The Random Forest classification accuracy", accuracy(y_test,y_pred))

# ################ Decision_Tree #######################
# clf = Decision_Tree(max_depth=10)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("The Decision Tree classification accuracy", accuracy(y_test,y_pred))

################ SVM #######################
# y = np.where(y == 0, -1, 1)
# clf = SVM()
# clf.fit(X, y)
# # predictions = clf.predict(X)
#
# print(clf.weights, clf.bias)
#
#
# def visualize_svm():
#     def get_hyperplane_value(x, w, b, offset):
#         return (-w[0] * x + b + offset) / w[1]
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
#     x0_1 = np.amin(X[:, 0])
#     x0_2 = np.amax(X[:, 0])
#     x1_1 = get_hyperplane_value(x0_1, clf.weights, clf.bias, 0)
#     x1_2 = get_hyperplane_value(x0_2, clf.weights, clf.bias, 0)
#     x1_1_m = get_hyperplane_value(x0_1, clf.weights, clf.bias, -1)
#     x1_2_m = get_hyperplane_value(x0_2, clf.weights, clf.bias, -1)
#     x1_1_p = get_hyperplane_value(x0_1, clf.weights, clf.bias, 1)
#     x1_2_p = get_hyperplane_value(x0_2, clf.weights, clf.bias, 1)
#     ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
#     ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
#     ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")
#     x1_min = np.amin(X[:, 1])
#     x1_max = np.amax(X[:, 1])
#     ax.set_ylim([x1_min - 3, x1_max + 3])
#     plt.show()
#
# visualize_svm()
# ############### Perceptron #######################
# perceptron = Perceptron()
# perceptron.fit(X_train, y_train)
# predictions = perceptron.predict(X_test)
# print("Naive Bayes classification accuracy", accuracy(y_test,predictions))
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
#
# x0_1 = np.amin(X_train[:, 0])
# x0_2 = np.amax(X_train[:, 0])
#
# x1_1 = (-perceptron.weights[0] * x0_1 - perceptron.bias) / perceptron.weights[1]
# x1_2 = (-perceptron.weights[0] * x0_2 - perceptron.bias) / perceptron.weights[1]
#
# ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
#
# ymin = np.amin(X_train[:, 1])
# ymax = np.amax(X_train[:, 1])
# ax.set_ylim([ymin - 3, ymax + 3])
#
# plt.show()
################ Naive_Bayes #######################
# nb = Naive_Bayes()
# nb.fit(X_train, y_train)
# predictions = nb.predict(X_test)
# print("Naive Bayes classification accuracy", accuracy(y_test,predictions))
################ Polynomial_Regression #######################
# m = 100
# X = 6 * np.random.rand(m, 1) - 3
# y = 0.5 * X**2 + X + 2 + np.random.rand(m, 1)
# y = y.reshape(-1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#
# regressor = Polynomial_Regression(order=4)
# regressor.fit(X_train, y_train)
# predictions = regressor.predict(X_test)
# y_pred_line = regressor.predict(X)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# m1 = plt.scatter(X_train,y_train,color=cmap(0,9),s=10)
# m2 = plt.scatter(X_test,y_test,color=cmap(0,2 ),s=10)
# plt.scatter(X_train,y_pred_line,color='black',linewidth=2,label="Prediction")
# plt.show()

# ################ SGD_Regression #######################
# regressor = SGD_Regression(lr=0.01,n_epochs=50)
# print(X_train.shape)
# print(y_train.shape)
# plt.figure(figsize=(8,6))
# plt.scatter(X[:,0], y, color= "b", marker="o", s=30)
# plt.show()

# regressor.fit(X_train, y_train)
# predictions = regressor.predict(X_test)
# plot_learning_curve(regressor, X_train, X_test, y_train, y_test)

# mse_value = mse(y_test, predictions)
# print(mse_value)
# y_pred_line = regressor.predict(X)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# m1 = plt.scatter(X_train,y_train,color=cmap(0,9),s=10)
# m2 = plt.scatter(X_test,y_test,color=cmap(0,2 ),s=10)
# plt.plot(X,y_pred_line,color='black',linewidth=2,label="Prediction")
# plt.show()

################ Logistic_Regression #######################
# regressor = Logistic_Regression(lr=0.0001)
# regressor.fit(X_train, y_train)
# predictions = regressor.predict(X_test)
# acc = accuracy(y_test,predictions)
# print(f'The accuracy rate is : {acc}')


# ############### Ridge_Regression #######################
# print(X_train.shape)
# print(y_train.shape)
# plt.figure(figsize=(8,6))
# plt.scatter(X[:,0], y, color= "b", marker="o", s=30)
# plt.show()
#
# regressor = Ridge_Regression(lr=0.01)
# regressor.fit(X_train, y_train)
# predictions = regressor.predict(X_test)
# plot_learning_curve(regressor, X_train, X_test, y_train, y_test)
#
# mse_value = mse(y_test, predictions)
# print(mse_value)
# y_pred_line = regressor.predict(X)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# m1 = plt.scatter(X_train,y_train,color=cmap(0,9),s=10)
# m2 = plt.scatter(X_test,y_test,color=cmap(0,2 ),s=10)
# plt.plot(X,y_pred_line,color='black',linewidth=2,label="Prediction")
# plt.show()


################ Linear_Regression #######################
# print(X_train.shape)
# print(y_train.shape)
# plt.figure(figsize=(8,6))
# plt.scatter(X[:,0], y, color= "b", marker="o", s=30)
# plt.show()
#
# regressor = Linear_Regression(lr=0.01)
# regressor.fit(X_train, y_train)
# predictions = regressor.predict(X_test)
# plot_learning_curve(regressor, X_train, X_test, y_train, y_test)

# mse_value = mse(y_test, predictions)
# print(mse_value)
# y_pred_line = regressor.predict(X)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# m1 = plt.scatter(X_train,y_train,color=cmap(0,9),s=10)
# m2 = plt.scatter(X_test,y_test,color=cmap(0,2 ),s=10)
# plt.plot(X,y_pred_line,color='black',linewidth=2,label="Prediction")
# plt.show()
################# KNN ######################
# clf = KNN(k=5)
# clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# acc = accuracy(y_test,predictions)
# print(f'The accuracy rate is : {acc}')
# plot_learning_curve(clf, X_train, X_test, y_train, y_test)

# print(X_train.shape)
# print(X_train[0])
# print(y_train.shape)
# print(y_train)
# plt.figure()
# plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.show()
