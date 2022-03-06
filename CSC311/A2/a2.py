"""
Created on Fri Oct 22 2021

@author: Yash Dave
"""

import pickle

with open('cluster_data.pickle', 'rb') as file:
    dataTrain, dataTest = pickle.load(file)
    Xtrain, Ttrain = dataTrain
    Xtest, Ttest = dataTest


import sklearn.linear_model as lin
import sklearn.discriminant_analysis as da
import sklearn.naive_bayes as nb
import sklearn.utils as linutils
import sklearn.neural_network as neural
import scipy.stats as stats

clf = lin.LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(Xtrain, Ttrain)

import bonnerlib2D as bl2d

# print("=================== Question 1 (a) ===================\n")
#
# print("Classifier Score Train:", clf.score(Xtrain, Ttrain))
# print("Classifier Score Test:", clf.score(Xtest, Ttest))
#
# # =================== Question 1 (b) ===================
#
# bl2d.plot_data(Xtrain, Ttrain)
# bl2d.boundaries(clf)
# bl2d.plt.title("Question 1(b): decision boundaries for linear classification")
# bl2d.plt.show()
#
#
# print("\n=================== Question 1 (e) ===================\n")
#
#
# def softmax(Z):
#     Y = bl2d.np.exp(Z)
#     S = bl2d.np.sum(Y, axis=1)
#     Y = bl2d.np.divide(Y, S.reshape((Y.shape[0], 1)))
#     return Y
#
#
# def predict(X, W, b): X = (N, 3) W = (t, 3)
#     Z = X @ W.transpose() + b
#     Y = softmax(Z)
#     return bl2d.np.argmax(Y, axis=1)
#
#
# Y1 = clf.predict(Xtest)
# Y2 = predict(Xtest, clf.coef_, clf.intercept_)
#
# print("Square Distance:", bl2d.np.sum(bl2d.np.power(Y1 - Y2, 2)))
#
# print("\n=================== Question 1 (f) ===================\n")
#
#
# def one_hot(Tint):
#     max_class = bl2d.np.amax(Tint) + 1
#     C = bl2d.np.arange(max_class)
#     return bl2d.np.array(Tint.reshape((Tint.shape[0], 1)) == C, dtype=int)
#
#
# print(one_hot(bl2d.np.array([4, 3, 2, 3, 2, 1, 2, 1, 0])))
#
# print("\n=================== Question 2 (a) ===================\n")
#
# def GDlinear(I, lrate):
#
#     bl2d.np.random.seed(7)
#     print("Learning rate =", lrate)
#     ones = bl2d.np.ones(Xtrain.shape[0])
#     Xtr = bl2d.np.c_[ones, Xtrain]
#     Ntr = Xtr.shape[0]
#     Xte = bl2d.np.c_[ones, Xtest]
#     Nte = Xte.shape[0]
#     Ttrain_one = one_hot(Ttrain)
#     Ttest_one = one_hot(Ttest)
#     W = bl2d.np.random.randn(Ttrain_one.shape[1], Xtr.shape[1]) / 10000
#     CE_train, CE_test, accuracy_train, accuracy_test = ([], [], [], [])
#
#     Ztr = Xtr @ W.transpose()
#     Ytr = softmax(Ztr)
#
#     for i in range(I):
#
#         W -= lrate * (1 / Ntr) * (Ytr - Ttrain_one).transpose() @ Xtr
#
#         Ztr = Xtr @ W.transpose()
#         Zte = Xte @ W.transpose()
#         Ytr = softmax(Ztr)
#         Yte = softmax(Zte)
#
#         cost_train = -Ttrain_one * bl2d.np.log(Ytr)
#         cost_train = bl2d.np.sum(cost_train)
#         cost_test = -Ttest_one * bl2d.np.log(Yte)
#         cost_test = bl2d.np.sum(cost_test)
#         CE_train.append(cost_train/Ntr)
#         CE_test.append(cost_test/Nte)
#
#         acc_train = bl2d.np.argmax(Ytr, axis=1) == Ttrain
#         acc_train = bl2d.np.array(acc_train, dtype=int)
#         acc_train = bl2d.np.sum(acc_train)
#         acc_train /= Ntr
#
#         acc_test = bl2d.np.argmax(Yte, axis=1) == Ttest
#         acc_test = bl2d.np.array(acc_test, dtype=int)
#         acc_test = bl2d.np.sum(acc_test)
#         acc_test /= Nte
#
#         accuracy_train.append(acc_train)
#         accuracy_test.append(acc_test)
#
#     iterations = [i for i in range(0, I)]
#
#     bl2d.plt.semilogx(iterations, CE_train, "b")
#     bl2d.plt.semilogx(iterations, CE_test, "r")
#     bl2d.plt.title("Question 2 (a): Training and Test Loss v.s. iterations")
#     bl2d.plt.xlabel("Iteration Number")
#     bl2d.plt.ylabel("Cross Entropy")
#     bl2d.plt.show()
#
#     bl2d.plt.semilogx(iterations, accuracy_train, "b")
#     bl2d.plt.semilogx(iterations, accuracy_test, "r")
#     bl2d.plt.title("Question 2 (a): Training and test accuracy v.s. iterations")
#     bl2d.plt.xlabel("Iteration Number")
#     bl2d.plt.ylabel("Accuracy")
#     bl2d.plt.show()
#
#     bl2d.plt.semilogx(iterations[50:], CE_test[50:], "r")
#     bl2d.plt.title("Question 2 (a): Test Loss from 50 on")
#     bl2d.plt.xlabel("Iteration Number")
#     bl2d.plt.ylabel("Cross Entropy")
#     bl2d.plt.show()
#
#     bl2d.plt.semilogx(iterations[50:], CE_train[50:], "b")
#     bl2d.plt.title("Question 2 (a): Training Loss from 50 on")
#     bl2d.plt.xlabel("Iteration Number")
#     bl2d.plt.ylabel("Cross Entropy")
#     bl2d.plt.show()
#
#     print("Training accuracy:", acc_train)
#     print(clf.score(Xtrain, Ttrain))
#     print("Difference in training accuracy:", acc_train - clf.score(Xtrain, Ttrain))
#
#     print("Training accuracy:", acc_test)
#     print(clf.score(Xtest, Ttest))
#     print("Difference in test accuracy:", acc_test - clf.score(Xtest, Ttest))
#
#     bl2d.plot_data(Xtrain, Ttrain)
#     bl2d.boundaries2(W[:, 1:], W[:, 0], predict)
#     bl2d.plt.title("Question 2(a): decision boundaries for linear classification")
#     bl2d.plt.show()
#
#
# GDlinear(10000, 0.1)
#
# print("=================== Question 2 (d) ===================\n")
#
#
# def SGDlinear(I, batch_size, lrate0, alpha, kappa):
#
#     bl2d.np.random.seed(7)
#
#     print("Batch size =", batch_size)
#     print("Initial learning rate =", lrate0)
#     print("Decay rate =", alpha)
#     print("Burn-in period =", kappa)
#
#     ones = bl2d.np.ones(Xtrain.shape[0])
#     Xtr = bl2d.np.c_[ones, Xtrain]
#     Ntr = Xtr.shape[0]
#     Xte = bl2d.np.c_[ones, Xtest]
#     Nte = Xte.shape[0]
#     Ttrain_one = one_hot(Ttrain)
#     Ttest_one = one_hot(Ttest)
#     W = bl2d.np.random.randn(Ttrain_one.shape[1], Xtr.shape[1]) / 10000
#     CE_train, CE_test, accuracy_train, accuracy_test = ([], [], [], [])
#     lrate = lrate0
#
#     for i in range(I):
#
#         if i >= kappa:
#             k = alpha * (i - kappa)
#             lrate = lrate0 / (1 + k)
#
#         # Epoch starts
#         Xtr_shuffled, Ttrain_one_shuffled, Ttrain_shuffled = linutils.shuffle(Xtr, Ttrain_one, Ttrain)
#
#         # We will sift through training data
#
#         Ztr_shuffled = Xtr_shuffled @ W.transpose()
#         Ytr_shuffled = softmax(Ztr_shuffled)
#
#         for j in range(0, Xtr.shape[0], batch_size):
#
#             # Gradient on each sweep
#             W -= lrate * (1 / Ztr_shuffled[j:j + batch_size].shape[0]) * (Ytr_shuffled[j:j + batch_size] - Ttrain_one_shuffled[j:j + batch_size]).transpose() @ Xtr_shuffled[j:j + batch_size]
#
#             Ztr_shuffled = Xtr_shuffled @ W.transpose()
#             Ytr_shuffled = softmax(Ztr_shuffled)
#
#         Zte_shuffled = Xte @ W.transpose()
#         Yte_shuffled = softmax(Zte_shuffled)
#
#         cost_train = -Ttrain_one_shuffled * bl2d.np.log(Ytr_shuffled)
#         cost_train = bl2d.np.sum(cost_train)
#         cost_test = -Ttest_one * bl2d.np.log(Yte_shuffled)
#         cost_test = bl2d.np.sum(cost_test)
#         CE_train.append(cost_train/Ntr)
#         CE_test.append(cost_test/Nte)
#
#         acc_train = bl2d.np.argmax(Ytr_shuffled, axis=1) == Ttrain_shuffled
#         acc_train = bl2d.np.array(acc_train, dtype=int)
#         acc_train = bl2d.np.sum(acc_train)
#         acc_train /= Ntr
#
#         acc_test = bl2d.np.argmax(Yte_shuffled, axis=1) == Ttest
#         acc_test = bl2d.np.array(acc_test, dtype=int)
#         acc_test = bl2d.np.sum(acc_test)
#         acc_test /= Nte
#
#         accuracy_train.append(acc_train)
#         accuracy_test.append(acc_test)
#
#     iterations = [i for i in range(0, I)]
#
#     bl2d.plt.semilogx(iterations, CE_train, "b")
#     bl2d.plt.semilogx(iterations, CE_test, "r")
#     bl2d.plt.title("Question 2 (d): Training and Test Loss v.s. iterations")
#     bl2d.plt.xlabel("Iteration Number")
#     bl2d.plt.ylabel("Cross Entropy")
#     bl2d.plt.show()
#
#     bl2d.plt.semilogx(iterations, accuracy_train, "b")
#     bl2d.plt.semilogx(iterations, accuracy_test, "r")
#     bl2d.plt.title("Question 2 (d): Training and test accuracy v.s. iterations")
#     bl2d.plt.xlabel("Iteration Number")
#     bl2d.plt.ylabel("Accuracy")
#     bl2d.plt.show()
#
#     bl2d.plt.semilogx(iterations[50:], CE_test[50:], "r")
#     bl2d.plt.title("Question 2 (d): Test Loss from 50 on")
#     bl2d.plt.xlabel("Iteration Number")
#     bl2d.plt.ylabel("Cross Entropy")
#     bl2d.plt.show()
#
#     bl2d.plt.semilogx(iterations[50:], CE_train[50:], "b")
#     bl2d.plt.title("Question 2 (d): Training Loss from 50 on")
#     bl2d.plt.xlabel("Iteration Number")
#     bl2d.plt.ylabel("Cross Entropy")
#     bl2d.plt.show()
#
#     print("Training accuracy:", acc_train)
#     print(clf.score(Xtrain, Ttrain))
#     print("Difference in training accuracy:", acc_train - clf.score(Xtrain, Ttrain))
#
#     print("Training accuracy:", acc_test)
#     print(clf.score(Xtest, Ttest))
#     print("Difference in test accuracy:", acc_test - clf.score(Xtest, Ttest))
#
#     bl2d.plot_data(Xtrain, Ttrain)
#     bl2d.boundaries2(W[:, 1:], W[:, 0], predict)
#     bl2d.plt.title("Question 2(d): decision boundaries for linear classification")
#     bl2d.plt.show()
#
#
# SGDlinear(500, 30, 0.01, 2, 300)
#
# print("\n=================== Question 3 (a) ===================\n")
#
# qclf = da.QuadraticDiscriminantAnalysis(store_covariance=True)
# qclf.fit(Xtrain, Ttrain)
#
# print("Quadratic Discriminant Score Train:", qclf.score(Xtrain, Ttrain))
# print("Quadratic Discriminant Score Test:", qclf.score(Xtest, Ttest))
#
# bl2d.plot_data(Xtrain, Ttrain)
# bl2d.boundaries(qclf)
# bl2d.plt.title("Question 3(a): decision boundaries for linear classification")
# bl2d.plt.show()
#
# print("\n=================== Question 3 (b) ===================\n")
#
# nbclf = nb.GaussianNB()
# nbclf.fit(Xtrain, Ttrain)
#
# print("Naive Bayes Score Train:", nbclf.score(Xtrain, Ttrain))
# print("Naive Bayes Score Test:", nbclf.score(Xtest, Ttest))
#
# bl2d.plot_data(Xtrain, Ttrain)
# bl2d.boundaries(nbclf)
# bl2d.plt.title("Question 3(b): decision boundaries for linear classification")
# bl2d.plt.show()
#
# print("\n=================== Question 3 (f) ===================\n")
#
#
# def EstMean(X, T):
#
#     mu_top = T.transpose() @ X
#     mu_bottom = bl2d.np.sum(T, axis=0)
#     mu = mu_top / mu_bottom.reshape((mu_top.shape[0], 1))
#     return mu
#
#
# print("Square Distance:", bl2d.np.sum(bl2d.np.square(EstMean(Xtrain, one_hot(Ttrain)) - qclf.means_)))
#
# print("\n=================== Question 3 (g) ===================\n")
#
#
# def EstCov(X, T):
#
#     N = X.shape[0]
#     I = X.shape[1]
#     J = X.shape[1]
#     mu = EstMean(X, T)
#     K = mu.shape[0]
#     Xni = X.reshape((N, 1, I))
#     mu_ki = mu.reshape((1, K, I))
#
#     Anki = (Xni - mu_ki).reshape((N, K, I, 1))
#     Ankj = (Xni - mu_ki).reshape((N, K, 1, J))
#
#     Bnkij = Anki * Ankj
#
#     Ttr = T.reshape((N, K, 1, 1))
#
#     Cnkij = Ttr * Bnkij
#
#     Dkij = bl2d.np.sum(Cnkij, axis=0)
#
#     Nk = bl2d.np.sum(Ttr, axis=0) - 1
#
#     Skij = Dkij / Nk
#
#     return Skij
#
#
# print("Square Distance:", bl2d.np.sum(bl2d.np.square(EstCov(Xtrain, one_hot(Ttrain)) - qclf.covariance_)))
#
# print("\n=================== Question 3 (h) ===================\n")
#
# def EstPrior(T):
#
#     C = bl2d.np.sum(T, axis=0)
#     return C / T.shape[0]
#
#
# print("Square Distance:", bl2d.np.sum(bl2d.np.square(EstPrior(one_hot(Ttrain)) - qclf.priors_)))
#
#
# print("\n=================== Question 3 (i) ===================\n")
#
#
# def EstPost(mean, cov, prior, X):
#
#     likelihood_prior = bl2d.np.empty((X.shape[0], mean.shape[0]))
#
#     for i in range(mean.shape[0]):
#
#         class_likelihood = stats.multivariate_normal.pdf(X, mean[i], cov[i]) * prior[i]
#         likelihood_prior[:, i] = class_likelihood
#
#     evidence = bl2d.np.sum(likelihood_prior, axis=0)
#
#     return likelihood_prior / evidence
#
#
# print("Square Distance:", bl2d.np.sum(bl2d.np.square(EstPost(EstMean(Xtest, one_hot(Ttest)), EstCov(Xtest, one_hot(Ttest)), EstPrior(one_hot(Ttest)), Xtest) - nbclf.predict_proba(Xtest))))
#
# print("\n=================== Question 3 (j) ===================\n")
#
# print("I don't know")
#
# print("\n=================== Question 3 (k) ===================\n")
#
# print("I don't know")

print("\n=================== Question 4 (a) ===================\n")


bl2d.np.random.seed(7)
neural_clf = neural.MLPClassifier(max_iter=10000, learning_rate_init=0.01, tol=10**(-6), hidden_layer_sizes=(5,), activation="logistic", solver="sgd")
neural_clf.fit(Xtrain, Ttrain)

print(Xtrain.shape)

for i in range(len(neural_clf.coefs_)):
    print(neural_clf.coefs_[i].shape)
    print(neural_clf.intercepts_[i].shape)

print("Neural Network 5 Hidden Units Train:", neural_clf.score(Xtrain, Ttrain))
print("Neural Network 5 Hidden Units Test:", neural_clf.score(Xtest, Ttest))

bl2d.plot_data(Xtrain, Ttrain)
bl2d.boundaries(neural_clf)
bl2d.plt.title("Question 4(a): Neural Network with 5 hidden units")
bl2d.plt.show()

print("\n=================== Question 4 (b) ===================\n")

def boundaries(ax, clf):

    # The extent of xy space
    # ax = bl2d.plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # form a mesh/grid over xy space
    h = 0.02    # mesh granularity
    xx, yy = bl2d.np.meshgrid(bl2d.np.arange(x_min, x_max, h), bl2d.np.arange(y_min, y_max, h))
    mesh = bl2d.np.c_[xx.ravel(), yy.ravel()]

    # evaluate the decision function at the grid points
    U = clf.predict(mesh)    ###### Modify this line ######

    # Use pale colours for the decision regions.
    U = U.reshape(xx.shape)
    mylevels = [-0.5, 0.5, 1.5, 2.5]
    ax.contourf(xx, yy, U, levels=mylevels, colors=('red', 'blue', 'green'), alpha=0.2)

    # draw the decision boundaries in solid black
    ax.contour(xx, yy, U, levels=3, colors='k', linestyles='solid')


ax1 = bl2d.plt.subplot(2, 2, 1)
ax2 = bl2d.plt.subplot(2, 2, 2)
ax3 = bl2d.plt.subplot(2, 2, 3)
ax4 = bl2d.plt.subplot(2, 2, 4)

bl2d.np.random.seed(7)
neural_clf_1 = neural.MLPClassifier(max_iter=10000, learning_rate_init=0.01, tol=10**(-6), hidden_layer_sizes=(1,), activation="logistic", solver="sgd")
neural_clf_1.fit(Xtrain, Ttrain)

print("Neural Network 1 Hidden Unit Train:", neural_clf_1.score(Xtrain, Ttrain))
print("Neural Network 1 Hidden Unit Test:", neural_clf_1.score(Xtest, Ttest))

colors = bl2d.np.array(['r', 'b', 'g'])    # red for class 0 , blue for class 1, green for class 2
ax1.scatter(Xtrain[:, 0], Xtrain[:, 1], color=colors[Ttrain], s=2)
xmin, ymin = bl2d.np.min(Xtrain, axis=0) - 0.1
xmax, ymax = bl2d.np.max(Xtrain, axis=0) + 0.1
bl2d.plt.xlim(xmin, xmax)
bl2d.plt.ylim(ymin, ymax)

boundaries(ax1, neural_clf_1)
ax1.set_title("Question 4(b): Neural Network with 1 hidden unit")


bl2d.np.random.seed(7)
neural_clf_2 = neural.MLPClassifier(max_iter=10000, learning_rate_init=0.01, tol=10**(-6), hidden_layer_sizes=(2,), activation="logistic", solver="sgd")
neural_clf_2.fit(Xtrain, Ttrain)

print("Neural Network 2 Hidden Units Train:", neural_clf_2.score(Xtrain, Ttrain))
print("Neural Network 2 Hidden Units Test:", neural_clf_2.score(Xtest, Ttest))

colors = bl2d.np.array(['r', 'b', 'g'])    # red for class 0 , blue for class 1, green for class 2
ax2.scatter(Xtrain[:, 0], Xtrain[:, 1], color=colors[Ttrain], s=2)
xmin, ymin = bl2d.np.min(Xtrain, axis=0) - 0.1
xmax, ymax = bl2d.np.max(Xtrain, axis=0) + 0.1
bl2d.plt.xlim(xmin, xmax)
bl2d.plt.ylim(ymin, ymax)

boundaries(ax2, neural_clf_2)
ax2.set_title("Question 4(b): Neural Network with 2 hidden units")


bl2d.np.random.seed(7)
neural_clf_4 = neural.MLPClassifier(max_iter=10000, learning_rate_init=0.01, tol=10**(-6), hidden_layer_sizes=(4,), activation="logistic", solver="sgd")
neural_clf_4.fit(Xtrain, Ttrain)

print("Neural Network 4 Hidden Units Train:", neural_clf_4.score(Xtrain, Ttrain))
print("Neural Network 4 Hidden Units Test:", neural_clf_4.score(Xtest, Ttest))

colors = bl2d.np.array(['r', 'b', 'g'])    # red for class 0 , blue for class 1, green for class 2
ax3.scatter(Xtrain[:, 0], Xtrain[:, 1], color=colors[Ttrain], s=2)
xmin, ymin = bl2d.np.min(Xtrain, axis=0) - 0.1
xmax, ymax = bl2d.np.max(Xtrain, axis=0) + 0.1
bl2d.plt.xlim(xmin, xmax)
bl2d.plt.ylim(ymin, ymax)

boundaries(ax3, neural_clf_4)
ax3.set_title("Question 4(b): Neural Network with 4 hidden units")


bl2d.np.random.seed(7)
neural_clf_10 = neural.MLPClassifier(max_iter=10000, learning_rate_init=0.01, tol=10**(-6), hidden_layer_sizes=(10,), activation="logistic", solver="sgd")
neural_clf_10.fit(Xtrain, Ttrain)

print("Neural Network 10 Hidden Units Train:", neural_clf_10.score(Xtrain, Ttrain))
print("Neural Network 10 Hidden Units Test:", neural_clf_10.score(Xtest, Ttest))

colors = bl2d.np.array(['r', 'b', 'g'])    # red for class 0 , blue for class 1, green for class 2
ax4.scatter(Xtrain[:, 0], Xtrain[:, 1], color=colors[Ttrain], s=2)
xmin, ymin = bl2d.np.min(Xtrain, axis=0) - 0.1
xmax, ymax = bl2d.np.max(Xtrain, axis=0) + 0.1
bl2d.plt.xlim(xmin, xmax)
bl2d.plt.ylim(ymin, ymax)

boundaries(ax4, neural_clf_10)

ax4.set_title("Question 4(b): Neural Network with 10 hidden units")

bl2d.plt.show()
