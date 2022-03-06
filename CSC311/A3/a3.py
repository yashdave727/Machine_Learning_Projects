import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as neural
import sklearn.utils as utils
import sklearn.cluster as cluster
import sklearn.mixture as mixture
import bonnerlib2D as bl2d

# ========== PRE QUESTION 1 ==============
with open('mnistTVT.pickle', 'rb') as f:
    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)

# im_Xtrain = Xtrain.reshape((Xtrain.shape[0], 28, 28))
# plt.imshow(im_Xtrain[0], cmap='Greys', interpolation='nearest')
# plt.show()
# ========== PRE QUESTION 1 ==============

# ============ Question 1 (a) ============

small_Xtrain = Xtrain[:500]
small_Ttrain = Ttrain[:500]
small_Xval = Xval[:500]
small_Tval = Tval[:500]
small_Xtest = Xtest[:500]
small_Ttest = Ttest[:500]

# print("\n============ Question 1 (b) ============\n")
#
# np.random.seed(7)
# neural_clf = neural.MLPClassifier(activation='logistic', hidden_layer_sizes=5, learning_rate_init=0.1, max_iter=10000, solver='sgd', alpha=0)
# neural_clf.fit(small_Xtrain, small_Ttrain)
# print("Neural Classifier Score:", neural_clf.score(small_Xtrain, small_Ttrain))
# print("Neural Classifier Score:", neural_clf.score(small_Xval, small_Tval))
#
# print("\n============ Question 1 (d) ============\n")
#
# probabilities = neural_clf.predict_proba(small_Xtrain)
# my_Ttrain = np.argmax(probabilities, axis=1)
# print("Neural Classifier Score Difference:", neural_clf.score(small_Xtrain, my_Ttrain) - neural_clf.score(small_Xtrain, small_Ttrain))
#
# # ============ Question 1 (e) ============
#
# np.random.seed(7)
# acc = []
# for i in range(7):
#     new_small_Xtrain, new_small_Ttrain = utils.resample(small_Xtrain, small_Ttrain)
#     nclf = neural.MLPClassifier(activation='logistic', hidden_layer_sizes=5, learning_rate_init=0.1, max_iter=10000, solver='sgd', alpha=0)
#     nclf.fit(new_small_Xtrain, new_small_Ttrain)
#     probab = nclf.predict_proba(Xval)
#     acc.append(probab)
#
# total = sum(acc)
# average = total / 7
#
# # ============ Question 1 (g) ============
#
# np.random.seed(7)
# acc = []
# acc_accuracy = []
# for i in range(100):
#     new_small_Xtrain, new_small_Ttrain = utils.resample(small_Xtrain, small_Ttrain)
#     nclf = neural.MLPClassifier(activation='logistic', hidden_layer_sizes=5, learning_rate_init=0.1, max_iter=10000, solver='sgd', alpha=0)
#     nclf.fit(new_small_Xtrain, new_small_Ttrain)
#     probab = nclf.predict_proba(small_Xval)
#     acc.append(probab)
#     total = sum(acc)
#     average = total / len(acc)
#     acc_accuracy.append(np.average(np.argmax(average, axis=1) == small_Tval))
#
# plt.plot(range(1, 101), acc_accuracy)
# plt.title("Average Accuracy vs Iteration")
# plt.xlabel("Iteration number")
# plt.ylabel("Average Accuracy")
# plt.show()

# ============ PRE QUESTION 2 ============

DIMENSION = 10
ACTIONS = ["up", "down", "left", "right"]
Q = np.zeros((DIMENSION, DIMENSION, len(ACTIONS)))


# ============ Question 2 (a) ============

AQUA = 25
BLUE = 15
YELLOW = 72

block_world = np.zeros((10, 10))
block_world[3][1] = AQUA
block_world[3][2] = AQUA
block_world[2][5] = AQUA
block_world[3][5] = AQUA
block_world[4][5] = AQUA
block_world[5][5] = AQUA
block_world[6][5] = AQUA
block_world[7][5] = AQUA
block_world[4][6] = AQUA
block_world[4][7] = AQUA
block_world[4][8] = AQUA
block_world[5][3] = BLUE
block_world[5][6] = YELLOW
block_world[7][2] = AQUA
block_world[7][3] = AQUA
block_world[7][4] = AQUA
block_world[7][8] = AQUA

plt.imshow(block_world)
plt.title("Question 2(a): Grid World")
plt.show()

# ============ Question 2 (b) ============

BLOCKS = [
    (1, 3)[::-1],
    (2, 3)[::-1],
    (5, 2)[::-1],
    (5, 3)[::-1],
    (5, 4)[::-1],
    (5, 5)[::-1],
    (5, 6)[::-1],
    (5, 7)[::-1],
    (6, 4)[::-1],
    (7, 4)[::-1],
    (8, 4)[::-1],
    (2, 7)[::-1],
    (3, 7)[::-1],
    (4, 7)[::-1],
    (8, 7)[::-1],
]
GOAL = 5, 6
INITIAL = 5, 3
REWARD_NO_GOAL = 0
REWARD_GOAL = 25
EDGE = 9

"""
Translate function that takes in a location and action and performs the action
returning a new location and it's corresponding reward
"""
def Trans(L, a):

    if a == "left":
        if L[1] == 0 or (L[0], L[1] - 1) in BLOCKS:
            return L, REWARD_NO_GOAL
        elif (L[0], L[1] - 1) == GOAL:
            return (L[0], L[1] - 1), REWARD_GOAL
        else:
            return (L[0], L[1] - 1), REWARD_NO_GOAL

    elif a == "right":
        if L[1] == EDGE or (L[0], L[1] + 1) in BLOCKS:
            return L, REWARD_NO_GOAL
        elif (L[0], L[1] + 1) == GOAL:
            return (L[0], L[1] + 1), REWARD_GOAL
        else:
            return (L[0], L[1] + 1), REWARD_NO_GOAL

    elif a == "down":
        if L[0] == EDGE or (L[0] + 1, L[1]) in BLOCKS:
            return L, REWARD_NO_GOAL
        elif (L[0] + 1, L[1]) == GOAL:
            return (L[0] + 1, L[1]), REWARD_GOAL
        else:
            return (L[0] + 1, L[1]), REWARD_NO_GOAL

    else:
        if L[0] == 0 or (L[0] - 1, L[1]) in BLOCKS:
            return L, REWARD_NO_GOAL
        elif (L[0] - 1, L[1]) == GOAL:
            return (L[0] - 1, L[1]), REWARD_GOAL
        else:
            return (L[0] - 1, L[1]), REWARD_NO_GOAL


# ============ Question 2 (c) ============

"""
Softmax function which takes in a random variable and returns the softmax of the
random variable
"""
def softmax(Z):
    Y = np.exp(Z)
    S = np.sum(Y)
    Y = np.divide(Y, S)
    return Y

"""
Chooses an action given a policy
"""
def choose(L, beta):
    pred = softmax(Q[L[0], L[1]] * beta)
    return np.random.choice(ACTIONS, p=pred)


# ============ Question 2 (d) ============

"""
Updates the Q matrix given a location, action and parameters alpha and gamma
"""
def updateQ(L, a, alpha, gamma):
    index = ACTIONS.index(a)
    location, reward = Trans(L, a)
    maximum = np.max(Q[location[0], location[1]])
    Q[L[0], L[1], index] += alpha * (reward + (gamma * maximum) - Q[L[0], L[1], index])
    return location


# ============ Question 2 (e) ============

"""
Performs one episode of going from the initial state tp the goal state
and returns the length of the episode
"""
def episode(L, alpha, gamma, beta):
    location = L
    episode_len = 0
    while location != GOAL:
        choice = choose(location, beta)
        location = updateQ(location, choice, alpha, gamma)
        episode_len += 1

    return episode_len


# ============ Question 2 (f) ============

"""
Performs N iterations of learn from a location L and returns the lengths of
each episode
"""
def learn(N, L, alpha, gamma, beta):

    global Q
    lengths = []
    Q = np.zeros((DIMENSION, DIMENSION, len(ACTIONS)))

    for _ in range(N):
        lengths.append(episode(L, alpha, gamma, beta))

    return lengths


# ============ Question 2 (g) ============


np.random.seed(7)
episode_lengths = learn(50, INITIAL, alpha=1, gamma=0.9, beta=1)
plt.plot(range(1, 51), episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Question 2(g): one run of Q learning")
plt.grid()
plt.show()


# ============ Question 2 (h) ============


np.random.seed(7)
episode_lengths = learn(50, INITIAL, alpha=1, gamma=0.9, beta=0)
plt.plot(range(1, 51), episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Question 2(h): one run of Q learning with beta=0")
plt.grid()
plt.show()


# ============ Question 2 (i) ============


np.random.seed(7)
acc = np.zeros((100, 50))
for i in range(100):
    episode_lengths = learn(50, INITIAL, alpha=1, gamma=0.9, beta=1)
    acc[i] = episode_lengths

avg_episode_lengths = np.average(acc, axis=0)
plt.plot(range(1, 51), avg_episode_lengths)
plt.xlim(0, 50)
plt.ylim(0, 1000)
plt.xlabel("Episode")
plt.ylabel("Average Episode Length")
plt.title("Question 2(i): 100 runs of Q learning")
plt.grid()
plt.show()


# ============ Question 2 (j) ============


np.random.seed(7)
acc = np.zeros((100, 50))
for i in range(100):
    episode_lengths = learn(50, INITIAL, alpha=1, gamma=0.9, beta=0.1)
    acc[i] = episode_lengths

avg_episode_lengths = np.average(acc, axis=0)
plt.plot(range(1, 51), avg_episode_lengths)
plt.xlim(0, 50)
plt.ylim(0, 1000)
plt.xlabel("Episode")
plt.ylabel("Average Episode Length")
plt.title("Question 2(j): 100 runs of Q learning beta=0.1")
plt.grid()
plt.show()


# ============ Question 2 (k) ============


np.random.seed(7)
acc = np.zeros((100, 50))
for i in range(100):
    episode_lengths = learn(50, INITIAL, alpha=1, gamma=0.9, beta=0.01)
    acc[i] = episode_lengths

avg_episode_lengths = np.average(acc, axis=0)
plt.plot(range(1, 51), avg_episode_lengths)
plt.xlim(0, 50)
plt.ylim(0, 1000)
plt.xlabel("Episode")
plt.ylabel("Average Episode Length")
plt.title("Question 2(k): 100 runs of Q learning beta=0.01")
plt.grid()
plt.show()


# ============ Question 2 (m) ============


def Qmax():
    return np.max(Q, axis=2)


np.random.seed(7)
episode_lengths = learn(50, INITIAL, alpha=1, gamma=0.9, beta=1)
plt.imshow(Qmax())
plt.title("Question 2(m): Qmax for beta=1")
plt.show()


# ============ Question 2 (o) ============


np.random.seed(7)
axs_lst = []
for i in range(9):
    learn(50, INITIAL, alpha=1, gamma=0.9, beta=1)
    axs_lst.append(plt.subplot(3, 3, i + 1))
    axs_lst[i].axis("off")
    axs_lst[i].imshow(Qmax())

plt.suptitle("Question 2(o): Qmax for beta=1")
plt.show()


# ============ Question 2 (p) ============


np.random.seed(7)
axs_lst = []
for i in range(9):
    learn(50, INITIAL, alpha=1, gamma=0.9, beta=0)
    axs_lst.append(plt.subplot(3, 3, i + 1))
    axs_lst[i].axis("off")
    axs_lst[i].imshow(Qmax())

plt.suptitle("Question 2(p): Qmax for beta=0")
plt.show()


print("\n============ Question 2 (r) ============\n")


def pi():
    return np.argmax(Q, axis=2)


def path(L):
    location = L
    env = block_world.copy()
    while location != GOAL:
        prev_loc = location
        pie = pi()
        choice = pie[location[0], location[1]]
        location, reward = Trans(location, ACTIONS[choice])
        env[prev_loc[0]][prev_loc[1]] = YELLOW

    return env


np.random.seed(7)
episode_length_lst = learn(50, INITIAL, alpha=1, gamma=0.9, beta=1)

world = path(INITIAL)
plt.imshow(world)
plt.title("Question 2(r): an optimal path, beta=1")
plt.show()

print("The length of the path is", np.count_nonzero(world) - len(BLOCKS) - 1)


print("\n============ Question 2 (s) ============\n")


np.random.seed(7)
axs_lst = []
for i in range(9):
    learn(50, INITIAL, alpha=1, gamma=0.9, beta=1)
    axs_lst.append(plt.subplot(3, 3, i + 1))
    axs_lst[i].axis("off")
    world = path(INITIAL)
    axs_lst[i].imshow(world)
    print(f"The length of path {i+1} is", np.count_nonzero(world) - len(BLOCKS) - 1)

plt.suptitle("Question 2(s): optimal paths, beta=1")
plt.show()


print("\n============ Question 2 (t) ============\n")


np.random.seed(7)
axs_lst = []
for i in range(9):
    learn(50, INITIAL, alpha=1, gamma=0.9, beta=0)
    axs_lst.append(plt.subplot(3, 3, i + 1))
    axs_lst[i].axis("off")
    world = path(INITIAL)
    axs_lst[i].imshow(world)
    print(f"The length of path {i+1} is", np.count_nonzero(world) - len(BLOCKS) - 1)

plt.suptitle("Question 2(t): optimal paths, beta=0")
plt.show()


# ============ PRE QUESTION 3 ============


with open('cluster_data.pickle', 'rb') as file:
    dataTrain, dataTest = pickle.load(file)
    Xtrain_pt3, Ttrain_pt3 = dataTrain
    Xtest_pt3, Ttest_pt3 = dataTest


def one_hot(Tint):
    K = np.max(Tint) + 1
    N = np.shape(Tint)[0]
    cList = np.array(range(K))
    cList = np.reshape(cList, (1, K))
    Tint = np.reshape(Tint, (N, 1))
    Thot = (Tint == cList).astype(np.float64)
    return Thot


print("\n============ Question 3 (a) ============\n")


cluster_clf = cluster.KMeans(n_clusters=3)
cluster_clf.fit(Xtrain_pt3)
prediction = cluster_clf.predict(Xtrain_pt3)
centers = cluster_clf.cluster_centers_
train_score = cluster_clf.score(Xtrain_pt3)
test_score = cluster_clf.score(Xtest_pt3)

print("Test score for 3 clusters train:", train_score)
print("Test score for 3 clusters test:", test_score)

bl2d.plot_clusters(Xtrain_pt3, one_hot(prediction))
plt.scatter(centers[:, 0], centers[:, 1], color="black")
plt.title("Question 3(a): K means")
plt.show()


print("\n============ Question 3 (b) ============\n")


mixture_clf = mixture.GaussianMixture(n_components=3, covariance_type="diag", tol=10**(-7))
mixture_clf.fit(Xtrain_pt3)
centers = mixture_clf.means_
train_score = mixture_clf.score(Xtrain_pt3)
test_score = mixture_clf.score(Xtest_pt3)

print("Test score for 3 clusters train:", train_score)
print("Test score for 3 clusters test:", test_score)

bl2d.plot_clusters(Xtrain_pt3, mixture_clf.predict_proba(Xtrain_pt3))
plt.scatter(centers[:, 0], centers[:, 1], color="black")
plt.title("Question 3(b): Gaussian mixture model (diagonal)")
plt.show()


print("\n============ Question 3 (c) ============\n")


mixture_2_clf = mixture.GaussianMixture(n_components=3, covariance_type="full", tol=10**(-7))
mixture_2_clf.fit(Xtrain_pt3)
centers = mixture_2_clf.means_
train_score = mixture_2_clf.score(Xtrain_pt3)
test_score = mixture_2_clf.score(Xtest_pt3)

print("Test score for 3 clusters train:", train_score)
print("Test score for 3 clusters test:", test_score)

bl2d.plot_clusters(Xtrain_pt3, mixture_2_clf.predict_proba(Xtrain_pt3))
plt.scatter(centers[:, 0], centers[:, 1], color="black")
plt.title("Question 3(c): Gaussian mixture model (full)")
plt.show()


print("\n============ Question 3 (e) ============\n")


print("I don't know")


print("\n============ Question 3 (h) ============\n")


MNIST_mixture_clf = mixture.GaussianMixture(n_components=10, covariance_type="diag", tol=10**(-3))
MNIST_mixture_clf.fit(Xtrain)
centers = MNIST_mixture_clf.means_
train_score = MNIST_mixture_clf.score(Xtrain)
test_score = MNIST_mixture_clf.score(Xtest)

print("Test score for 10 clusters train 50,000:", train_score)
print("Test score for 10 clusters test 50,000:", test_score)

im_centers = centers.reshape((centers.shape[0], 28, 28))

axs_lst = []
for i in range(10):
    axs_lst.append(plt.subplot(4, 3, i + 1))
    axs_lst[i].axis("off")
    axs_lst[i].imshow(im_centers[i], cmap='Greys', interpolation='nearest')

plt.suptitle("Question 3(h): mean vectors for 50,000 MNIST training points")
plt.show()


print("\n============ Question 3 (i) ============\n")


MNIST_mixture_clf = mixture.GaussianMixture(n_components=10, covariance_type="diag", tol=10**(-3))
MNIST_mixture_clf.fit(small_Xtrain)
centers = MNIST_mixture_clf.means_
train_score = MNIST_mixture_clf.score(small_Xtrain)
test_score = MNIST_mixture_clf.score(small_Xtest)

print("Test score for 10 clusters train 500:", train_score)
print("Test score for 10 clusters test 500:", test_score)

im_centers = centers.reshape((centers.shape[0], 28, 28))

axs_lst = []
for i in range(10):
    axs_lst.append(plt.subplot(4, 3, i + 1))
    axs_lst[i].axis("off")
    axs_lst[i].imshow(im_centers[i], cmap='Greys', interpolation='nearest')

plt.suptitle("Question 3(i): mean vectors for 500 MNIST training points")
plt.show()


print("\n============ Question 3 (j) ============\n")


tiny_Xtrain = Xtrain[:10]
tiny_Ttrain = Ttrain[:10]
tiny_Xval = Xval[:10]
tiny_Tval = Tval[:10]
tiny_Xtest = Xtest[:10]
tiny_Ttest = Ttest[:10]

MNIST_mixture_clf = mixture.GaussianMixture(n_components=10, covariance_type="diag", tol=10**(-3))
MNIST_mixture_clf.fit(tiny_Xtrain)
centers = MNIST_mixture_clf.means_
train_score = MNIST_mixture_clf.score(tiny_Xtrain)
test_score = MNIST_mixture_clf.score(tiny_Xtest)

print("Test score for 10 clusters train 10:", train_score)
print("Test score for 10 clusters test 10:", test_score)

im_centers = centers.reshape((centers.shape[0], 28, 28))

axs_lst = []
for i in range(10):
    axs_lst.append(plt.subplot(4, 3, i + 1))
    axs_lst[i].axis("off")
    axs_lst[i].imshow(im_centers[i], cmap='Greys', interpolation='nearest')

plt.suptitle("Question 3(j): mean vectors for 10 MNIST training points")
plt.show()
