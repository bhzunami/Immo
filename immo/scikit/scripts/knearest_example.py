import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

n_neighbors = [15, 5, 1]
plot_titles = ['Underfit', 'Just right', 'Overfit']
plt.figure(figsize=(14, 5))
for i, k in enumerate(n_neighbors):
    knn = neighbors.KNeighborsRegressor(k, weights='uniform')
    y_ = knn.fit(X, y).predict(T)
    ax = plt.subplot(1, len(n_neighbors), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    #plt.subplot(3, 1, i + 1)
    plt.scatter(X, y, c='k', label='data', s=5)
    plt.plot(T, y_, label='prediction'.format(k))
    plt.axis('tight')

    plt.xlabel("x")
    plt.ylabel("y")
    #plt.xlim((0, 1))
    #plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title(plot_titles[i])

    #plt.legend()
    plt.title("KNeighborsRegressor (k = %i)" % (k))
plt.savefig('images/knears_overfit.png')

# for i, weights in enumerate(['uniform', 'distance']):
#     knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
#     y_ = knn.fit(X, y).predict(T)

#     plt.subplot(2, 1, i + 1)
#     plt.scatter(X, y, c='k', label='data')
#     plt.plot(T, y_, c='g', label='prediction')
#     plt.axis('tight')
#     plt.legend()
#     plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
#                                                                 weights))

# plt.show()