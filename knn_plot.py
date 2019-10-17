import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import os


def plot_knn(X, y, weights: list, metrics: list, title: str = 'KNN', max_k: int = 10, z_score: bool = True,
             pca: bool = False, pca_dims: int = 10, savefig: bool = False, show: bool = True, n: int = 1):

    color_map = {
        'l1': 'b',
        'l2': 'r',
        'cosine': 'g',
        'chebyshev': 'y',
        'huber': 'm',
        'l2_log10': 'c',
        'log10': 'k'
    }

    marker_map = {
        'uniform': 'o--',
        'distance': 'o-'
    }

    legend = []

    plt.figure(dpi=150)

    for weight in weights:

        for metric in metrics:

            px = [x for x in range(1, max_k + 1)]
            py = np.zeros((len(px),))

            nidx = 0

            while nidx < n:

                kf = StratifiedKFold(n_splits=10, random_state=nidx, shuffle=True)
                kf.get_n_splits(X)

                for train_index, test_index in kf.split(X, y):
                    X_train, X_test = X[train_index].copy(), X[test_index].copy()
                    y_train, y_test = y[train_index].copy(), y[test_index].copy()

                    if z_score:
                        u = np.mean(X_train, axis=0)
                        o = np.std(X_train, axis=0)

                        X_train -= u
                        X_train /= o

                        X_test -= u
                        X_test /= o

                    if pca:
                        pca = PCA(n_components=pca_dims)
                        pca.fit(X_train)
                        X_train = pca.transform(X_train)
                        X_test = pca.transform(X_test)

                    for k in range(1, max_k + 1):
                        neigh = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weight)
                        neigh.fit(X_train, y_train)
                        score = neigh.score(X_test, y_test)

                        py[k - 1] += score

                nidx += 1

            py = 100 * py / (10*n)

            if type(metric) != str:
                metric_key = metric.__name__
            else:
                metric_key = metric

            plt.plot(px, py, '%s%s' % (color_map[metric_key], marker_map[weight]))
            legend.append("%s (%s)" % (metric_key, weight))

            for i in range(0, len(px)):
                plt.annotate("%.1f" % py[i], [px[i], py[i]])

    plt.grid()
    plt.ylabel('%d-Folds Validation Accuracy' % max_k)
    plt.xlabel('k')
    plt.xticks(px)
    # plt.title(title)
    plt.legend(legend)

    if savefig:
        plt.savefig(os.path.join('figs', "%s.png" % title.replace(':', '').replace(' ', '_')))
    elif show:
        plt.show()


def plot_knn_wrapper(kwargs):
    return plot_knn(**kwargs)


if __name__ == '__main__':
    from multiprocessing import Pool
    from metrics import log10, l2_log10

    df = pd.read_csv('data/clean.csv')

    X_cols = [c for c in df.keys()][1:]

    X = df[X_cols].values
    y = df['CLASS'].values

    weights = ['uniform', 'distance']
    metrics = ['chebyshev', 'l1', 'l2', log10, l2_log10]

    jobs = []
    pool = Pool(12)

    for metric in metrics:
        jobs.append({
            'X': X,
            'y': y,
            'weights': ['uniform', 'distance'],
            'metrics': [metric],
            'title': 'KNN: %s' % (metric if type(metric) == str else metric.__name__),
            'savefig': True,
            'show': False,
            'n': 10
        })

    metrics = ['l1', 'l2', log10, l2_log10]

    jobs.append({
        'X': X,
        'y': y,
        'weights': ['uniform'],
        'metrics': metrics,
        'title': 'KNN: Uniform',
        'savefig': True,
        'show': False,
        'n': 10
    })
    jobs.append({
        'X': X,
        'y': y,
        'weights': ['distance'],
        'metrics': metrics,
        'title': 'KNN: Distance',
        'savefig': True,
        'show': False,
        'n': 10
    })

    pool.map(plot_knn_wrapper, jobs)

