import h5py
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

BPATH = 'results'
DSETS = ['sift1m', 'deep1m', 'labelme', 'mnist']
METHODS = ['pq', 'opq', 'rvq', 'ervq', 'cq', 'lsq', 'srd', 'src']


def plot_qerror(bpath, dataset, m, n):

    a = get_field(bpath, dataset, 'train_error', 'lsq', m, n, it=25)
    b = get_field(bpath, dataset, 'train_error', 'srd', m, n, it=25)
    c = get_field(bpath, dataset, 'train_error', 'srd', m, n, it=50)
    d = get_field(bpath, dataset, 'train_error', 'srd', m, n, it=100)

    plt.plot(np.mean(a, 0), lw=2, label='LSQ')
    plt.plot(np.mean(b, 0), lw=2, label='SR-D (25)')
    plt.plot(np.mean(c, 0), lw=2, label='SR-D (50)')
    plt.plot(np.mean(d, 0), lw=2, label='SR-D (100)')

    plt.legend(loc='upper right')

    plt.show()


def compare_recalls(bpath, dataset, m, n):
    a = 100*get_r_at_1(bpath, dataset, 'lsq', m, n, it=25)
    b = 100*get_r_at_1(bpath, dataset, 'srd', m, n, it=25)
    c = 100*get_r_at_1(bpath, dataset, 'srd', m, n, it=50)
    d = 100*get_r_at_1(bpath, dataset, 'srd', m, n, it=100)

    print(' & '.join(['{:.2f}'.format(x) for x in[a, b, c, d]]))

    # TODO update with values
    plt.plot([25, 50, 100], [a, a+.1, a+.15], marker='o', label='LSQ')
    plt.plot([25, 50, 100], [b, c, d],        marker='s', label='SR-D')
    plt.legend(loc='upper left')
    plt.xlabel('Iterations')
    plt.ylabel('Recall@1')

    if dataset == 'labelme':
        plt.title('LabelMe')
    else:
        plt.title('MNIST')
    # plt.show()


def get_field(bpath, dataset, field, method, m, n, it=25, fieldname=None):
    bpath = path.join(bpath, dataset)
    fname = "{0}_m{1}_it{2}.h5".format(method, m, it)
    fname = path.join(bpath, fname)
    # print(fname)

    fields = []
    with h5py.File(fname, 'r') as f:
        for i in np.arange(n):
            # print(fname, field)
            fields.append(f['{0}/{1}'.format(i+1, field)][:])

    return np.vstack(fields)


def get_r_at_1(bpath, dataset, method, m, n, it=25):
    if method == 'cq':
        return 0.0
    if method not in ['pq', 'opq']:
        m = m - 1
    recalls = get_field(bpath, dataset, 'recall', method, m, n, it)
    return np.mean(recalls, 0)[0]


def get_r_at_1(bpath, dataset, method, m, n, it=25):
    if method == 'cq':
        return 0.0
    if method not in ['pq', 'opq']:
        m = m - 1

    recalls = get_field(bpath, dataset, 'recall', method, m, n, it)
    return np.mean(recalls, 0)[0]


def print_recalls(bpath, datasets, methods, m, n, it=25):
    for dset in datasets:
        rs = []
        for method in methods:
            r = get_r_at_1(bpath, dset, method, m, n, it)
            rs.append('{:.2f}'.format(100*r))
        print(dset, m, ' & '.join(rs))


def print_large_recalls(bpath, datasets, methods, iters, m, n, it=25):

    for dset in datasets:
        for method in methods:
            rs = []
            for iter in iters:
                recalls = get_field(bpath, dset, 'recall_{0}'.format(iter), method, m, n, it)
                rs.append('{:.4f}'.format(np.mean(recalls, 0)[0]))

            print( it, rs )


if __name__ == "__main__":

    n = 10
    # print_recalls(BPATH, DSETS, METHODS, 8, n)
    # print_recalls(BPATH, DSETS, METHODS, 16, n)

    # print_recalls(BPATH, ['sift1m'], ['srd'], 8, n, 100)
    print_large_recalls(BPATH, ['large_recalls'], ['srd'], [32,64,128,256], 8, n, 50)
    print_large_recalls(BPATH, ['large_recalls'], ['srd'], [32,64,128,256], 8, n, 100)

    # plot_qerror(BPATH, 'labelme', 7,n)
    # plot_qerror(BPATH, 'mnist', 7, n)

    # plt.figure(figsize=(8, 3))
    # plt.subplot(121)
    # compare_recalls(BPATH, 'labelme', 8, n)
    # plt.subplot(122)
    # compare_recalls(BPATH, 'mnist', 8, n)
    # plt.tight_layout()
    # plt.show()
