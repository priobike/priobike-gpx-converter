import csv

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def on_pick(event):
    i = event.ind[0]
    print(f'd: {d[i]}, #approx_points: {n_approx_points[i]}, a_threshold: {a[i]}, d_threshold_1: {d1[i]}, d_threshold_2: {d2[i]}')


with open('result.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    result = []
    for row in reader:
        result.append({
            'number_of_gpx_points': int(row[0].split(' ')[-1]),
            'number_of_approx_points': int(row[1].split(' ')[-1]),
            'a_threshold': float(row[2].split(' ')[-1]),
            'd_threshold_1': float(row[3].split(' ')[-1]),
            'd_threshold_2': float(row[4].split(' ')[-1]),
            'd': float(row[5].split(' ')[-1])
        })
    n_approx_points = [x['number_of_approx_points'] for x in result]
    a = [x['a_threshold'] for x in result]
    d1 = [x['d_threshold_1'] for x in result]
    d2 = [x['d_threshold_2'] for x in result]
    d = [x['d'] for x in result]
    d_num = np.multiply(d, n_approx_points)
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(a, d1, d2, c=d, cmap='plasma')
    ax.set_title('d')
    ax.set_xlabel('a threshold')
    ax.set_ylabel('d1 threshold')
    ax.set_zlabel('d2 threshold')
    plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min(d), vmax=max(d)), cmap='plasma'),
                 ax=ax, orientation='horizontal')

    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax1.scatter(a, d1, d2, c=n_approx_points, cmap='plasma')
    ax1.set_title('#approx points')
    ax1.set_xlabel('a threshold')
    ax1.set_ylabel('d1 threshold')
    ax1.set_zlabel('d2 threshold')
    plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min(n_approx_points), vmax=max(n_approx_points)), cmap='plasma'),
                 ax=ax1, orientation='horizontal')

    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    ax2.scatter(a, d1, d2, c=d_num, cmap='plasma')
    ax2.set_title('d * #approx_points')
    ax2.set_xlabel('a threshold')
    ax2.set_ylabel('d1 threshold')
    ax2.set_zlabel('d2 threshold')
    plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min(d_num), vmax=max(d_num)), cmap='plasma'),
                 ax=ax2, orientation='horizontal')
    argmin = np.argmin(d_num)
    print(f'd: {d[argmin]}, #approx_points: {n_approx_points[argmin]}, a_threshold: {a[argmin]}, d_threshold_1: {d1[argmin]}, d_threshold_2: {d2[argmin]}')

    fig1 = plt.figure()
    ax3 = fig1.add_subplot()
    plt.grid()
    ax3.scatter(d, n_approx_points, picker=True)
    fig1.canvas.mpl_connect('pick_event', on_pick)
    ax3.set_xlabel('d')
    ax3.set_ylabel('n_approx_points')
    plt.show()
