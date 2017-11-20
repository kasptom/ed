import math
import random
from matplotlib import pyplot

# generate points

N = 30

val_range = 30
signum_x = 1
signum_y = 1

# points = [(0, 0), (1, 1), (0, 1), (1, 0), (0.5, 1), (1, 0.5), (10, 10), (11, 11), (12, 12), (10, 12), (10, 11),
#           (11.5, 10)]

points = []

for i in range(N):
    if random.random() >= 0.5:
        signum_x = -signum_x
    if random.random() >= 0.5:
        signum_y = -signum_y
    points.append((random.random() * val_range * signum_x, random.random() * val_range * signum_y))

# dbscan
eps = 12
minPkt = 5

labels = [0 for i in range(N)]


def dist(point_a, point_b):
    return math.sqrt(math.pow(point_a[0] - point_b[0], 2) + math.pow(point_a[1] - point_b[1], 2))


def find_neighbours(idx, points, eps):
    neighs = []
    for i in range(len(points)):
        if i != idx and dist(points[i], points[idx]) <= eps:
            neighs.append(i)
    return neighs


C = 0
for i in range(N):
    if labels[i] != 0:
        continue
    neighbours = find_neighbours(i, points, eps)
    if len(neighbours) < minPkt:
        labels[i] = -1  # -1 == noise
        continue

    C += 1
    labels[i] = C

    for j in range(len(neighbours)):
        if neighbours[j] == i:
            continue
        if labels[neighbours[j]] == -1:
            labels[neighbours[j]] = C
        if labels[neighbours[j]] != 0:
            continue
        labels[neighbours[j]] = C
        ngbrs = find_neighbours(neighbours[j], points, eps)
        if len(ngbrs) >= minPkt:
            for k in range(len(ngbrs)):
                if ngbrs[k] not in neighbours:
                    neighbours.append(k)
                    labels[ngbrs[k]] = C

for i in range(N):
    print(points[i][0], points[i][1], labels[i])

    # for i in range(C):
    #     group = []
    #     for j in range(len(points)):
    #         if labels[j] == i:
    #             group.append(points[j])

    colors = [(0, 0, 0) for i in range(len(labels))]
    for value in set(labels):
        color = (random.random(), random.random(), random.random())
        for m in range(len(labels)):
            if labels[m] == value:
                colors[m] = color

    pyplot.scatter([point[0] for point in points], [point[1] for point in points], c=colors)

pyplot.show()
