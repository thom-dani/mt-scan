import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import mt_scan
from sklearn.metrics import adjusted_rand_score



def sample_from_grid_distribution(distribution, n_samples):

    distribution_max = distribution.max()
    resolution = distribution.shape[0]
    points = []

    while len(points) < n_samples:
        
        x = np.random.uniform(0, resolution)
        y = np.random.uniform(0, resolution)

        ix = min(int(x), resolution - 1)
        iy = min(int(y), resolution - 1)

        density = distribution[iy, ix]

        u = np.random.uniform(0, distribution_max)

        if u < density:
            points.append([x, y])

    return np.array(points, dtype=np.float32)


def generate_ring_distribution(resolution):

    distribution=np.zeros((resolution, resolution), dtype=np.float64)

    for  x in range(resolution):
        for y in range(resolution):
            middle_x = x + 1./2
            middle_y = y + 1./2
            R_corner_1 = np.sqrt((x + 1./2- float(resolution)/2)**2 + (y + 1./2 - float(resolution)/2)**2) 
            R_corner_2 = np.sqrt((x + 1./2- float(resolution)/2)**2 + (y - 1./2 - float(resolution)/2)**2) 
            R_corner_3 = np.sqrt((x - 1./2- float(resolution)/2)**2 + (y + 1./2 - float(resolution)/2)**2) 
            R_corner_4 = np.sqrt((x - 1./2- float(resolution)/2)**2 + (y - 1./2 - float(resolution)/2)**2) 
            R_max=np.max([R_corner_1, R_corner_2, R_corner_3, R_corner_4])
            R_min=np.min([R_corner_1, R_corner_2, R_corner_3, R_corner_4])
            if(R_min >= R1) & (R_max < R2):
                distribution[x, y] = 2.0
            if(R_min >= R3) & (R_max < R4):
                distribution[x, y]= 0.75
            if (R_min >= R5) & (R_max < R6):
                distribution[x, y] = 0.25

    return distribution

def compute_labels_ground_truth(points, resolution):
    labels=np.zeros(points.shape[0], dtype=int)
    for i,p in enumerate(points):
        x=p[0]
        y=p[1]
        R = np.sqrt((x - resolution/2)**2 + (y - resolution/2)**2)
        if (R >= R1) and (R < R2):
            labels[i]=0
        if(R >= R3) and (R < R4):
            labels[i]=1
        if (R >= R5) and (R < R6):
            labels[i]=2
    return labels 

def cells_to_vertices(distribution):

    res_y, res_x = distribution.shape
    V = np.zeros((res_y + 1, res_x + 1), dtype=distribution.dtype)

    count = np.zeros_like(V)

    V[1:, 1:] += distribution
    count[1:, 1:] += 1

    V[1:, :-1] += distribution
    count[1:, :-1] += 1

    V[:-1, 1:] += distribution
    count[:-1, 1:] += 1

    V[:-1, :-1] += distribution
    count[:-1, :-1] += 1

    return V / count



def plot(x, y, distribution, points):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(distribution, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="plasma")
    axes[0].set_title("Distribution")

    axes[1].scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
    axes[1].set_title("Sampled points")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.show()


def save_points_csv(points, labels, path):
    data = np.column_stack((points, labels))
    np.savetxt(path, data, delimiter=",", header="x,y,labels", comments="")


if __name__ == "__main__":

    resolution      = 256

    R1=0 
    R2=0.46*resolution/2 
    R3=0.48*resolution/2 
    R4=0.68*resolution/2 
    R5=0.7*resolution/2 
    R6=0.9*resolution/2 

    distribution = generate_ring_distribution(resolution)
    distribution_vertices = cells_to_vertices(distribution)
    n_points_range=range(1_000, 10_001, 1_000)

"""
    for i in n_points_range:

        points=sample_from_grid_distribution(distribution, n_samples=i)
        labels_ground_truth=compute_labels_ground_truth(points, resolution)
        labels_distrib=mt_scan.compute_labels_distribution(points, distribution_vertices.astype(np.float32))
        #labels_distrib=mt_scan.compute_labels(points, 50, 512)
        ari_distribution=adjusted_rand_score(labels_ground_truth, labels_distrib)
        print(f"n_points: {i}, ari: {ari_distribution}")

"""

n_points=1000

points=sample_from_grid_distribution(distribution, n_samples=n_points)
labels_ground_truth=compute_labels_ground_truth(points, resolution)
save_points_csv(points, labels_ground_truth, "../tmp/points.csv")
labels_distrib=mt_scan.compute_labels_distribution(points, distribution_vertices.astype(np.float32))
#labels_distrib=mt_scan.compute_labels(points, 50, 512)

ari_distribution=adjusted_rand_score(labels_ground_truth, labels_distrib)
print(f"n_points: {n_points}, ari: {ari_distribution}")


"""
    plt.figure(figsize=(7, 7))

    plt.imshow(
        distribution,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap="viridis",
        alpha=0.4,
    )

    # Plot sampled points
    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=2,
        alpha=0.7,
    )

    plt.gca().set_aspect("equal")

    plt.title("Samples")
    plt.show()
"""




