import open3d as o3d
import numpy as np


def load_and_show_xyz(filename):
    # Load .xyz file into numpy
    points = np.loadtxt(filename)

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Show
    o3d.visualization.draw_geometries([pcd], window_name=filename)


if __name__ == "__main__":
    files = [
        "complete.xyz",
        "hole.xyz",
        "partial.xyz"
    ]

    for fname in files:
        load_and_show_xyz(fname)
