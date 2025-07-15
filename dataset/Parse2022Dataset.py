import open3d as o3d
import torch
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
import trimesh
import sys
from os.path import join
from numpy import linalg as LA
import json
import pathlib

root_path = str(pathlib.Path(__file__).absolute().parent.parent)
sys.path.append(f"{root_path}/utils/")
from pcutils import normalize, make_holes_pcd_2, make_holes_pcd_3, make_holes_base, get_rotation_x, get_rotation_z, \
    add_rotation_to_pcloud, make_holes_horizontally, augmented_normalize


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]


def rotate_pcd_shapeNet(pcd, posA=1, posB=2):
    n = pcd.shape[0]
    for i in range(n):
        temp = pcd[i][posA]
        pcd[i][posA] = pcd[i][posB]
        pcd[i][posB] = temp
    return pcd


class Parse2022Dataset(data.Dataset):
    def __init__(self, dir_labels, dir_preds_fixed, dir_holes, n_partial_models, npoints=2048, do_holes=False, function=None,
                 train = True):
        self.dir_labels = dir_labels
        self.dir_preds_fixed = dir_preds_fixed
        self.dir_holes = dir_holes
        self.train = train

        def convert_to_str_list(path_list):
            return [str(path).replace('\\', '/') for path in path_list]

        self.labels = convert_to_str_list(sorted(pathlib.Path(dir_labels).glob("*.pcd")))
        self.preds_fixed = convert_to_str_list(sorted(pathlib.Path(dir_preds_fixed).glob("*.pcd")))
        self.holes = convert_to_str_list(sorted(pathlib.Path(dir_holes).glob("*.pcd")))

        index_3d_uniques = list(set([pathlib.Path(label).stem.split("_")[0] for label in self.labels]))

        split_value = 0.9
        index_3d_train = index_3d_uniques[:int(split_value * len(index_3d_uniques))]
        index_3d_test = index_3d_uniques[int(split_value * len(index_3d_uniques)):]
        
        # print(f"Train: {sorted(index_3d_train)} - Test: {sorted(index_3d_test)}")

        def split_data(data_list):
            if self.train:
                data_list = [data for data in data_list if pathlib.Path(data).stem.split("_")[0] in index_3d_train]
            else:
                data_list = [data for data in data_list if pathlib.Path(data).stem.split("_")[0] in index_3d_test]
            return data_list
        
        # def split_data(data_list):
        #     if self.train:
        #         data_list = data_list[:int(split_value * len(data_list))]
        #     else:
        #         data_list = data_list[int(split_value * len(data_list)):]
        #     return data_list

        self.labels = split_data(self.labels)
        self.preds_fixed = split_data(self.preds_fixed)
        self.holes = split_data(self.holes)

        self.npoints = npoints
        self.n_partial_models = n_partial_models

        self.len = len(self.labels) * n_partial_models

        self.do_holes = do_holes
        self.function = function

    def __getitem__(self, index):
        label_data = self.labels[index]
        pred_fixed_data = self.preds_fixed[index]
        hole_data = self.holes[index]

        def read_pcd(filename, n_points=0):
            if filename[filename.rfind('.') + 1:] == 'pcd':
                pcd = o3d.io.read_point_cloud(filename)
                pcd = np.array(pcd.points)
                if n_points != 0:
                    pcd = resample_pcd(pcd, n_points)
            else:
                pcd = trimesh.load(filename).sample(n_points)
            return pcd

        name = pathlib.Path(label_data).name

        # complete = trimesh.load(self.complete_dir + "/" + join( rpath, name)).sample(self.npoints )
        complete = read_pcd(label_data, self.npoints)
        partial = read_pcd(pred_fixed_data, self.npoints)
        hole = read_pcd(hole_data, self.npoints)

        if self.do_holes:
            scale_shift = random.uniform(-0.5, 0.5)
            complete = augmented_normalize(complete, unit_ball=False, rand_shift=scale_shift)

            # rotations
            rot_z = get_rotation_z(np.deg2rad(random.uniform(0, 360)))
            rot_x = get_rotation_x(np.deg2rad(random.uniform(0, 15)))
            rotation_mat = np.dot(rot_x, rot_z)
            complete = add_rotation_to_pcloud(complete, rotation_mat)

            # holes
            partial, hole = make_holes_base(complete, [0.05, 0.175])
            # partial, hole = make_holes_pcd_3(complete, [0.05, 0.15])

            # translations
            # print(f'Partial shape: {partial.shape} - complete shape: {complete.shape}')
            z_shift = random.uniform(-0.3, 0.3)
            partial = partial + np.array([0, 0, z_shift])
            hole = hole + np.array([0, 0, z_shift])
            complete = complete + np.array([0, 0, z_shift])

            x_shift = random.uniform(-0.03, 0.03)
            partial = partial + np.array([x_shift, 0, 0])
            hole = hole + np.array([x_shift, 0, 0])
            complete = complete + np.array([x_shift, 0, 0])

            y_shift = random.uniform(-0.03, 0.03)
            partial = partial + np.array([0, y_shift, 0])
            hole = hole + np.array([0, y_shift, 0])
            complete = complete + np.array([0, y_shift, 0])

            # partial = add_rotation_to_pcloud(partial, rotation_mat)
            # hole = add_rotation_to_pcloud(hole, rotation_mat)
        else:
            complete = normalize(complete, unit_ball=False)
            partial = normalize(partial, unit_ball=False)
            hole = normalize(hole, unit_ball=False)

        # print(partial)
        # print(complete)
        # print(hole)
        return name, resample_pcd(partial, self.npoints), resample_pcd(hole, self.npoints // 2), resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len


def save_point_cloud(filename, pcd):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(filename, pc)


# Preprocess
def convert_nifti_to_pcd(dir_labels, dir_preds_fixed, dir_holes):
    import nibabel as nib
    import numpy as np
    from tqdm import tqdm

    pattern = ".nii.gz"
    labels_paths = sorted(pathlib.Path(dir_labels).glob(f"*{pattern}"))
    preds_fixed_paths = sorted(pathlib.Path(dir_preds_fixed).glob(f"*{pattern}"))

    def load_nifti_file(nifti_file):
        img = nib.load(nifti_file)
        data = img.get_fdata()
        return data

    def save_numpy_as_pcd(data, data_path):
        # Get the coordinates of non-zero points
        coords = np.column_stack(np.nonzero(data))
        # Convert to float32 for Open3D compatibility
        coords = coords.astype(np.float64)
        # Save to PCD file
        if coords.size == 0:
            coords = np.zeros((1, 3), dtype=np.float64)  # Ensure at least one point
        save_path = str(data_path).replace(pattern, '.pcd')
        save_point_cloud(save_path, coords)

    for i in tqdm(range(len(labels_paths))):
        label_path = labels_paths[i]
        pred_fixed_path = preds_fixed_paths[i]

        # Load NIfTI files
        label_data = load_nifti_file(label_path)
        pred_fixed_data = load_nifti_file(pred_fixed_path)

        # Pred fix does not have outliers, so we can use it directly
        hole_data = np.logical_xor(label_data, pred_fixed_data)
        hole_path = pathlib.Path(dir_holes).joinpath(pathlib.Path(label_path).name)

        # Save as PCD
        save_numpy_as_pcd(label_data, label_path)
        save_numpy_as_pcd(pred_fixed_data, pred_fixed_path)
        save_numpy_as_pcd(hole_data, hole_path)



if __name__ == '__main__':
    # V1
    # dir_labels = f"{root_path}/data/parse2022/labels"
    # dir_preds_fixed = f"{root_path}/data/parse2022/preds_fixed"
    # # Created by the script
    # dir_holes = f"{root_path}/data/parse2022/holes"

    # V2
    dir_labels = f"{root_path}/../TreesAutoEncoder/data_crops/parse2022_LC_64_50/labels_3d"
    dir_preds_fixed = f"{root_path}/../TreesAutoEncoder/data_crops/parse2022_LC_64_50/preds_fixed_3d"
    # Created by the script
    dir_holes = f"{root_path}/../TreesAutoEncoder/data_crops/parse2022_LC_64_50/holes_3d"


    # Preprocess NIfTI files to PCD
    if not os.path.exists(dir_holes):
        os.makedirs(dir_holes, exist_ok=True)
        convert_nifti_to_pcd(dir_labels, dir_preds_fixed, dir_holes)

    # Create dataset instance
    dataset = Parse2022Dataset(dir_labels, dir_preds_fixed, dir_holes, 1, npoints=2048)

    # Example usage
    name, partial, hole, complete = dataset[0]
    save_point_cloud('complete.xyz', complete)
    save_point_cloud('partial.xyz', partial)
    save_point_cloud('hole.xyz', hole)
