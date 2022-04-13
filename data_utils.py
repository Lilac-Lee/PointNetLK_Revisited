""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), 
Deep Global Registration (https://github.com/chrischoy/DeepGlobalRegistration),
SECOND (https://github.com/traveller59/second.pytorch), modified. """

import os
import glob
import numpy as np
import torch
import torch.utils.data
import six
import copy
import csv
import open3d as o3d

import utils


def load_3dmatch_batch_data(p0_fi, p1_fi, voxel_ratio):
    p0 = np.load(p0_fi)['pcd']
    p1 = np.load(p1_fi)['pcd']
    
    # voxelization
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(p0)
    p0_downsampled_pcd = pcd0.voxel_down_sample(voxel_size=voxel_ratio)   # open3d 0.8.0.0+
    p0_downsampled = np.array(p0_downsampled_pcd.points)
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(p1)
    p1_downsampled_pcd = pcd1.voxel_down_sample(voxel_size=voxel_ratio)   # open3d 0.8.0.0+
    p1_downsampled = np.array(p1_downsampled_pcd.points)
    
    return p0_downsampled, p1_downsampled
    

def find_voxel_overlaps(p0, p1, voxel):
    xmin, ymin, zmin = np.max(np.stack([np.min(p0, 0), np.min(p1, 0)]), 0)
    xmax, ymax, zmax = np.min(np.stack([np.max(p0, 0), np.max(p1, 0)]), 0)
    
    # truncate the point cloud
    eps = 1e-6
    p0_ = p0[np.all(p0>[xmin+eps,ymin+eps,zmin+eps], axis=1) & np.all(p0<[xmax-eps,ymax-eps,zmax-eps], axis=1)]
    p1_ = p1[np.all(p1>[xmin+eps,ymin+eps,zmin+eps], axis=1) & np.all(p1<[xmax-eps,ymax-eps,zmax-eps], axis=1)]
    
    # recalculate the constraints
    xmin, ymin, zmin = np.max(np.stack([np.min(p0, 0), np.min(p1, 0)]), 0)
    xmax, ymax, zmax = np.min(np.stack([np.max(p0, 0), np.max(p1, 0)]), 0)
    vx = (xmax - xmin) / voxel
    vy = (ymax - ymin) / voxel
    vz = (zmax - zmin) / voxel
    
    return p0_, p1_, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz


class ThreeDMatch_Testing(torch.utils.data.Dataset):
    def __init__(self, dataset_path, category, overlap_ratio, voxel_ratio, voxel, max_voxel_points, num_voxels, rigid_transform, vis, voxel_after_transf):
        self.dataset_path = dataset_path
        self.pairs = []
        with open(category, 'r') as fi:
            cinfo_fi = fi.read().split()   # category names
            for i in range(len(cinfo_fi)):
                cat_name = cinfo_fi[i]
                cinfo_name = cat_name + '*%.2f.txt' % overlap_ratio
                cinfo = glob.glob(os.path.join(self.dataset_path, cinfo_name))
                for fi_name in cinfo:
                    with open(fi_name) as fi:
                        fi_list = [x.strip().split() for x in fi.readlines()]
                    for fi in fi_list:
                        self.pairs.append([fi[0], fi[1]])
                        
        self.voxel_ratio = voxel_ratio
        self.voxel = int(voxel)
        self.max_voxel_points = max_voxel_points
        self.num_voxels = num_voxels
        self.perturbation = load_pose(rigid_transform, len(self.pairs))
        self.vis = vis
        self.voxel_after_transf = voxel_after_transf
        
    def __len__(self):
        return len(self.pairs)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6], twist-params
        g = utils.exp(x).to(p0) # [1, 4, 4]
        p1 = utils.transform(g, p0)
        igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt
    
    def __getitem__(self, index):
        p0_pre, p1_pre = load_3dmatch_batch_data(os.path.join(self.dataset_path, self.pairs[index][0]), os.path.join(self.dataset_path, self.pairs[index][1]), self.voxel_ratio)
        
        if self.voxel_after_transf:
            x = torch.from_numpy(self.perturbation[index][np.newaxis,...])
            p1_pre, igt = self.do_transform(torch.from_numpy(p1_pre).double(), x)
        
            p0_pre_mean = np.mean(p0_pre,0)
            p1_pre_mean = np.mean(p1_pre.numpy(),0)
            p0_pre_ = p0_pre - p0_pre_mean
            p1_pre_ = p1_pre.numpy() - p1_pre_mean
            
            # voxelization
            p0, p1, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz = find_voxel_overlaps(p0_pre_, p1_pre_, self.voxel)   # constraints of P1 ^ P2, where contains roughly overlapped area
            
            p0 = p0 + p0_pre_mean
            p1 = p1 + p1_pre_mean
            xmin0 = xmin + p0_pre_mean[0]
            ymin0 = ymin + p0_pre_mean[1]
            zmin0 = zmin + p0_pre_mean[2]
            xmax0 = xmax + p0_pre_mean[0]
            ymax0 = ymax + p0_pre_mean[1]
            zmax0 = zmax + p0_pre_mean[2]

            xmin1 = xmin + p1_pre_mean[0]
            ymin1 = ymin + p1_pre_mean[1]
            zmin1 = zmin + p1_pre_mean[2]
            xmax1 = xmax + p1_pre_mean[0]
            ymax1 = ymax + p1_pre_mean[1]
            zmax1 = zmax + p1_pre_mean[2]
            
            voxels_p0, coords_p0, num_points_per_voxel_p0 = points_to_voxel_second(p0, (xmin0, ymin0, zmin0, xmax0, ymax0, zmax0), 
                            (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
            voxels_p1, coords_p1, num_points_per_voxel_p1 = points_to_voxel_second(p1, (xmin1, ymin1, zmin1, xmax1, ymax1, zmax1), 
                            (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
        else:
            # voxelization
            p0, p1, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz = find_voxel_overlaps(p0_pre, p1_pre, self.voxel)   # constraints of P1 ^ P2, where contains roughly overlapped area
            voxels_p0, coords_p0, num_points_per_voxel_p0 = points_to_voxel_second(p0, (xmin, ymin, zmin, xmax, ymax, zmax), 
                            (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
            voxels_p1, coords_p1, num_points_per_voxel_p1 = points_to_voxel_second(p1, (xmin, ymin, zmin, xmax, ymax, zmax), 
                            (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
        
        coords_p0_idx = coords_p0[:,1]*(int(self.voxel**2)) + coords_p0[:,0]*(int(self.voxel)) + coords_p0[:,2]
        coords_p1_idx = coords_p1[:,1]*(int(self.voxel**2)) + coords_p1[:,0]*(int(self.voxel)) + coords_p1[:,2]
        
        if self.voxel_after_transf:
            # calculate for the voxel medium
            xm_x0 = np.linspace(xmin0+vx/2, xmax0-vx/2, int(self.voxel))
            xm_y0 = np.linspace(ymin0+vy/2, ymax0-vy/2, int(self.voxel))
            xm_z0 = np.linspace(zmin0+vz/2, zmax0-vz/2, int(self.voxel))
            mesh3d0 = np.vstack(np.meshgrid(xm_x0,xm_y0,xm_z0)).reshape(3,-1).T
            xm_x1 = np.linspace(xmin1+vx/2, xmax1-vx/2, int(self.voxel))
            xm_y1 = np.linspace(ymin1+vy/2, ymax1-vy/2, int(self.voxel))
            xm_z1 = np.linspace(zmin1+vz/2, zmax1-vz/2, int(self.voxel))
            mesh3d1 = np.vstack(np.meshgrid(xm_x1,xm_y1,xm_z1)).reshape(3,-1).T
            
            voxel_coords_p0 = mesh3d0[coords_p0_idx]
            voxel_coords_p1 = mesh3d1[coords_p1_idx]
        else:
            # calculate for the voxel medium
            xm_x = np.linspace(xmin+vx/2, xmax-vx/2, int(self.voxel))
            xm_y = np.linspace(ymin+vy/2, ymax-vy/2, int(self.voxel))
            xm_z = np.linspace(zmin+vz/2, zmax-vz/2, int(self.voxel))
            mesh3d = np.vstack(np.meshgrid(xm_x,xm_y,xm_z)).reshape(3,-1).T
            voxel_coords_p0 = mesh3d[coords_p0_idx]
            voxel_coords_p1 = mesh3d[coords_p1_idx]
            
        # find voxels where number of points >= 80% of the maximum number of points
        idx_conditioned_p0 = coords_p0_idx[np.where(num_points_per_voxel_p0>=0.1*self.max_voxel_points)]
        idx_conditioned_p1 = coords_p1_idx[np.where(num_points_per_voxel_p1>=0.1*self.max_voxel_points)]
        idx_conditioned, _, _ = np.intersect1d(idx_conditioned_p0, idx_conditioned_p1, assume_unique=True, return_indices=True)
        _, _, idx_p0 = np.intersect1d(idx_conditioned, coords_p0_idx, assume_unique=True, return_indices=True)
        _, _, idx_p1 = np.intersect1d(idx_conditioned, coords_p1_idx, assume_unique=True, return_indices=True)
        voxel_coords_p0 = voxel_coords_p0[idx_p0]
        voxel_coords_p1 = voxel_coords_p1[idx_p1]
        voxels_p0 = voxels_p0[idx_p0]
        voxels_p1 = voxels_p1[idx_p1]
        
        if not self.voxel_after_transf:
            x = torch.from_numpy(self.perturbation[index][np.newaxis,...])
            voxels_p1_, igt = self.do_transform(torch.from_numpy(voxels_p1.reshape(-1,3)), x)
            voxels_p1 = voxels_p1_.reshape(voxels_p1.shape)
            voxel_coords_p1, _ = self.do_transform(torch.from_numpy(voxel_coords_p1).double(), x)
            p1, _ = self.do_transform(torch.from_numpy(p1), x)
        
        if self.vis:
            return voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt, p0, p1
        else:    
            return voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt


class ToyExampleData(torch.utils.data.Dataset):
    def __init__(self, p0, p1, voxel_ratio, voxel, max_voxel_points, num_voxels, rigid_transform, vis):
        self.voxel_ratio = voxel_ratio
        self.voxel = int(voxel)
        self.max_voxel_points = max_voxel_points
        self.num_voxels = num_voxels
        self.perturbation = rigid_transform
        self.p0 = p0
        self.p1 = p1
        self.vis = vis

    def __len__(self):
        return len(self.p0)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6], twist-params
        g = utils.exp(x).to(p0) # [1, 4, 4]
        p1 = utils.transform(g, p0)
        igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt

    def __getitem__(self, index):
        p0_pre = self.p0[index]
        p1_pre = self.p1[index]
        
        # voxelization
        p0, p1, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz = find_voxel_overlaps(p0_pre, p1_pre, self.voxel)   # constraints of P1 ^ P2, where contains roughly overlapped area
        voxels_p0, coords_p0, num_points_per_voxel_p0 = points_to_voxel_second(p0, (xmin, ymin, zmin, xmax, ymax, zmax), 
                        (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
        voxels_p1, coords_p1, num_points_per_voxel_p1 = points_to_voxel_second(p1, (xmin, ymin, zmin, xmax, ymax, zmax), 
                        (vx, vy, vz), self.max_voxel_points, reverse_index=False, max_voxels=self.num_voxels)
        
        coords_p0_idx = coords_p0[:,1]*(int(self.voxel**2)) + coords_p0[:,0]*(int(self.voxel)) + coords_p0[:,2]
        coords_p1_idx = coords_p1[:,1]*(int(self.voxel**2)) + coords_p1[:,0]*(int(self.voxel)) + coords_p1[:,2]
        
        # calculate for the voxel medium
        xm_x = np.linspace(xmin+vx/2, xmax-vx/2, int(self.voxel))
        xm_y = np.linspace(ymin+vy/2, ymax-vy/2, int(self.voxel))
        xm_z = np.linspace(zmin+vz/2, zmax-vz/2, int(self.voxel))
        mesh3d = np.vstack(np.meshgrid(xm_x,xm_y,xm_z)).reshape(3,-1).T
        voxel_coords_p0 = mesh3d[coords_p0_idx]
        voxel_coords_p1 = mesh3d[coords_p1_idx]
        
        # find voxels where number of points >= 80% of the maximum number of points
        idx_conditioned_p0 = coords_p0_idx[np.where(num_points_per_voxel_p0>=0.1*self.max_voxel_points)]
        idx_conditioned_p1 = coords_p1_idx[np.where(num_points_per_voxel_p1>=0.1*self.max_voxel_points)]
        idx_conditioned, _, _ = np.intersect1d(idx_conditioned_p0, idx_conditioned_p1, assume_unique=True, return_indices=True)
        _, _, idx_p0 = np.intersect1d(idx_conditioned, coords_p0_idx, assume_unique=True, return_indices=True)
        _, _, idx_p1 = np.intersect1d(idx_conditioned, coords_p1_idx, assume_unique=True, return_indices=True)
        voxel_coords_p0 = voxel_coords_p0[idx_p0]
        voxel_coords_p1 = voxel_coords_p1[idx_p1]
        voxels_p0 = voxels_p0[idx_p0]
        voxels_p1 = voxels_p1[idx_p1]
        
        x = torch.from_numpy(self.perturbation[index][np.newaxis,...])
        voxels_p1_, igt = self.do_transform(torch.from_numpy(voxels_p1.reshape(-1,3)), x)
        voxels_p1 = voxels_p1_.reshape(voxels_p1.shape)
        voxel_coords_p1, _ = self.do_transform(torch.from_numpy(voxel_coords_p1).double(), x)
        p1, _ = self.do_transform(torch.from_numpy(p1), x)
        
        if self.vis:
            return voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt, p0, p1
        else:
            return voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt
    
    
class RandomTransformSE3:
    """ randomly generate rigid transformations """

    def __init__(self, mag=1, mag_randomly=True):
        self.mag = mag
        self.randomly = mag_randomly
        self.gt = None
        self.igt = None

    def generate_transform(self):
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6], twist params
        g = utils.exp(x).to(p0)   # [1, 4, 4]
        gt = utils.exp(-x).to(p0)  # [1, 4, 4]
        p1 = utils.transform(g, p0)
        self.gt = gt   # p1 --> p0
        self.igt = g   # p0 --> p1
        
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)


def add_noise(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += torch.clamp(sigma * torch.randn(N, C), -1 * clip, clip)

    return pointcloud


class PointRegistration(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, sigma=0.00, clip=0.00):
        self.dataset = dataset
        self.transf = rigid_transform
        self.sigma = sigma
        self.clip = clip

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _ = self.dataset[index]   # one point cloud
        p_ = add_noise(pm, sigma=self.sigma, clip=self.clip)
        p1 = self.transf(p_)
        igt = self.transf.igt.squeeze(0)
        p0 = pm
        
        # p0: template, p1: source, igt:transform matrix from p0 to p1
        return p0, p1, igt
        

class PointRegistration_fixed_perturbation(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, sigma=0.00, clip=0.00):
        torch.manual_seed(713)
        self.dataset = dataset
        self.transf_ = load_pose(rigid_transform, len(self.dataset))
        list_order = torch.randperm(len(self.dataset))
        self.transf = self.transf_[list_order]
        self.sigma = sigma
        self.clip = clip

    def __len__(self):
        return len(self.dataset)

    def transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6], twist-vector (rotation and translation)
        g = utils.exp(x).to(p0)   # [1, 4, 4]
        p1 = utils.transform(g, p0)
        igt = g.squeeze(0)
        
        return p1, igt
    
    def __getitem__(self, index):
        pm, _ = self.dataset[index]   # one point cloud
        p_ = add_noise(pm, sigma=self.sigma, clip=self.clip)
        p0 = pm
        x = torch.from_numpy(self.transf[index][np.newaxis, ...]).to(p0)
        p1, igt = self.transform(p_, x)
        
        # p0: template, p1: source, igt:transform matrix from p0 to p1
        return p0, p1, igt
        
        
# adapted from SECOND: https://github.com/nutonomy/second.pytorch/blob/master/second/core/point_cloud/point_cloud_ops.py
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coords_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    ndim = 3
    grid_size = (coords_range[3:] - coords_range[:3]) / voxel_size
    grid_size = np.around(grid_size, 0, grid_size).astype(np.int32)

    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coords_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            # print(voxel_num)
            if voxel_num > max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num


# adapted from SECOND: https://github.com/nutonomy/second.pytorch/blob/master/second/core/point_cloud/point_cloud_ops.py
def points_to_voxel_second(points,
                     coords_range,
                     voxel_size,
                     max_points=100,
                     reverse_index=False,
                     max_voxels=20000):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.
    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coords_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.
    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coords_range, np.ndarray):
        coords_range = np.array(coords_range, dtype=points.dtype)
    voxelmap_shape = (coords_range[3:] - coords_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.around(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.ones(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype) * np.mean(points, 0)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_num = _points_to_voxel_kernel(
        points, voxel_size, coords_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel


def load_pose(trans_pth, num_pose):
    with open(trans_pth, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        poses = []
        for row in csvreader:
            row = [float(i) for i in row]
            poses.append(row)
        init_gt = np.array(poses)[:num_pose]
    print('init_trans shape is {}'.format(init_gt.shape))
    
    return init_gt


def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []

    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue

        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue

        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
                
    return samples


class Globset(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """
    def __init__(self, rootdir, pattern, fileloader, transform=None, classinfo=None):
        super().__init__()

        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        self.rootdir = rootdir
        self.pattern = pattern
        self.fileloader = fileloader
        self.transform = transform

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.rootdir)
        fmt_str += '    File Patterns: {}\n'.format(self.pattern)
        fmt_str += '    File Loader: {}\n'.format(self.fileloader)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.fileloader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def num_classes(self):
        return len(self.classes)

    def class_name(self, cidx):
        return self.classes[cidx]

    def indices_in_class(self, cidx):
        targets = np.array(list(map(lambda s: s[1], self.samples)))
        return np.where(targets == cidx).tolist()

    def select_classes(self, cidxs):
        indices = []
        for i in cidxs:
            idxs = self.indices_in_class(i)
            indices.extend(idxs)
        return indices

    def split(self, rate):
        """ dateset -> dataset1, dataset2. s.t.
            len(dataset1) = rate * len(dataset),
            len(dataset2) = (1-rate) * len(dataset)
        """
        orig_size = len(self)
        select = np.zeros(orig_size, dtype=int)
        csize = np.zeros(len(self.classes), dtype=int)
        dsize = np.zeros(len(self.classes), dtype=int)

        for i in range(orig_size):
            _, target = self.samples[i]
            csize[target] += 1
        dsize = (csize * rate).astype(int)
        for i in range(orig_size):
            _, target = self.samples[i]
            if dsize[target] > 0:
                select[i] = 1
                dsize[target] -= 1

        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        samples1 = list(map(lambda i: dataset1.samples[i], np.where(select == 1)[0]))
        samples2 = list(map(lambda i: dataset2.samples[i], np.where(select == 0)[0]))

        dataset1.samples = samples1
        dataset2.samples = samples2
        return dataset1, dataset2


class Mesh:
    def __init__(self):
        self._vertices = [] # array-like (N, D)
        self._faces = [] # array-like (M, K)
        self._edges = [] # array-like (L, 2)

    def clone(self):
        other = copy.deepcopy(self)
        return other

    def clear(self):
        for key in self.__dict__:
            self.__dict__[key] = []

    def add_attr(self, name):
        self.__dict__[name] = []

    @property
    def vertex_array(self):
        return np.array(self._vertices)

    @property
    def vertex_list(self):
        return list(map(tuple, self._vertices))

    @staticmethod
    def faces2polygons(faces, vertices):
        p = list(map(lambda face: \
                        list(map(lambda vidx: vertices[vidx], face)), faces))
        return p

    @property
    def polygon_list(self):
        p = Mesh.faces2polygons(self._faces, self._vertices)
        return p

    def on_unit_sphere(self, zero_mean=False):
        # radius == 1
        v = self.vertex_array # (N, D)
        if zero_mean:
            a = np.mean(v[:, 0:3], axis=0, keepdims=True) # (1, 3)
            v[:, 0:3] = v[:, 0:3] - a
        n = np.linalg.norm(v[:, 0:3], axis=1) # (N,)
        m = np.max(n) # scalar
        v[:, 0:3] = v[:, 0:3] / m
        self._vertices = v
        return self

    def on_unit_cube(self, zero_mean=False):
        # volume == 1
        v = self.vertex_array # (N, D)
        if zero_mean:
            a = np.mean(v[:, 0:3], axis=0, keepdims=True) # (1, 3)
            v[:, 0:3] = v[:, 0:3] - a
        m = np.max(np.abs(v)) # scalar
        v[:, 0:3] = v[:, 0:3] / (m * 2)
        self._vertices = v
        return self

    def rot_x(self):
        # camera local (up: +Y, front: -Z) -> model local (up: +Z, front: +Y).
        v = self.vertex_array
        t = np.copy(v[:, 1])
        v[:, 1] = -np.copy(v[:, 2])
        v[:, 2] = t
        self._vertices = list(map(tuple, v))
        return self

    def rot_zc(self):
        # R = [0, -1;
        #      1,  0]
        v = self.vertex_array
        x = np.copy(v[:, 0])
        y = np.copy(v[:, 1])
        v[:, 0] = -y
        v[:, 1] = x
        self._vertices = list(map(tuple, v))
        return self

def offread(filepath, points_only=True):
    """ read Geomview OFF file. """
    with open(filepath, 'r') as fin:
        mesh, fixme = _load_off(fin, points_only)
    if fixme:
        _fix_modelnet_broken_off(filepath)
    return mesh

def _load_off(fin, points_only):
    """ read Geomview OFF file. """
    mesh = Mesh()

    fixme = False
    sig = fin.readline().strip()
    if sig == 'OFF':
        line = fin.readline().strip()
        num_verts, num_faces, num_edges = tuple([int(s) for s in line.split(' ')])
    elif sig[0:3] == 'OFF': # ...broken data in ModelNet (missing '\n')...
        line = sig[3:]
        num_verts, num_faces, num_edges = tuple([int(s) for s in line.split(' ')])
        fixme = True
    else:
        raise RuntimeError('unknown format')

    for v in range(num_verts):
        vp = tuple(float(s) for s in fin.readline().strip().split(' '))
        mesh._vertices.append(vp)

    if points_only:
        return mesh, fixme

    for f in range(num_faces):
        fc = tuple([int(s) for s in fin.readline().strip().split(' ')][1:])
        mesh._faces.append(fc)

    return mesh, fixme


def _fix_modelnet_broken_off(filepath):
    oldfile = '{}.orig'.format(filepath)
    os.rename(filepath, oldfile)
    with open(oldfile, 'r') as fin:
        with open(filepath, 'w') as fout:
            sig = fin.readline().strip()
            line = sig[3:]
            print('OFF', file=fout)
            print(line, file=fout)
            for line in fin:
                print(line.strip(), file=fout)


def objread(filepath, points_only=True):
    """Loads a Wavefront OBJ file. """
    _vertices = []
    _normals = []
    _texcoords = []
    _faces = []
    _mtl_name = None

    material = None
    for line in open(filepath, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v = tuple(map(float, values[1:4]))
            _vertices.append(v)
        elif values[0] == 'vn':
            v = tuple(map(float, values[1:4]))
            _normals.append(v)
        elif values[0] == 'vt':
            _texcoords.append(tuple(map(float, values[1:3])))
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'mtllib':
            _mtl_name = values[1]
        elif values[0] == 'f':
            face_ = []
            texcoords_ = []
            norms_ = []
            for v in values[1:]:
                w = v.split('/')
                face_.append(int(w[0]) - 1)
                if len(w) >= 2 and len(w[1]) > 0:
                    texcoords_.append(int(w[1]) - 1)
                else:
                    texcoords_.append(-1)
                if len(w) >= 3 and len(w[2]) > 0:
                    norms_.append(int(w[2]) - 1)
                else:
                    norms_.append(-1)
            #_faces.append((face_, norms_, texcoords_, material))
            _faces.append(face_)

    mesh = Mesh()
    mesh._vertices = _vertices
    if points_only:
        return mesh

    mesh._faces = _faces

    return mesh


class Mesh2Points:
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh = mesh.clone()
        v = mesh.vertex_array
        return torch.from_numpy(v).type(dtype=torch.float)


class OnUnitCube:
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True) # [N, D] -> [1, D]
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0] # [N, D] -> [D]
        s = torch.max(c) # -> scalar
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)

    def __call__(self, tensor):
        #return self.method1(tensor)
        return self.method2(tensor)


class ModelNet(Globset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """
    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = offread
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class ShapeNet2(Globset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """
    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)
        

class Resampler:
    """ [N, D] -> [M, D] """
    def __init__(self, num):
        self.num = num

    def __call__(self, tensor):
        num_points, dim_p = tensor.size()
        out = torch.zeros(self.num, dim_p).to(tensor)

        selected = 0
        while selected < self.num:
            remainder = self.num - selected
            idx = torch.randperm(num_points)
            sel = min(remainder, num_points)
            val = tensor[idx[:sel]]
            out[selected:(selected + sel)] = val
            selected += sel
        return out