import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob

import trimesh
from tqdm import tqdm
# from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        # self.preprocess_images_le2()
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')


    def preprocess_images_le2(self):
        """
        Undistort images according to lightneus, second pass
        """
        print("Undistorting images")
        num_imgs, rows, cols, channels = self.images.shape

        #TODO: put below magic nums from calibration in conf
        g= 2.0 # Autogain unknown for ct1a, estimates from 1 to 3
        gamma = 2.2 # gamma correction, generally constant
        k = 2.5 # decay power from emitted light
        # f = 767.45 # average of fx and fy, TODO compute differently in different directions
        f = 542.7
        h = 1080
        w = 1350
        # (252,0) pixel coord varies between 160<alpha<170
        p0 = np.array([252,0]) # point at corner of fov
        pcenter = np.array([(h-1)/2, (w-1)/2])
        centered_p0 = p0 - pcenter
        fov = 165/2 # varies between 160 and 170 for this c3vd endoscope
        alpha = fov/2
        d = f * np.tan(alpha)
        px_size = np.array([d*np.sin(alpha), d*np.cos(alpha)]) / centered_p0 # get pixel sizes, row/col correspond to y/x

        # Compute alpha and then Le for every point based on pixel size
        rows, cols = np.meshgrid(np.arange(w), np.arange(h))
        row_dist = np.abs(rows - pcenter[0])
        col_dist = np.abs(cols - pcenter[1])
        dists = np.stack((row_dist, col_dist), axis=-1) * np.expand_dims(px_size, axis=(0,1))
        dists = np.linalg.norm(dists, axis=2)
        alpha = np.arctan2(dists, f)
        Le = np.power(np.cos(alpha), k)
        cv.imwrite("./Le.png", Le * 255)
        cv.imwrite("./le_img.png", (self.images[0] / torch.Tensor(Le).unsqueeze(-1).detach().cpu()).numpy())

        # Compute normalized images
        Le = torch.Tensor(Le).unsqueeze(-1).detach().cpu()
        self.images = torch.pow((torch.pow(self.images, gamma) / (Le * g)), 1 / gamma)


    def preprocess_images_le(self):
        """
        Undistort images according to lightneus and factor in emitted light
        """
        num_imgs, rows, cols, channels = self.images.shape
        row_idxs, col_idxs = np.indices((rows, cols))
        pixel_coords = np.stack((row_idxs, col_idxs), axis=-1) # [H, W, 2] where each entry in first 2d is [row, col]

        print("Undistorting images")
        for i in tqdm(range(num_imgs)):
            img = self.images[i].numpy()
            depth_map = np.sqrt(np.sum(img, axis=2)) # Approximate depth as the pixel intensity
            row_col_depth = np.dstack((pixel_coords, depth_map)).reshape(-1, 3)
            g_t = 2.0 # Autogain unknown for ct1a, estimates from 1 to 3
            gamma = 2.2 # generally a constant
            k = 2.5 # estimate from lightneus precursor for cosine power for decay
            Le = np.apply_along_axis( # Compute for every pixel
                self.get_calibrated_photometric_endoscope_model, 
                1, row_col_depth, 
                k, g_t, gamma)
            Le = Le.reshape((img.shape[0], img.shape[1]))
            Le = np.repeat(Le[:, :, np.newaxis], 3, axis=2) # Repeat along every channel
            img = np.power(np.power(img, gamma) / (g_t * Le), 1/gamma)
            self.images[i] = torch.Tensor(img).to('cuda')

    def light_spread_function(self, z, k):
        return np.power(np.abs(np.cos(z)), k)

    def get_calibrated_photometric_endoscope_model(self, row_col_depth, k, g_t, gamma):
        r, c, z = row_col_depth
        mu_prime = self.light_spread_function(z, k)
        f_r_theta = 1/np.pi # Lambertian BRDF, might work better with value closer to 0 like 1/2pi
        xc_to_pixel = np.linalg.norm(np.array([r, c, z])) # Find distance from center of image to pixel
        theta = 2 * (np.arccos(np.linalg.norm(np.array([r, c])) / xc_to_pixel)) # Compute angle of incidence, and then find angle theta
        xc_to_pixel += 1e-7 # to avoid divide by zero
        L = (mu_prime / xc_to_pixel) * f_r_theta * np.cos(theta) * g_t
        L  = np.power(np.abs(L), gamma) # TODO: might need to preserve sign
        return L

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3].cpu(), p[:, :, None].cpu()).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3].cpu(), rays_v[:, :, None].cpu()).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

