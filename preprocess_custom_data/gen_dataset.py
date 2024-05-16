import numpy as np
import trimesh
import cv2 as cv
import sys
import os
from glob import glob


if __name__ == '__main__':
    work_dir = sys.argv[1]
    pose_path = os.path.join(work_dir, 'pose.txt')
    with open(pose_path, 'r') as f:
        poses_raw = []
        for line in f:
            pose = line.strip('\n')
            pose = pose.split(',')
            pose = np.array(pose, dtype=np.float32)
            pose = pose.reshape((4, 4)).transpose()
            poses_raw.append(pose)
    # poses_hwf = poses_hwf[:,:-2].reshape((poses_hwf.shape[0], 3,5)) # n_images, 3, 5
    # intrinsic_data = np.array([[542.6570493669919, 0.0, 677.0332776189574], [0.0, 542.8110949281338, 537.3647990360622],
    #                          [0.0, 0.0, 1.0]])
    intrinsic_data = np.array([[767.3835050134344, 0.0, 679.0564552465218], [0.0, 767.5024011539994, 543.6475158055246],
                               [0.0, 0.0, 1.0]])
    intrinsic = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    intrinsic[:3, :3] = intrinsic_data
    cam_dict = dict()
    n_images = len(poses_raw)

    # Convert space
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose = poses_raw[i]
        pose = pose @ convert_mat
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)


    pcd = trimesh.load(os.path.join(work_dir, 'coverage_mesh.obj'))
    trimesh.PointCloud(pcd.sample(1000000)).export("sparse_points.ply")
    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    out_dir = os.path.join(work_dir, 'preprocessed')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

    image_list = glob(os.path.join(work_dir, 'images/*.png'))
    image_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))

    for i, image_path in enumerate(image_list):
        img = cv.imread(image_path)
        cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)
        cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), np.ones_like(img) * 255)

    np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    print('Process done!')
