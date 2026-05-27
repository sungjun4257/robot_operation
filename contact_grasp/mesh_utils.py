# -*- coding: utf-8 -*-
"""Helper classes and functions to sample grasps for a given object mesh."""

from __future__ import print_function

import argparse
from collections import OrderedDict
import errno
import json
import os
import numpy as np
import pickle
from tqdm import tqdm
import trimesh
import trimesh.transformations as tra

import torch

# import tensorflow.compat.v1 as tf

class Object(object):
    """Represents a graspable object."""

    def __init__(self, filename):
        """Constructor.

        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        try:
            self.mesh = trimesh.load(filename)
        except:
            print('Could not load mesh from filename: {}'.format(filename))
            raise ValueError('Could not load mesh from filename: {}'.format(filename))
        
        self.scale = 1.0

        # print(filename)
        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('object', self.mesh)

    def rescale(self, scale=1.0):
        """Set scale of object mesh.

        :param scale
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def resize(self, size=1.0):
        """Set longest of all three lengths in Cartesian space.

        :param size
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)

    def in_collision_with(self, mesh, transform):
        """Check whether the object is in collision with the provided mesh.

        :param mesh:
        :param transform:
        :return: boolean value
        """
        return self.collision_manager.in_collision_single(mesh, transform=transform)


class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        """Create a Franka Panda parallel-yaw gripper object.

        Keyword Arguments:
            q {list of int} -- opening configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
        """
        self.joint_limits = [0.0, 0.04]
        self.root_folder = root_folder
        
        self.default_pregrasp_configuration = 0.04
        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        fn_base = os.path.join(root_folder, 'contact_grasp/gripper_models/panda_gripper/hand.stl')
        fn_finger = os.path.join(root_folder, 'contact_grasp/gripper_models/panda_gripper/finger.stl')

        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])
        
        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])


        self.contact_ray_origins = []
        self.contact_ray_directions = []

        # coords_path = os.path.join(root_folder, 'gripper_control_points/panda_gripper_coords.npy')
        with open(os.path.join(root_folder,'contact_grasp/gripper_control_points/panda_gripper_coords.pickle'), 'rb') as f:
            self.finger_coords = pickle.load(f, encoding='latin1')
        finger_direction = self.finger_coords['gripper_right_center_flat'] - self.finger_coords['gripper_left_center_flat']
        self.contact_ray_origins.append(np.r_[self.finger_coords['gripper_left_center_flat'], 1])
        self.contact_ray_origins.append(np.r_[self.finger_coords['gripper_right_center_flat'], 1])
        self.contact_ray_directions.append(finger_direction / np.linalg.norm(finger_direction))
        self.contact_ray_directions.append(-finger_direction / np.linalg.norm(finger_direction))

        self.contact_ray_origins = np.array(self.contact_ray_origins)
        self.contact_ray_directions = np.array(self.contact_ray_directions)

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.

        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_l, self.finger_r, self.base]
        
    def get_closing_rays_contact(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.

        Arguments:
            transform {[nump.array]} -- a 4x4 homogeneous matrix
            contact_ray_origin {[nump.array]} -- a 4x1 homogeneous vector
            contact_ray_direction {[nump.array]} -- a 4x1 homogeneous vector

        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return transform[:3, :].dot(
            self.contact_ray_origins.T).T, transform[:3, :3].dot(self.contact_ray_directions.T).T
        
    def get_control_point_tensor(self, batch_size, use_torch=True, symmetric = False, convex_hull=True):
        """
        Outputs a 5 point gripper representation of shape (batch_size x 5 x 3).

        Arguments:
            batch_size {int} -- batch size

        Keyword Arguments:
            use_tf {bool} -- outputing a tf tensor instead of a numpy array (default: {True})
            symmetric {bool} -- Output the symmetric control point configuration of the gripper (default: {False})
            convex_hull {bool} -- Return control points according to the convex hull panda gripper model (default: {True})

        Returns:
            np.ndarray -- control points of the panda gripper 
        """

        control_points = np.load(os.path.join(self.root_folder, 'gripper_control_points/panda.npy'))[:, :3]
        if symmetric:
            control_points = [[0, 0, 0], control_points[1, :],control_points[0, :], control_points[-1, :], control_points[-2, :]]
        else:
            control_points = [[0, 0, 0], control_points[0, :], control_points[1, :], control_points[-2, :], control_points[-1, :]]

        control_points = np.asarray(control_points, dtype=np.float32)
        if not convex_hull:
            # actual depth of the gripper different from convex collision model
            control_points[1:3, 2] = 0.0584
        control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])

        if use_torch:
            return torch.from_numpy(control_points)
            return tf.convert_to_tensor(control_points)

        return control_points

    def to_open3d(self):
        import open3d as o3d
        """
        Gripper의 전체 메쉬 (base + fingers)를 open3d 형식으로 변환.
        Returns:
            open3d.geometry.TriangleMesh
        """
        meshes = []

        for mesh in self.get_meshes():
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()
            meshes.append(o3d_mesh)

        # 전체 메쉬 합치기
        gripper_o3d = meshes[0]
        for m in meshes[1:]:
            gripper_o3d += m

        gripper_o3d.paint_uniform_color([0.1, 0.1, 0.9])  # 파란색 계열
        return gripper_o3d

def create_gripper(name, configuration=None, root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    """Create a gripper object.

    Arguments:
        name {str} -- name of the gripper

    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})

    Raises:
        Exception: If the gripper name is unknown.

    Returns:
        [type] -- gripper object
    """
    if name.lower() == 'panda':
        return PandaGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))


def in_collision_with_gripper(object_mesh, gripper_transforms, gripper_name, silent=False):
    """Check collision of object with gripper.

    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper
        gripper_name {str} -- name of gripper

    Keyword Arguments:
        silent {bool} -- verbosity (default: {False})

    Returns:
        [list of bool] -- Which gripper poses are in collision with object mesh
    """
    manager = trimesh.collision.CollisionManager()
    manager.add_object('object', object_mesh)
    gripper_meshes = [create_gripper(gripper_name).hand]
    min_distance = []
    for tf in tqdm(gripper_transforms, disable=silent):
        min_distance.append(np.min([manager.min_distance_single(
            gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes]))

    return [d == 0 for d in min_distance], min_distance

def grasp_contact_location(transforms, successfuls, collisions, object_mesh, gripper_name='panda', silent=False):
    """Computes grasp contacts on objects and normals, offsets, directions

    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})

    Returns:
        list of dicts of contact information per grasp ray
    """
    res = []
    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)


    for p, colliding, outcome in tqdm(zip(transforms, collisions, successfuls), total=len(transforms), disable=silent):
        contact_dict = {}
        contact_dict['collisions'] = 0
        contact_dict['valid_locations'] = 0
        contact_dict['successful'] = outcome
        contact_dict['grasp_transform'] = p
        contact_dict['contact_points'] = []
        contact_dict['contact_directions'] = []
        contact_dict['contact_face_normals'] = []
        contact_dict['contact_offsets'] = []

        if colliding:
            contact_dict['collisions'] = 1
            
        else:
            ray_origins, ray_directions = gripper.get_closing_rays_contact(p)

            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False)

            if len(locations) > 0:
                # this depends on the width of the gripper
                valid_locations = np.linalg.norm(ray_origins[index_ray]-locations, axis=1) <= 2.0*gripper.q

                if sum(valid_locations) > 1:
                    contact_dict['valid_locations'] = 1
                    contact_dict['contact_points'] = locations[valid_locations]
                    contact_dict['contact_face_normals'] = object_mesh.face_normals[index_tri[valid_locations]]
                    contact_dict['contact_directions'] = ray_directions[index_ray[valid_locations]]
                    contact_dict['contact_offsets'] = np.linalg.norm(ray_origins[index_ray[valid_locations]] - locations[valid_locations], axis=1)
                    # dot_prods = (contact_dict['contact_face_normals'] * contact_dict['contact_directions']).sum(axis=1)
                    # contact_dict['contact_cosine_angles'] = np.cos(dot_prods)
                    res.append(contact_dict)
                
    return res

def grasp_contact_location_jun_batch(transforms, successfuls, collisions, object_mesh, gripper_name='panda', silent=False):
    """Computes grasp contacts on objects and normals, offsets, directions

    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})

    Returns:
        list of dicts of contact information per grasp ray
    """
    res = []
    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
        
    
    for p, colliding, outcome in tqdm(zip(transforms, collisions, successfuls), total=len(transforms), disable=silent):
        contact_dict = {}
        contact_dict['collisions'] = 0
        contact_dict['valid_locations'] = 0
        contact_dict['successful'] = outcome
        contact_dict['grasp_transform'] = p
        contact_dict['contact_points'] = []
        contact_dict['contact_directions'] = []
        contact_dict['contact_face_normals'] = []
        contact_dict['contact_offsets'] = []

        if colliding:
            contact_dict['collisions'] = 1
            res.append(None)
        else:
            ray_origins, ray_directions = gripper.get_closing_rays_contact(p)

            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=True)

            if len(locations) > 0:
                # this depends on the width of the gripper
                valid_locations = np.linalg.norm(ray_origins[index_ray]-locations, axis=1) <= 2.0*gripper.q

                if sum(valid_locations) > 1:
                    contact_dict['valid_locations'] = 1
                    contact_dict['contact_points'] = locations[valid_locations]
                    contact_dict['contact_face_normals'] = object_mesh.face_normals[index_tri[valid_locations]]
                    contact_dict['contact_directions'] = ray_directions[index_ray[valid_locations]]
                    contact_dict['contact_offsets'] = np.linalg.norm(ray_origins[index_ray[valid_locations]] - locations[valid_locations], axis=1)
                    # dot_prods = (contact_dict['contact_face_normals'] * contact_dict['contact_directions']).sum(axis=1)
                    # contact_dict['contact_cosine_angles'] = np.cos(dot_prods)
                    res.append(contact_dict)

                    import open3d as o3d
                    from PIL import Image
                
                    gripper_o3d = gripper.to_open3d()
                    # bbox = gripper_o3d.get_axis_aligned_bounding_box()
                    # print("Gripper bounding box extent:", bbox.get_extent())
                    # Optional: gripper pose transform
                    gripper_o3d.transform(p)
                
                    # Object mesh: trimesh → open3d 변환
                    object_o3d = o3d.geometry.TriangleMesh()
                    object_o3d.vertices = o3d.utility.Vector3dVector(object_mesh.vertices)
                    object_o3d.triangles = o3d.utility.Vector3iVector(object_mesh.faces)
                    object_o3d.compute_vertex_normals()
                    object_o3d.paint_uniform_color([0.8, 0.8, 0.8])  # 연한 회색
                
                    data_dir = "/home/vision/packages/FoundationPose/result"
                    color_rs = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
                    depths_rs = np.array(Image.open(os.path.join(data_dir, 'depth.png'))) / 1000.0
                    fx, fy = 612.6182250976562, 612.7216796875
                    cx, cy = 318.95758056640625, 240.05343627929688
                
                    # scale = 1000.0
                
                    # get point cloud
                    xmap, ymap = np.arange(depths_rs.shape[1]), np.arange(depths_rs.shape[0])
                    xmap, ymap = np.meshgrid(xmap, ymap)
                    points_z = depths_rs 
                    points_x = (xmap - cx) / fx * points_z
                    points_y = (ymap - cy) / fy * points_z
                    mask = (points_z > 0) & (points_z < 1)
                    points_rs = np.stack([points_x, points_y, points_z], axis=-1)
                    points_rs = points_rs[mask].astype(np.float32)
                    color_rs = color_rs[mask].astype(np.float32)
                
                    cloud_scene = o3d.geometry.PointCloud()
                    cloud_scene.points = o3d.utility.Vector3dVector(points_rs)
                    cloud_scene.colors = o3d.utility.Vector3dVector(color_rs)
                
                    # translation 위치에 작은 빨간 구 시각화
                    translation = np.array(p)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                    sphere.paint_uniform_color([1, 0, 0])
                    sphere.transform(p)
                    # print(contact_dict)
                
                    
                    
                    if contact_dict['valid_locations'] == 1:
                        contact_cloud = o3d.geometry.PointCloud()
                        contact_cloud.points = o3d.utility.Vector3dVector(contact_dict['contact_points'])
                        contact_cloud.paint_uniform_color([1, 0, 0])  # 빨간 점
                
                        normal_line1 = o3d.geometry.LineSet()
                        normal_start1 = np.array(contact_dict['contact_points'][0])  # Normal 벡터의 시작점
                        normal_end1 = normal_start1 + np.array(contact_dict['contact_face_normals'][0]) * 0.01  # Normal 벡터의 끝점 (길이 0.05로 설정)
                        normal_line1.points = o3d.utility.Vector3dVector([normal_start1, normal_end1])
                        normal_line1.lines = o3d.utility.Vector2iVector([[0, 1]])  # 벡터를 잇는 선
                        normal_line1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 빨간색
                
                    else:
                        contact_cloud = None
                    
                    
                    
                    # # 시각화
                    # if contact_cloud is not None:
                    #     o3d.visualization.draw_geometries([sphere, object_o3d, cloud_scene, contact_cloud, gripper_o3d, normal_line1])
                    # else:
                    #     o3d.visualization.draw_geometries([sphere, object_o3d, cloud_scene, gripper_o3d])
                else:
                    res.append(None)

    
                
    

    return res


def grasp_contact_location_jun(transforms, successfuls, collisions, object_mesh, gripper_name='panda', silent=False):
    """Computes grasp contacts on objects and normals, offsets, directions

    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})

    Returns:
        list of dicts of contact information per grasp ray
    """
    res = []
    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
        
    

    

    p = transforms
    colliding = collisions
    outcome = successfuls

    contact_dict = {}
    contact_dict['collisions'] = 0
    contact_dict['valid_locations'] = 0
    contact_dict['successful'] = outcome
    contact_dict['grasp_transform'] = p
    contact_dict['contact_points'] = []
    contact_dict['contact_directions'] = []
    contact_dict['contact_face_normals'] = []
    contact_dict['contact_offsets'] = []

    if colliding:
        contact_dict['collisions'] = 1
    else:
        ray_origins, ray_directions = gripper.get_closing_rays_contact(p)
        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins, ray_directions, multiple_hits=False)
        if len(locations) > 0:
            # this depends on the width of the gripper
            valid_locations = np.linalg.norm(ray_origins[index_ray]-locations, axis=1) <= 2.0*gripper.q
            if sum(valid_locations) > 1:
                contact_dict['valid_locations'] = 1
                contact_dict['contact_points'] = locations[valid_locations]
                contact_dict['contact_face_normals'] = object_mesh.face_normals[index_tri[valid_locations]]
                contact_dict['contact_directions'] = ray_directions[index_ray[valid_locations]]
                contact_dict['contact_offsets'] = np.linalg.norm(ray_origins[index_ray[valid_locations]] - locations[valid_locations], axis=1)
                # dot_prods = (contact_dict['contact_face_normals'] * contact_dict['contact_directions']).sum(axis=1)
                # contact_dict['contact_cosine_angles'] = np.cos(dot_prods)
                res.append(contact_dict)
            else:
                res.append(None)
                
    import open3d as o3d
    from PIL import Image

    gripper_o3d = gripper.to_open3d()
    bbox = gripper_o3d.get_axis_aligned_bounding_box()
    print("Gripper bounding box extent:", bbox.get_extent())
    # Optional: gripper pose transform
    gripper_o3d.transform(transforms)

    # Object mesh: trimesh → open3d 변환
    object_o3d = o3d.geometry.TriangleMesh()
    object_o3d.vertices = o3d.utility.Vector3dVector(object_mesh.vertices)
    object_o3d.triangles = o3d.utility.Vector3iVector(object_mesh.faces)
    object_o3d.compute_vertex_normals()
    object_o3d.paint_uniform_color([0.8, 0.8, 0.8])  # 연한 회색

    data_dir = "/home/vision/packages/FoundationPose/result"
    color_rs = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depths_rs = np.array(Image.open(os.path.join(data_dir, 'depth.png'))) / 1000.0
    fx, fy = 612.6182250976562, 612.7216796875
    cx, cy = 318.95758056640625, 240.05343627929688

    # scale = 1000.0

    # get point cloud
    xmap, ymap = np.arange(depths_rs.shape[1]), np.arange(depths_rs.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths_rs 
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    mask = (points_z > 0) & (points_z < 1)
    points_rs = np.stack([points_x, points_y, points_z], axis=-1)
    points_rs = points_rs[mask].astype(np.float32)
    color_rs = color_rs[mask].astype(np.float32)

    cloud_scene = o3d.geometry.PointCloud()
    cloud_scene.points = o3d.utility.Vector3dVector(points_rs)
    cloud_scene.colors = o3d.utility.Vector3dVector(color_rs)

    # translation 위치에 작은 빨간 구 시각화
    translation = np.array(transforms)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sphere.paint_uniform_color([1, 0, 0])
    sphere.transform(transforms)
    

    
    
    if contact_dict['valid_locations'] == 1:
        contact_cloud = o3d.geometry.PointCloud()
        contact_cloud.points = o3d.utility.Vector3dVector(contact_dict['contact_points'])
        contact_cloud.paint_uniform_color([1, 0, 0])  # 빨간 점

        normal_line1 = o3d.geometry.LineSet()
        normal_start1 = np.array(contact_dict['contact_points'][0])  # Normal 벡터의 시작점
        normal_end1 = normal_start1 + np.array(contact_dict['contact_face_normals'][0]) * 0.01  # Normal 벡터의 끝점 (길이 0.05로 설정)
        normal_line1.points = o3d.utility.Vector3dVector([normal_start1, normal_end1])
        normal_line1.lines = o3d.utility.Vector2iVector([[0, 1]])  # 벡터를 잇는 선
        normal_line1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 빨간색

    else:
        contact_cloud = None
    
    
    
    # 시각화
    if contact_cloud is not None:
        o3d.visualization.draw_geometries([sphere, object_o3d, cloud_scene, contact_cloud, gripper_o3d, normal_line1])
    else:
        o3d.visualization.draw_geometries([sphere, object_o3d, cloud_scene, gripper_o3d])

    return res
if __name__ == "__main__":
    file_path  = "/home/vision/packages/FoundationPose/result/config.json"
    with open(file_path, "rb") as f:
        config_data = json.load(f)

    mesh = trimesh.load(config_data["mesh_path"])
    pose = np.load("/home/vision/packages/FoundationPose/result/output_pose.npy").astype(np.float32)

    # JSON 파일 경로
    file_path = '/home/vision/packages/Scale-Balanced-Grasp/best_grasp.json'

    # JSON 파일 읽기
    with open(file_path, "r") as json_file:
        grasp_data = json.load(json_file)

    pose_matrix = np.eye(4)  # 4x4 단위 행렬 생성

    translation = np.array(grasp_data["translation"])
    rotation_matrix = np.array(grasp_data["rotation_matrix"])

    # 4. 🧠 중심에서 반 depth만큼 approach 방향으로 이동
    depth = grasp_data["depth"]
    # if depth <= 0.041:
        # half_depth_vector = np.array([0.024 + 0.077 - 0.01, 0, 0])  # gripper 기준 x축으로 2.4cm 뒤
    
    print(depth)
    half_depth_vector = np.array([depth - 0.041 - 0.077 + 0.009, 0, 0])  # gripper 기준 x축으로 2.4cm 뒤
    translation = translation + rotation_matrix @ half_depth_vector

    pose_matrix[:3, :3] = rotation_matrix  # 상단 좌측 3x3에 회전 행렬 넣기
    pose_matrix[:3, 3] = translation       # 상단 우측 열에 translation 벡터 넣기

    

    from scipy.spatial.transform import Rotation as R

    # 보정 회전: x축 기준 -90도 회전
    correction = np.eye(4)
    correction[:3, :3] = R.from_euler('xyz', [90, 0, 90], degrees=True).as_matrix()


    


    # 보정 적용
    pose_matrix = pose_matrix @ correction
    
    mesh = mesh.apply_transform(pose)
    res = grasp_contact_location_jun(transforms = pose_matrix ,successfuls = None,collisions= None, object_mesh = mesh)
    print(res)