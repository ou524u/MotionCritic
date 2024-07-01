from render.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
os.environ['PYOPENGL_PLATFORM'] = "egl"


import torch
# from visualize.simplify_loc2rot import joints2smpl
import pyrender
# import matplotlib.pyplot as plt

import io
import imageio
from shapely import geometry
import trimesh
from pyrender.constants import RenderFlags
# import math
# import ffmpeg
# from PIL import Image
# import utils.rotation_conversions as geometry_u

import cv2
# import argparse



class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P
    


def wrap_text(comment):
    lines = []
    items = comment.split(',')

    output = ""
    for i, item in enumerate(items):
        if i == 0:
            lines.append(items[i])
        else:
            output += (item + ',')
            if i % 2 == 0 or i == len(items)-1:
                lines.append(output)
                output = ""

    return lines

def render_single(motion, device, comment, file_path, no_comment=False, isYellow=True):
    # if os.path.exists(file_path):
    #     print(f"File {file_path} already exists. Skipping rendering.")
    #     return

    comment_lines = wrap_text(comment)
    motion = motion.to(device)
    rot2xyz = Rotation2xyz(device=device)
    faces = rot2xyz.smpl_model.faces

    vertices = rot2xyz(motion, mask=None,
                       pose_rep='rot6d', translation=True, glob=True,
                       jointstype='vertices', betas=None, beta=0, glob_rot=None,
                       vertstrans=True)

    frames = 60
    MINS = torch.min(torch.min(vertices[0], dim=0)[0], dim=1)[0]
    MAXS = torch.max(torch.max(vertices[0], dim=0)[0], dim=1)[0]

    out_list = []

    minx = (MINS[0] - 0.5)
    maxx = (MAXS[0] + 0.5)
    minz = (MINS[2] - 0.5)
    maxz = (MAXS[2] + 0.5)

    vid = []

    # Define cam_pose tensor
    cam_pose = np.array([[ 1, 0, 0, ((minx+maxx)/2).cpu().numpy()],
                [ 0, np.cos(-np.pi / 6), -np.sin(-np.pi / 6), 1.5],
                [ 0, np.sin(-np.pi / 6), np.cos(-np.pi / 6), max(4, (minz+(1.5-MINS[1])*2).cpu().numpy(), (maxx-minx).cpu().numpy())],
                [ 0, 0, 0, 1] ])
    

    # Define polygon_size tensor
    polygon_size = torch.tensor([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])

    # Create polygon tensor
    polygon = geometry.Polygon(polygon_size)

    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

    # Create polygon_pose tensor
    polygon_pose = torch.tensor([[1., 0., 0., 0.],
                                [0., torch.cos(torch.tensor(np.pi / 2, device=device)), -torch.sin(torch.tensor(np.pi / 2, device=device)), MINS[1]],
                                [0., torch.sin(torch.tensor(np.pi / 2, device=device)), torch.cos(torch.tensor(np.pi / 2, device=device)), 0.],
                                [0., 0., 0., 1.]])

    print(f">>> rendering {file_path}")

    valid_frames = 0
    for i in range(frames):
        valid_frames += 1

        mesh = trimesh.Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

        base_color = (0.11, 0.53, 0.8, 0.5)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color
        )
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
        polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)
        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=300)
        scene.add(mesh)
        scene.add(polygon_render, pose=polygon_pose)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose.copy())
        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose.copy())
        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose.copy())
        scene.add(camera, pose=cam_pose)

        # Render scene
        r = pyrender.OffscreenRenderer(960, 960)
        color, _ = r.render(scene, flags=RenderFlags.RGBA)
        r.delete()

        # Convert the color image to OpenCV format (BGR)
        color_cv = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGBA2BGR)

        if no_comment is False:
            # Add text to the image
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 1.0
            font_color = (0, 0, 0)  # BGR color
            font_thickness = 1
            text_position = (10, 720)  # Position where you want to place the text

            # cv2.putText(color_cv, comment, text_position, font, font_scale, font_color, font_thickness)
            for line in comment_lines:
                cv2.putText(color_cv, line, text_position, font, font_scale, font_color, font_thickness)
                text_position = (text_position[0], text_position[1] + 40)
        if not isYellow:
            color_cv = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGBA)
        vid.append(color_cv)

    out = np.stack(vid, axis=0)
    if no_comment:
        file_path = file_path[:-4] + "-nocomment.mp4"

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    imageio.mimsave(file_path, out, fps=24)
    # print("render complete")

    # return cam_pose, polygon_size


def render_multi(motions, device, comments, file_paths, no_comment=False, isYellow=True):
    # assert len(comments) >= motions.shape[0]
    # assert len(file_paths) >= motions.shape[0]
    for i in range(motions.shape[0]):
        render_single(motions[i:i+1], device, comments[i], file_paths[i], no_comment=no_comment, isYellow=isYellow)