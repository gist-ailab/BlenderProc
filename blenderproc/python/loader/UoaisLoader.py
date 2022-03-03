import os
import sys
from random import choice
from typing import List, Union

import bpy
import random
import numpy as np
from mathutils import Matrix, Vector

from blenderproc.python.utility.SetupUtility import SetupUtility
import blenderproc.python.camera.CameraUtility as CameraUtility
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.utility.MathUtility import change_source_coordinate_frame_of_transformation_matrix
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.loader.BopLoader import load_bop_objs



def get_uoais_obj_ids_per_ds(dataset_parent_path: str, dataset_names: str, train=True):
    mode = "train" if train else "val"
    obj_ids_per_ds = {}
    for dataset_name in dataset_names:
        idx_path = os.path.join(dataset_parent_path, dataset_name, mode + "_obj.txt")
        obj_ids_per_ds[dataset_name] = []
        with open(idx_path) as f:
            for file_name in f.readlines():
                object_idx = int(file_name.split('.')[0].split('_')[1])
                obj_ids_per_ds[dataset_name].append(object_idx)
        obj_ids_per_ds[dataset_name] = sorted(obj_ids_per_ds[dataset_name])
    return obj_ids_per_ds


def load_uoais_objs(dataset_path: str, obj_ids_per_ds: dict,
                     num_of_objs_to_sample_per_ds: dict, obj_instances_limit: int,) -> List[MeshObject]:
   
    sampled_objs = []
    for dataset_name in obj_ids_per_ds.keys():
        model_type = "cad" if dataset_name == "tless" else ""
        sampled_objs += load_bop_objs(bop_dataset_path = os.path.join(dataset_path, dataset_name),
                                            mm2m = True,
                                            model_type = model_type,
                                            sample_objects = True,
                                            obj_ids = obj_ids_per_ds[dataset_name],
                                            obj_instances_limit = obj_instances_limit,
                                            num_of_objs_to_sample = num_of_objs_to_sample_per_ds[dataset_name])
    return sampled_objs

def set_random_intrinsics(cfg):

    min_f = cfg["simulation"]["camera"]["focal_length"]["min"]
    max_f = cfg["simulation"]["camera"]["focal_length"]["max"]
    mean_cx = float(cfg["simulation"]["camera"]["resolution"]["x"]-1) / 2 
    mean_cy = float(cfg["simulation"]["camera"]["resolution"]["y"]-1) / 2 
    min_delta_c = cfg["simulation"]["camera"]["delta_optical_center"]["min"]
    max_delta_c = cfg["simulation"]["camera"]["delta_optical_center"]["max"]
    min_cx = mean_cx + min_delta_c
    max_cx = mean_cx + max_delta_c
    min_cy = mean_cy + min_delta_c
    max_cy = mean_cy + max_delta_c

    focal = random.uniform(min_f, max_f)
    cx = random.uniform(min_cx, max_cx)
    cy = random.uniform(min_cy, max_cy)
    K = [[focal, 0, cx],
        [0, focal, cy], 
        [0, 0, 1]]

    CameraUtility.set_intrinsics_from_K_matrix(K, cfg["simulation"]["camera"]["resolution"]["x"], cfg["simulation"]["camera"]["resolution"]["y"])
