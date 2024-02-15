# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : kitti object tracking dataset utilities
# ---------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import json
import config_dataset
from modules.dataset_utils.kitti_dataset_utils.kitti_math_utils import (
    latlonToMercator, latToScale, create_pose_matrix)

img_ext = '.png'
txt_ext = '.txt'

# ---------------------------------------------------------------------------------------------------------------------
def get_frame_path_list(sequence, image_dir, dataset_rootdir):
    image_seq_dir = os.path.join(dataset_rootdir, image_dir, sequence)
    num_frames = len([fname for fname in os.listdir(image_seq_dir) if fname.endswith(img_ext)])

    frame_path_list = []
    for frameid in range(num_frames):
        file_name = "{:06d}".format(frameid) + img_ext
        frame_path_list.append(os.path.join(dataset_rootdir, image_dir, sequence, file_name))
    return frame_path_list

# ---------------------------------------------------------------------------------------------------------------------
def parse_calib_file(sequence, calibration_dir, dataset_rootdir):
    calib_dict = {}
    calib_file = os.path.join(dataset_rootdir, calibration_dir, sequence + txt_ext)
    with open(calib_file, 'r') as file:
        for line in file:
            entries = line.strip().split()
            calib_type = entries[0].replace(':', '')
            calib_data = entries[1:]
            if calib_type == 'R_rect': calib_data = [ calib_data[:3], calib_data[3:6], calib_data[6:] ]
            else: calib_data = [ calib_data[:3], calib_data[3:6], calib_data[6:9], calib_data[9:] ]
            calib_dict[calib_type] = calib_data
    return calib_dict

# ---------------------------------------------------------------------------------------------------------------------
def parse_oxts_line(line):
    oxts = {}
    entries = line.strip().split()
    oxts['lat'] = float(entries[0])
    oxts['lon'] = float(entries[1])
    oxts['alt'] = float(entries[2])

    oxts['roll'] = float(entries[3])
    oxts['pitch'] = float(entries[4])
    oxts['yaw'] = float(entries[5])

    oxts['vn'] = float(entries[6])
    oxts['ve'] = float(entries[7])
    oxts['vf'] = float(entries[8])
    oxts['vl'] = float(entries[9])
    oxts['vu'] = float(entries[10])

    oxts['ax'] = float(entries[11])
    oxts['ay'] = float(entries[12])
    oxts['az'] = float(entries[13])
    oxts['af'] = float(entries[14])
    oxts['al'] = float(entries[15])
    oxts['au'] = float(entries[16])

    oxts['wx'] = float(entries[17])
    oxts['wy'] = float(entries[18])
    oxts['wz'] = float(entries[19])
    oxts['wf'] = float(entries[20])
    oxts['wl'] = float(entries[21])
    oxts['wu'] = float(entries[22])

    oxts['posacc'] = float(entries[23])
    oxts['velacc'] = float(entries[24])

    oxts['navstat'] = float(entries[25])
    oxts['numsats'] = float(entries[26])
    oxts['posmode'] = float(entries[27])
    oxts['velmode'] = float(entries[28])
    oxts['orimode'] = float(entries[29])
    return oxts

# ---------------------------------------------------------------------------------------------------------------------
def parse_oxts_file(sequence, oxts_dir, dataset_rootdir):
    oxts_list = []
    oxts_file = os.path.join(dataset_rootdir, oxts_dir, sequence + txt_ext)
    with open(oxts_file, 'r') as file:
        for line in file:
            oxts = parse_oxts_line(line)
            scale = latToScale(oxts['lat'])
            oxts['tx'], oxts['ty'] = latlonToMercator(oxts['lat'], oxts['lon'], scale)
            oxts['tz'] = oxts['alt']
            oxts_list.append(oxts)
    return oxts_list

# ---------------------------------------------------------------------------------------------------------------------
def create_poses(oxts):
    tx0, ty0, tz0 = oxts[0]['tx'], oxts[0]['ty'], oxts[0]['tz']
    rx0, ry0, rz0 = oxts[0]['roll'], oxts[0]['pitch'], oxts[0]['yaw']
    SE3_t0, _, _ = create_pose_matrix(tx0, ty0, tz0, rx0, ry0, rz0)
    SE3_inv_t0 = np.linalg.inv(SE3_t0)

    poses = []
    for oxt in oxts:
        tx, ty, tz = oxt['tx'], oxt['ty'], oxt['tz']
        rx, ry, rz = oxt['roll'], oxt['pitch'], oxt['yaw']
        SE3, _, _ = create_pose_matrix(tx, ty, tz, rx, ry, rz)
        pose_wrt_ego_t0 = SE3_inv_t0 * SE3
        poses.append(pose_wrt_ego_t0)
    return poses