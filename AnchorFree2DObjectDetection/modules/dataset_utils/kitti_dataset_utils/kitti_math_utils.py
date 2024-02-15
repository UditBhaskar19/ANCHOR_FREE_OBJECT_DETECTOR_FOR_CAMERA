import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
def latlonToMercator(lat, lon, scale):
    # converts lat/lon coordinates to mercator coordinates using mercator scale
    er = 6378137
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log( np.tan((90 + lat) * np.pi / 360) )
    return mx, my

# ---------------------------------------------------------------------------------------------------------------------
def latToScale(lat):
    # compute mercator scale from latitude
    scale = np.cos(lat * np.pi / 180.0)
    return scale

# ---------------------------------------------------------------------------------------------------------------------
def create_pose_matrix(tx, ty, tz, rx, ry, rz):
    Rx = np.array([[1, 0, 0], 
                   [0, np.cos(rx), -np.sin(rx)], 
                   [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], 
                   [0, 1, 0], 
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], 
                   [np.sin(rz), np.cos(rz), 0], 
                   [0, 0, 1]], dtype=np.float32)
    R = Rz * Ry * Rx
    T = np.array([tx, ty, tz], dtype=np.float32)
    SE3 = np.eye(4, dtype=np.float32)
    SE3[:3, :3] = R
    SE3[:3, -1] = T
    return SE3, R, T
