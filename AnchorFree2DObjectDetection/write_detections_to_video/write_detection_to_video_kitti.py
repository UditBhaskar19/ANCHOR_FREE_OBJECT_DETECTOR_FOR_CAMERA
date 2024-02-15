import os, sys, cv2, torch
from torchvision import transforms

module_rootdir = '../'
dataset_rootdir = '../../'
label_rootdir = module_rootdir
weight_rootdir = module_rootdir
sys.path.append(module_rootdir)

import config_dataset
from modules.first_stage.inference import inference
from modules.first_stage.set_parameters_for_inference import set_param_for_video_inference
from modules.dataset_utils.kitti_dataset_utils.kitti_file_utils_test import get_frame_path_list
from modules.dataset_utils.kitti_dataset_utils.kitti_remap_utils import load_specific_sequence_groundtruths_json 

DETECT_PEDESTRIAN = True
DETECT_VEHICLE = True

NMS_THRESH = 0.35
SCORE_THRESH = [0.7, 0.6]

OUT_VIDEO_DIR = 'video_inference/kitti'
OUT_VIDEO_DIR = os.path.join(module_rootdir, OUT_VIDEO_DIR)
OUT_VIDEO_EXT = '.mov'
FPS = 15

# -------------------------------------------------------------------------------------------------------------------------
def set_video_output_dir(out_video_dir: str):
    if not os.path.exists(out_video_dir): os.makedirs(out_video_dir, exist_ok=True)

# -------------------------------------------------------------------------------------------------------------------------
def press_q_to_quit(key):
    return key == 113

preprocess = transforms.Compose([
    transforms.ToTensor(),                 # scale in [0, 1]
    transforms.Normalize(                  # normalize with mean = [0.485, 0.456, 0.406] & std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406], 
        std =  [0.229, 0.224, 0.225])
    ])

def preprocess_video_frame_for_detector(frame, img_w, img_h, device):
    frame = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = preprocess(frame)
    frame = frame.unsqueeze(0).to(device)
    return frame

def draw_detections(frame, objclass, bboxs, resize_factor_w, resize_factor_h):
    bboxs[:, [0,2]] *= resize_factor_w
    bboxs[:, [1,3]] *= resize_factor_h

    veh_box = bboxs[objclass == 0]
    ped_box = bboxs[objclass == 1]

    for idx in range(veh_box.shape[0]):
        box = veh_box[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(frame, tl, br, (0,255,255), thickness=2)

    for idx in range(ped_box.shape[0]):
        box = ped_box[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(frame, tl, br, (0,0,255), thickness=2)
    return frame

def select_vehicle_detections(pred_score, pred_class, pred_box):
    flag = ( pred_class == 0 )
    pred_score = pred_score[flag]
    pred_class = pred_class[flag]
    pred_box = pred_box[flag]
    return pred_score, pred_class, pred_box

def select_ped_detections(pred_score, pred_class, pred_box):
    flag = ( pred_class == 1 )
    pred_score = pred_score[flag]
    pred_class = pred_class[flag]
    pred_box = pred_box[flag]
    return pred_score, pred_class, pred_box

# -------------------------------------------------------------------------------------------------------------------------
def write_detections_to_video(
    frames_path, fps,
    device, deltas_mean, deltas_std,
    resize_factor_w, resize_factor_h,
    out_video_path, dataset_param,
    nms_thresh, score_threshold,
    grid_coord, detector):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_write = cv2.VideoWriter(out_video_path, fourcc, fps, (dataset_param.IMG_W, dataset_param.IMG_H))

    for frame_path in frames_path:
        vframe = cv2.imread(frame_path)

        frame = preprocess_video_frame_for_detector(
            vframe, dataset_param.IMG_RESIZED_W, dataset_param.IMG_RESIZED_H, device)
        
        pred = inference(
            detector, frame, grid_coord,
            deltas_mean, deltas_std,
            score_threshold, nms_thresh)
        
        pred_score = pred['pred_score'].cpu().numpy()
        pred_class = pred['pred_class'].cpu().numpy()
        pred_box = pred['pred_box'].cpu().numpy()

        if DETECT_PEDESTRIAN & ~DETECT_VEHICLE:   # for person only
            pred_score, pred_class, pred_box \
                = select_ped_detections(pred_score, pred_class, pred_box)
            
        elif ~DETECT_PEDESTRIAN & DETECT_VEHICLE:   # for vehicle only
            pred_score, pred_class, pred_box \
                = select_vehicle_detections(pred_score, pred_class, pred_box)

        vframe = draw_detections(vframe, pred_class, pred_box, resize_factor_w, resize_factor_h)
        video_write.write(vframe)
        cv2.imshow("Source", vframe)
        cv2.waitKey(1)

    video_write.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    weights_file = 'model_weights/1705990924432/anchor_free_detector.pt'

    param_dict = set_param_for_video_inference(
        dataset_type = 'kitti',
        module_rootdir = module_rootdir,
        trained_weights_file = os.path.join(weight_rootdir, weights_file))

    device = param_dict['device']
    dataset_param = param_dict['dataset_param']
    detector = param_dict['detector']

    deltas_mean = torch.tensor(dataset_param.deltas_mean, dtype=torch.float32, device=device)
    deltas_std = torch.tensor(dataset_param.deltas_std, dtype=torch.float32, device=device)
    grid_coord = dataset_param.grid_coord.to(device)

    resize_factor_w = dataset_param.IMG_W / dataset_param.IMG_RESIZED_W
    resize_factor_h = dataset_param.IMG_H / dataset_param.IMG_RESIZED_H

    nms_thresh = NMS_THRESH
    score_threshold = torch.tensor(SCORE_THRESH, dtype=torch.float32).to(device)

    set_video_output_dir(OUT_VIDEO_DIR)

    for idx, scene in enumerate(config_dataset.kitti_all_sequences_folders_test[:17]):
        image_dir = config_dataset.kitti_image_dir_test
        frames_path = get_frame_path_list(scene, image_dir, dataset_rootdir)
        
        video_name = scene + OUT_VIDEO_EXT
        out_video_path = os.path.join(OUT_VIDEO_DIR, video_name)

        print(f"video {idx}: {video_name}")
        print(f"out video path: {out_video_path}")
        write_detections_to_video(
            frames_path, FPS,
            device, deltas_mean, deltas_std,
            resize_factor_w, resize_factor_h,
            out_video_path, dataset_param,
            nms_thresh, score_threshold,
            grid_coord, detector)
        print('=' * 100)