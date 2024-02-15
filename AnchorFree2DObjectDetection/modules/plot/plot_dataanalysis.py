# -------------------------------------------------------------------------------------------------
# AUthor: Udit Bhaskar
# Description: various plot functions
# -------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------
def plot_categorical_data(description, data, figsize=(8, 4)):
    
    keys = list(data.keys())
    vals = np.array(list(data.values()))
    
    sorted_idx = np.argsort(vals)[::-1]
    vals = vals[sorted_idx]
    keys = [keys[sorted_idx[i]] for i in range(sorted_idx.shape[0])]
    
    plt.figure(figsize=figsize)
    plt.bar(keys, vals)
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    plt.title(description)
    plt.show()

# -------------------------------------------------------------------------------------------------
def plot_bbox_width_and_heights(
    description,
    bbox_summary,
    figsize=(8, 4),
    plot_traffic_light = True,
    plot_traffic_sign = True,
    plot_car = True,
    plot_bus = True,
    plot_tuck = True,
    plot_rider = True,
    plot_bike = True,
    plot_ped = True,
    plot_motor = True,
    plot_train = True):

    s = 1.0
    _, ax = plt.subplots(figsize=figsize)
    if plot_car: ax.scatter(bbox_summary['obj_category_bboxes_h']['car'], bbox_summary['obj_category_bboxes_w']['car'], s, color='cyan', marker='.', label='car')
    if plot_traffic_sign: ax.scatter(bbox_summary['obj_category_bboxes_h']['traffic sign'], bbox_summary['obj_category_bboxes_w']['traffic sign'], s, color='yellow', marker='.', label='traffic sign')
    if plot_traffic_light: ax.scatter(bbox_summary['obj_category_bboxes_h']['traffic light'], bbox_summary['obj_category_bboxes_w']['traffic light'], s, color='orange', marker='.', label='traffic light')
    if plot_ped: ax.scatter(bbox_summary['obj_category_bboxes_h']['person'], bbox_summary['obj_category_bboxes_w']['person'], s, color='green', marker='.', label='ped')
    if plot_tuck: ax.scatter(bbox_summary['obj_category_bboxes_h']['truck'], bbox_summary['obj_category_bboxes_w']['truck'], s, color='blue', marker='.', label='truck')
    if plot_bus: ax.scatter(bbox_summary['obj_category_bboxes_h']['bus'], bbox_summary['obj_category_bboxes_w']['bus'], s, color='red', marker='.', label='bus')
    if plot_bike: ax.scatter(bbox_summary['obj_category_bboxes_h']['bike'], bbox_summary['obj_category_bboxes_w']['bike'], s, color='magenta', marker='.', label='bike')
    if plot_rider: ax.scatter(bbox_summary['obj_category_bboxes_h']['rider'], bbox_summary['obj_category_bboxes_w']['rider'], s, color='purple', marker='.', label='rider')
    if plot_motor: ax.scatter(bbox_summary['obj_category_bboxes_h']['motor'], bbox_summary['obj_category_bboxes_w']['motor'], s, color='brown', marker='.', label='motor')
    if plot_train: ax.scatter(bbox_summary['obj_category_bboxes_h']['train'], bbox_summary['obj_category_bboxes_w']['train'], s, color='black', marker='.', label='train')
    ax.set_xlabel('bounding box Height')
    ax.set_ylabel('bounding box Width')
    ax.legend()
    plt.title(description)
    plt.show()

# -------------------------------------------------------------------------------------------------
def plot_remapped_bbox_width_and_heights(
    description,
    bbox_summary,
    figsize=(8, 4),
    plot_veh = True,
    plot_ped = True):

    s = 1.0
    _, ax = plt.subplots(figsize=figsize)
    if plot_veh: ax.scatter(bbox_summary['obj_category_bboxes_h']['vehicle'], bbox_summary['obj_category_bboxes_w']['vehicle'], s, color='cyan', marker='.', label='vehicle')
    if plot_ped: ax.scatter(bbox_summary['obj_category_bboxes_h']['person'], bbox_summary['obj_category_bboxes_w']['person'], s, color='green', marker='.', label='person')
    ax.set_xlabel('bounding box Height')
    ax.set_ylabel('bounding box Width')
    ax.legend()
    plt.title(description)
    plt.show()
   

def plot_remapped_bbox_width_and_heights_kitti(
    description,
    bbox_summary,
    figsize=(8, 4),
    plot_veh = True,
    plot_ped = True):

    s = 1.0
    _, ax = plt.subplots(figsize=figsize)
    if plot_veh: ax.scatter(bbox_summary['obj_category_bboxes_h']['vehicle'], bbox_summary['obj_category_bboxes_w']['vehicle'], s, color='red', marker='.', label='Vehicle')
    if plot_ped: ax.scatter(bbox_summary['obj_category_bboxes_h']['person'], bbox_summary['obj_category_bboxes_w']['person'], s, color='green', marker='.', label='Pedestrian')
    ax.set_xlabel('bounding box Height')
    ax.set_ylabel('bounding box Width')
    ax.legend()
    plt.title(description)
    plt.show()
    
# -------------------------------------------------------------------------------------------------
def plot_intervals(intervals, boxes_area_sorted, figsize=(15, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(boxes_area_sorted.shape[0]), boxes_area_sorted)
    for i in intervals:
        ax.plot(i, boxes_area_sorted[i], color='red', marker='x', markersize=20, markeredgewidth=5)
    plt.show()

# -------------------------------------------------------------------------------------------------
def plot_partitioned_bbox_data(boxes_sorted, LOWER_INTERVAL, UPPER_INTERVAL, figsize = (15, 7)):
    partition12 = boxes_sorted[LOWER_INTERVAL[0]:UPPER_INTERVAL[0]]
    partition34 = boxes_sorted[LOWER_INTERVAL[1]:UPPER_INTERVAL[1]]
    partition56 = boxes_sorted[LOWER_INTERVAL[2]:UPPER_INTERVAL[2]]

    s  = 1
    description = 'bounding boxes data partitions for anchor generation'
    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(partition12[:,3]-partition12[:,1], partition12[:,2]-partition12[:,0], s, color='blue', marker='.', label='block 1 and 2')
    ax.scatter(partition34[:,3]-partition34[:,1], partition34[:,2]-partition34[:,0], s, color='green', marker='.', label='block 3 and 4')
    ax.scatter(partition56[:,3]-partition56[:,1], partition56[:,2]-partition56[:,0], s, color='red', marker='.', label='block 5 and 6')
    ax.set_xlabel('bounding box Height')
    ax.set_ylabel('bounding box Width')
    ax.legend()
    plt.title(description)
    plt.show()

# -------------------------------------------------------------------------------------------------
def plot_bbox_area_and_aspect_ratio_histogram(bbox_summary, nbins, category, figsize):
    w = bbox_summary['obj_category_bboxes_w'][category]
    h = bbox_summary['obj_category_bboxes_h'][category]
    box_area = w * h
    box_aspect_ratio = w / h

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    ax[0].hist(box_area, nbins, color='blue', alpha=0.7)
    ax[0].set_xlabel('box area')
    ax[0].set_ylabel('count')
    ax[1].hist(box_aspect_ratio, nbins, color='blue', alpha=0.7)
    ax[1].set_xlabel('box aspect ratio')
    ax[1].set_ylabel('count')
    plt.suptitle(category)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------------------------
def plot_iou_and_boxarea(boxs_area, ious, blk_name, figsize):
    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(boxs_area, ious, s=1, color='red', marker='.')
    ax.set_xlabel('bbox area')
    ax.set_ylabel('bbox to anc max iou')
    plt.title(f'scatter plot for block: {blk_name}')
    plt.show()

# -------------------------------------------------------------------------------------------------
def iou_histogram(ious, nbins, blk_name):
    plt.hist(ious, nbins, color='blue', alpha=0.7)
    plt.xlabel('Values')
    plt.ylabel('Count')
    plt.title(f'Histogram of Max IOUs for block {blk_name}')
    plt.grid(True)
    plt.show()

# -------------------------------------------------------------------------------------------------
def boxoffsets_histogram(gtoffsets, nbins, blk_name, figsize):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    ax[0,0].hist(gtoffsets[:,0], nbins, color='blue', alpha=0.7)
    ax[0,0].set_xlabel('offset x')
    ax[0,0].set_ylabel('count')
    
    ax[0,1].hist(gtoffsets[:,1], nbins, color='blue', alpha=0.7)
    ax[0,1].set_xlabel('offset y')
    ax[0,1].set_ylabel('count')

    ax[1,0].hist(gtoffsets[:,2], nbins, color='blue', alpha=0.7)
    ax[1,0].set_xlabel('offset w')
    ax[1,0].set_ylabel('count')

    ax[1,1].hist(gtoffsets[:,3], nbins, color='blue', alpha=0.7)
    ax[1,1].set_xlabel('offset h')
    ax[1,1].set_ylabel('count')

    plt.suptitle(f'box offset distribution for block {blk_name}')
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------------------------
def plot_intervals_roc(score_threshs, fp_rate_per_image, detection_rate, figsize=(15, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fp_rate_per_image, detection_rate)
    for i in range(len(detection_rate)):
        ax.plot(fp_rate_per_image[i], detection_rate[i], color='red', marker='o', markersize=6, markeredgewidth=1)
        ax.annotate(f'{score_threshs[i]}', (fp_rate_per_image[i], detection_rate[i]), textcoords="offset points", xytext=(10, 10), ha='center')
    
    # y_min = -0.05
    # y_max = max(detection_rate) + 0.1

    # x_min = -0.01
    # x_max = max(fp_rate_per_image) + 10

    ax.set_xlabel('number of false positive per image')
    ax.set_ylabel('detection rate')

    ax.grid(True, which='both', linestyle='--', alpha=0.5, color='gray')
    # ax.set_xticks(np.arange(x_min, x_max, 5), minor=True)
    # ax.set_yticks(np.arange(y_min, y_max, 0.05), minor=True)

    # ax.set_xlim(0.0, x_max)  # Set x-axis limits
    # ax.set_ylim(y_min, y_max)  # Set y-axis limits
    plt.show()



# -------------------------------------------------------------------------------------------------
def plot_intervals_roc2(score_threshs, fp_rate_per_image, detection_rate, figsize=(15, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fp_rate_per_image, detection_rate)

    for i in range(len(detection_rate)):
        legend_txt = f"{i}: ( FP Rate: {fp_rate_per_image[i]}, DET Rate: {detection_rate[i]}, Threshold: {score_threshs[i]} )"
        ax.plot(fp_rate_per_image[i], detection_rate[i], marker='o', markersize=6, markeredgewidth=1, label=legend_txt)
        # ax.annotate(f'{i}', (fp_rate_per_image[i], detection_rate[i]), textcoords="offset points", xytext=(10, 10), ha='center')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)
    ax.set_xlabel('number of false positive per image')
    ax.set_ylabel('detection rate')
    ax.grid(True, which='both', linestyle='--', alpha=0.5, color='gray')
    plt.show()
