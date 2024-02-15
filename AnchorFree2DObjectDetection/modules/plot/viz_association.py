# -------------------------------------------------------------------------------------------------
# AUthor: Udit Bhaskar
# Description: various plot functions
# -------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------
def viz_gt_and_assiciated_bbox(
    matched_bbox, matched_cls, gt_bbox, 
    img_h, img_w, figsize):
    """ Validate if the associated boxes are consistent with the gt box """

    black_image1 = np.zeros((img_h, img_w, 3), dtype=int)
    black_image2 = np.zeros((img_h, img_w, 3), dtype=int)

    num_bboxes = matched_bbox.shape[0]
    red_channel = np.random.randint(low=0, high=255, size=num_bboxes)
    green_channel = np.random.randint(low=0, high=255, size=num_bboxes)
    blue_channel = np.random.randint(low=0, high=255, size=num_bboxes)
    rgb_vals = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    for i in range(gt_bbox.shape[0]):
        color_cv2 = (int(rgb_vals[i, 0]), int(rgb_vals[i, 1]), int(rgb_vals[i, 2]))
        tl = ( int(gt_bbox[i, 0]), int(gt_bbox[i, 1]) )
        br = ( int(gt_bbox[i, 2]), int(gt_bbox[i, 3]) )
        cv2.rectangle(black_image1, tl, br, color_cv2, thickness=1)

        mask = matched_cls == i
        associated_bbox = matched_bbox[mask]
        for j in range(associated_bbox.shape[0]):
            tl = ( int(associated_bbox[j, 0]), int(associated_bbox[j, 1]) )
            br = ( int(associated_bbox[j, 2]), int(associated_bbox[j, 3]) )
            cv2.rectangle(black_image2, tl, br, color_cv2, thickness=1)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(black_image1)
    ax[0].axis('off')
    ax[1].imshow(black_image2)
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------------------------
def viz_pos_and_neg_locations(
    matched_cls, grid_coord, gt_bbox,
    img_h, img_w, figsize):
    """ Validate if the locations that lie within the box are positive and the rest are negative """

    black_image = np.zeros((img_h, img_w, 3), dtype=int)
    Yellow = (255,255,0)

    for i in range(gt_bbox.shape[0]):
        tl = ( int(gt_bbox[i, 0]), int(gt_bbox[i, 1]) )
        br = ( int(gt_bbox[i, 2]), int(gt_bbox[i, 3]) )
        cv2.rectangle(black_image, tl, br, Yellow, thickness=1)

    valid_mask = matched_cls != -1
    invalid_mask = matched_cls == -1

    plt.figure(figsize = figsize)
    plt.scatter(grid_coord[invalid_mask,0], grid_coord[invalid_mask,1], c='blue', marker='o', s=10, alpha=1.0)
    plt.scatter(grid_coord[valid_mask,0], grid_coord[valid_mask,1], c='red', marker='o', s=10, alpha=1.0)
    plt.axis('off')
    plt.imshow(black_image)
    plt.show() 

# -------------------------------------------------------------------------------------------------
def viz_box_associations(
    grid_coords, matched_cls, gt_bbox, 
    img_h, img_w, figsize):
    """ Validate if the locations are associated correctly. 
    If a point is within multiple boxes, then the point is associated to that box which has minimum area """

    black_image = np.zeros((img_h, img_w, 3), dtype=int)

    num_bboxes = matched_cls.shape[0]
    red_channel = np.random.randint(low=0, high=255, size=num_bboxes)
    green_channel = np.random.randint(low=0, high=255, size=num_bboxes)
    blue_channel = np.random.randint(low=0, high=255, size=num_bboxes)
    rgb_vals = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    plt.figure(figsize = figsize)

    for i in range(gt_bbox.shape[0]):
        color_cv2 = (int(rgb_vals[i, 0]), int(rgb_vals[i, 1]), int(rgb_vals[i, 2]))
        tl = ( int(gt_bbox[i, 0]), int(gt_bbox[i, 1]) )
        br = ( int(gt_bbox[i, 2]), int(gt_bbox[i, 3]) )
        cv2.rectangle(black_image, tl, br, color_cv2, thickness=1)

        color_plt = (int(rgb_vals[i, 0]) / 255.0, int(rgb_vals[i, 1]) / 255.0, int(rgb_vals[i, 2]) / 255.0)
        loc_coord_box_i = grid_coords[matched_cls == i]
        plt.scatter(loc_coord_box_i[:,0], loc_coord_box_i[:,1], c=color_plt, marker='o', s=6, alpha=1.0)

    plt.axis('off')
    plt.imshow(black_image)
    plt.show()