# -------------------------------------------------------------------------------------------------
# AUthor: Udit Bhaskar
# Description: various plot functions
# -------------------------------------------------------------------------------------------------
import cv2, math
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------
def show_image_data(image_mat, figsize=(25,15)):
    plt.figure(figsize = figsize)
    plt.imshow(image_mat)
    plt.axis('off')
    plt.show()

def display_img(img_path, figsize=(25,15)):
    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize = figsize)
    plt.imshow(image_rgb)
    plt.show()

# -------------------------------------------------------------------------------------------------
def vizualize_annotations(
    img_path, bboxs, class_labels, 
    lane_img_path, polygon_coord,
    alpha, beta,
    figsize=(25,15)):

    # read the image
    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    lane_image_rgb = cv2.cvtColor(cv2.imread(lane_img_path), cv2.COLOR_BGR2RGB)
    
    # color in RGB
    Black = (0,0,0)
    Yellow = (255,255,0)
    Red = (255, 0, 0)
    Green = (0, 255, 0)

    # font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2
    dx = 20; dy = 8

    # fill polygon for drivable surface
    for polygon in polygon_coord:
        polygon = polygon.astype(np.int32)
        cv2.fillPoly( image_rgb, pts=[polygon], color=Red )

    # objects
    for idx in range(bboxs.shape[0]):
        # bounding box
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=2)
        cv2.circle(image_rgb, tl, radius=4, color=Red, thickness=5)
        cv2.circle(image_rgb, br, radius=4, color=Green, thickness=5)

        # object catagory
        class_label = class_labels[idx]
        (w, h), _ = cv2.getTextSize(class_label, font, font_scale, font_thickness)
        box_coords = ((tl[0], tl[1] - h - dy), (tl[0] + w, tl[1]))
        text_coords = (tl[0], tl[1] - dy // 2)
        cv2.rectangle(image_rgb, box_coords[0], box_coords[1], Yellow, cv2.FILLED)
        cv2.putText(image_rgb, class_label, text_coords, font, font_scale, Black, font_thickness)

    # lane lines
    image_rgb = cv2.addWeighted(image_rgb, alpha, lane_image_rgb, beta, 0)

    plt.figure(figsize = figsize)
    plt.title(img_path)
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.show()

# -------------------------------------------------------------------------------------------------
def vizualize_remapped_annotations(
    img_path, bboxs, class_labels,
    box_mode='un-normalized',
    figsize=(25,15)):

    # read the image
    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    # color in RGB
    Black = (0,0,0)
    Yellow = (255,255,0)
    Red = (255, 0, 0)
    Green = (0, 255, 0)

    # font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2
    dx = 20; dy = 8

    if box_mode == 'normalized':
        bboxs[:, [0,2]] = bboxs[:, [0,2]] * image_rgb.shape[1]
        bboxs[:, [1,3]] = bboxs[:, [1,3]] * image_rgb.shape[0]

    # objects
    for idx in range(bboxs.shape[0]):
        # bounding box
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=2)
        cv2.circle(image_rgb, tl, radius=4, color=Red, thickness=5)
        cv2.circle(image_rgb, br, radius=4, color=Green, thickness=5)

        # object catagory
        box_area = ( box[2] - box[0] ) * ( box[3] - box[1] )
        # text = f'{class_labels[idx]} : {box_area}'
        text = class_labels[idx]
        (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        box_coords = ((tl[0], tl[1] - h - dy), (tl[0] + w, tl[1]))
        text_coords = (tl[0], tl[1] - dy // 2)
        cv2.rectangle(image_rgb, box_coords[0], box_coords[1], Yellow, cv2.FILLED)
        cv2.putText(image_rgb, text, text_coords, font, font_scale, Black, font_thickness)

    plt.figure(figsize = figsize)
    plt.title(img_path)
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.show()

# -------------------------------------------------------------------------------------------------
def draw_bbox_on_img_data(numpy_image, bboxs, figsize=(25,15)):
    image_rgb = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    # objects
    Yellow = (255,255,0)
    Red = (255, 0, 0)
    for idx in range(bboxs.shape[0]):
        # bounding box
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=2)
    plt.figure(figsize = figsize)
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.show()


def vizualize_bbox(img_path, bboxs, figsize=(25,15)):
    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # objects
    Yellow = (255,255,0)
    Red = (255, 0, 0)
    for idx in range(bboxs.shape[0]):
        # bounding box
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=4)
    plt.figure(figsize = figsize)
    plt.title(img_path)
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.show()

# -------------------------------------------------------------------------------------------------
def vizualize_bbox_nxn(img_path, bboxs, nrows=3, ncols=2, figsize=(25,15)):
    imgs = []
    Yellow = (255,255,0)
    for idx, path in enumerate(img_path):
        image_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=4)
        imgs.append(image_rgb)
    
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax[i,j].imshow(imgs[idx])
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.show()




def vizualize_bbox_resized(img_path, bboxs, img_w, img_h, thickness=1, figsize=(10,8)):
    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    Yellow = (255,255,0)
    Red = (255, 0, 0)
    for idx in range(bboxs.shape[0]):
        # bounding box
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=thickness)
    plt.figure(figsize = figsize)
    plt.title(img_path)
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.show()



def vizualize_bbox_resized2(img_path, image_rgb, bboxs, figsize=(10,8)):
    Yellow = (255,255,0)
    Red = (255, 0, 0)
    for idx in range(bboxs.shape[0]):
        # bounding box
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=2)
    plt.figure(figsize = figsize)
    plt.title(img_path)
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.show()




def  viz_gt_pos_false_boxes_resized(
        img_path, detections, false_positives, gt_box, 
        img_w, img_h, figsize=(10,8)):
    
    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    Yellow = (255,255,0)
    Red = (255, 0, 0)
    Green = (0, 255, 0)

    # plot false positives
    for idx in range(false_positives.shape[0]):
        # bounding box
        box = false_positives[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=1)

    # plot detections
    for idx in range(detections.shape[0]):
        # bounding box
        box = detections[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Red, thickness=2)

    # plot gt
    for idx in range(gt_box.shape[0]):
        # bounding box
        box = gt_box[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Green, thickness=1)

    plt.figure(figsize = figsize)
    plt.title(img_path)
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.show()





def draw_associations_single_image(
    img_path, 
    img_w, img_h,
    valid_locations, 
    invalid_locations,
    figsize=(10,8)):

    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(invalid_locations[:,0], invalid_locations[:,1], c='blue', marker='o', s=1.0, alpha=1.0)
    ax.scatter(valid_locations[:,0], valid_locations[:,1], c='yellow', marker='o', s=1.0, alpha=1.0)
    ax.axis('off')
    ax.imshow(image_rgb)
    plt.tight_layout()
    plt.show()



def draw_associations_single_image2(
    img_path, 
    img_w, img_h,
    valid_locations, 
    invalid_locations,
    ignored_locations,
    figsize=(10,8)):

    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(invalid_locations[:,0], invalid_locations[:,1], c='blue', marker='o', s=1.0, alpha=1.0)
    ax.scatter(valid_locations[:,0], valid_locations[:,1], c='yellow', marker='o', s=1.0, alpha=1.0)
    ax.scatter(ignored_locations[:,0], ignored_locations[:,1], c='red', marker='o', s=2.0, alpha=1.0)
    ax.axis('off')
    ax.imshow(image_rgb)
    plt.tight_layout()
    plt.show()
    

def visualize_centerness(
    img_path, 
    img_w, img_h,
    valid_locations,
    colors, 
    size = 1.0,
    alpha = 1.0,
    figsize=(10,8)):

    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(valid_locations[:,0], valid_locations[:,1], c=colors, marker='o', s=size, alpha=alpha)
    ax.axis('off')
    ax.imshow(image_rgb)
    plt.tight_layout()
    plt.show()







def draw_associations_single_levels(img, ax, valid_locations, invalid_locations, bbox, size):
    Yellow = (255,255,0)
    Red = (255,0,0)
    Green = (0,255,0)
    Blue = (0,0,255)
    for i in range(-bbox.shape[0]):
        tl = ( int(bbox[i, 0]), int(bbox[i, 1]) )
        br = ( int(bbox[i, 2]), int(bbox[i, 3]) )
        cv2.rectangle(img, tl, br, Yellow, thickness=1)
    ax.scatter(invalid_locations[:,0], invalid_locations[:,1], c='blue', marker='o', s=size, alpha=1.0)
    ax.scatter(valid_locations[:,0], valid_locations[:,1], c='red', marker='o', s=size, alpha=1.0)
    return img



def vizualize_associations_all_levels(
    img_path, 
    img_w, img_h,
    valid_locations, 
    invalid_locations, 
    valid_associated_boxes, 
    bbox, num_levels,
    figsize=(10,8)):

    num_cols = 2
    num_rows = math.ceil(num_levels / num_cols)
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    ax[-1, -1].axis('off')

    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    sizes = {'c0': 0.4, 'c1': 4, 'c2': 10, 'c3': 14, 'c4': 20, 'c5': 20}

    for i, level in enumerate(valid_locations.keys()):
        row_idx = math.floor( i / num_cols )
        col_idx = i - row_idx * num_cols
        valid_loc = valid_locations[level]
        invalid_loc = invalid_locations[level]
        size = sizes[level]
        img = draw_associations_single_levels(
            image_rgb.copy(), ax[row_idx, col_idx], valid_loc, invalid_loc, bbox, size)
        ax[row_idx, col_idx].axis('off')
        ax[row_idx, col_idx].imshow(img)

    plt.tight_layout()
    plt.show()
