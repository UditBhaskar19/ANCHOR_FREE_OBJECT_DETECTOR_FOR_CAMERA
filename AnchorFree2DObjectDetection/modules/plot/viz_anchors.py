# -------------------------------------------------------------------------------------------------
# AUthor: Udit Bhaskar
# Description: various plot functions
# -------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------
def visualize_anchors(n, scale_w, scale_h, clusters, figsize=(10, 10)):
    Red = (255,0,0)
    black_image = np.zeros((scale_h, scale_w, 3), dtype=int)

    print(f'block: {n}')
    
    for box in range(clusters.shape[0]):
        w = clusters[box, 0]
        h = clusters[box, 1]
        tl = ( int((black_image.shape[1] - w)*0.5), 
               int((black_image.shape[0] - h)*0.5))
        br = ( int((black_image.shape[1] + w)*0.5), 
               int((black_image.shape[0] + h)*0.5))
        cv2.rectangle(black_image, tl, br, Red, thickness=2)
    
    plt.figure(figsize=figsize)
    plt.imshow(black_image)
    plt.show()

# -------------------------------------------------------------------------------------------------
def visualize_anchors_all_blocks(
    num_blocks, num_anchors, 
    scale_w, scale_h, 
    anchor_dict, 
    figsize=(10, 10)):

    Red = (255,0,0)
    Yellow = (255,255,0)
    black_image_blocks = np.zeros((num_blocks, scale_h, scale_w, 3), dtype=int)

    for idx, (key, clusters) in enumerate(anchor_dict.items()):
        for box in range(clusters.shape[0]):
            w = clusters[box, 0]
            h = clusters[box, 1]
            tl = ( int((scale_w - w)*0.5), int((scale_h - h)*0.5))
            br = ( int((scale_w + w)*0.5), int((scale_h + h)*0.5))
            cv2.rectangle(black_image_blocks[idx], tl, br, Yellow, thickness=4)

    print(f'num blocks: {num_blocks}   num anchors: {num_anchors}')
    if num_blocks == 6 or num_blocks == 5 : fig, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    elif num_blocks == 4 or num_blocks == 3 : fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    else: raise Exception("wrong value selected")

    if num_blocks == 6:
        ax[0,0].imshow(black_image_blocks[0])
        ax[0,1].imshow(black_image_blocks[1])
        ax[1,0].imshow(black_image_blocks[2])
        ax[1,1].imshow(black_image_blocks[3])
        ax[2,0].imshow(black_image_blocks[4])
        ax[2,1].imshow(black_image_blocks[5])

    elif num_blocks == 5:
        ax[0,0].imshow(black_image_blocks[0])
        ax[0,1].imshow(black_image_blocks[1])
        ax[1,0].imshow(black_image_blocks[2])
        ax[1,1].imshow(black_image_blocks[3])
        ax[2,0].imshow(black_image_blocks[4])

    elif num_blocks == 4:
        ax[0,0].imshow(black_image_blocks[0])
        ax[0,1].imshow(black_image_blocks[1])
        ax[1,0].imshow(black_image_blocks[2])
        ax[1,1].imshow(black_image_blocks[3])

    elif num_blocks == 3:
        ax[0,0].imshow(black_image_blocks[0])
        ax[0,1].imshow(black_image_blocks[1])
        ax[1,0].imshow(black_image_blocks[2])

    else: raise Exception("wrong value selected")

    plt.figure(figsize=figsize)
    plt.show()

# -------------------------------------------------------------------------------------------------
def viz_pos_anchors(img_path, bboxs, box_mode='normalized', figsize=(25,15)):
    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if box_mode == 'normalized':
        bboxs[:, [0,2]] = bboxs[:, [0,2]] * image_rgb.shape[1]
        bboxs[:, [1,3]] = bboxs[:, [1,3]] * image_rgb.shape[0]

    # color in RGB
    Yellow = (255,255,0)
    for idx in range(bboxs.shape[0]):
        # bounding box
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=1)
        
    plt.figure(figsize = figsize)
    plt.imshow(image_rgb)
    plt.show()    

# -------------------------------------------------------------------------------------------------
def viz_gtbox_and_pos_anchors(img_path, bboxs, anchors, box_mode='normalized', figsize=(25,15)):
    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if box_mode == 'normalized':
        bboxs[:, [0,2]] = bboxs[:, [0,2]] * image_rgb.shape[1]
        bboxs[:, [1,3]] = bboxs[:, [1,3]] * image_rgb.shape[0]
        anchors[:, [0,2]] = anchors[:, [0,2]] * image_rgb.shape[1]
        anchors[:, [1,3]] = anchors[:, [1,3]] * image_rgb.shape[0]

    # color in RGB
    Yellow = (255,255,0)
    Red = (255,0,0)

    # anchor boxes
    for idx in range(anchors.shape[0]):
        box = anchors[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=1)

    # bounding boxes
    for idx in range(bboxs.shape[0]):
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Red, thickness=3)

    plt.figure(figsize = figsize)
    plt.imshow(image_rgb)
    plt.show() 


def viz_gtbox_and_pos_anchors_resized(
    img_path, 
    bboxs, anchors, 
    img_w, img_h, 
    box_mode='unnormalized', 
    figsize=(25,15)):

    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    if box_mode == 'normalized':
        bboxs[:, [0,2]] = bboxs[:, [0,2]] * image_rgb.shape[1]
        bboxs[:, [1,3]] = bboxs[:, [1,3]] * image_rgb.shape[0]
        anchors[:, [0,2]] = anchors[:, [0,2]] * image_rgb.shape[1]
        anchors[:, [1,3]] = anchors[:, [1,3]] * image_rgb.shape[0]

    # color in RGB
    Yellow = (255,255,0)
    Red = (255,0,0)

    # anchor boxes
    for idx in range(anchors.shape[0]):
        box = anchors[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Yellow, thickness=1)

    # bounding boxes
    for idx in range(bboxs.shape[0]):
        box = bboxs[idx, :4]
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_rgb, tl, br, Red, thickness=1)

    plt.figure(figsize = figsize)
    plt.imshow(image_rgb)
    plt.show() 

# -------------------------------------------------------------------------------------------------
def viz_anchor_coordinates(
    img_path, bboxs, pos_anchors, neg_anchors, neu_anchors, 
    box_mode='normalized', figsize=(25,15)):

    # import matplotlib
    # matplotlib.use('TKAgg')

    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # image_rgb = plt.imread(img_path)
    if box_mode == 'normalized':
        if bboxs.shape[0] > 0:
            bboxs[:, [0,2]] = bboxs[:, [0,2]] * image_rgb.shape[1]
            bboxs[:, [1,3]] = bboxs[:, [1,3]] * image_rgb.shape[0]

        pos_anchors[:, 0] = pos_anchors[:, 0] * image_rgb.shape[1]
        pos_anchors[:, 1] = pos_anchors[:, 1] * image_rgb.shape[0]
        neg_anchors[:, 0] = neg_anchors[:, 0] * image_rgb.shape[1]
        neg_anchors[:, 1] = neg_anchors[:, 1] * image_rgb.shape[0]
        neu_anchors[:, 0] = neu_anchors[:, 0] * image_rgb.shape[1]
        neu_anchors[:, 1] = neu_anchors[:, 1] * image_rgb.shape[0]

    # bounding boxes
    if bboxs.shape[0] > 0:
        Green = (0,255,0)
        for idx in range(bboxs.shape[0]):
            box = bboxs[idx, :4]
            tl = (int(box[0]), int(box[1]))
            br = (int(box[2]), int(box[3]))
            cv2.rectangle(image_rgb, tl, br, Green, thickness=3)

    size = 30
    marker = 'o'
    alpha = 1.0
    plt.figure(figsize = figsize)    
    plt.scatter(neg_anchors[:,0], neg_anchors[:,1], c='yellow', marker=marker, s=1, alpha=alpha)
    plt.scatter(neu_anchors[:,0], neu_anchors[:,1], c='blue', marker=marker, s=10, alpha=alpha)
    plt.scatter(pos_anchors[:,0], pos_anchors[:,1], c='red', marker=marker, s=size, alpha=alpha)
    
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.show() 