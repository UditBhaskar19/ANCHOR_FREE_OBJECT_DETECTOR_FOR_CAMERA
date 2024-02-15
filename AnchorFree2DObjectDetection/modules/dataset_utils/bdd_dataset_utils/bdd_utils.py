# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : bdd dataset utilities 
# ---------------------------------------------------------------------------------------------------------------------
import json, os
import numpy as np
from typing import List, Dict, Union, Tuple

# ---------------------------------------------------------------------------------------------------------------------
def aggregate_image_path(
    dataset_rootdir: str,
    train_images_dir: str, 
    val_images_dir: str, 
    test_images_dir: str) -> Dict[str, List[str]]:
    """ Function that aggregates image file paths for train, validation and test sets. 
    Each of these sets are present in directories : train_images_dir, val_images_dir, test_images_dir """

    train_images_dir_ = os.path.join(dataset_rootdir, train_images_dir)
    test_images_dir_ = os.path.join(dataset_rootdir, test_images_dir)
    val_images_dir_ = os.path.join(dataset_rootdir, val_images_dir)

    train_images_names = [fname for fname in os.listdir(train_images_dir_) if fname.endswith('.jpg')]
    test_images_names = [fname for fname in os.listdir(test_images_dir_) if fname.endswith('.jpg')]
    val_images_names = [fname for fname in os.listdir(val_images_dir_) if fname.endswith('.jpg')]

    train_images_path = [ os.path.join(train_images_dir, fname) for fname in train_images_names ]
    val_images_path = [ os.path.join(val_images_dir, fname) for fname in val_images_names ]
    test_images_path = [ os.path.join(test_images_dir, fname) for fname in test_images_names ]

    return {
        'train_images_names': train_images_names,
        'test_images_names': test_images_names,
        'val_images_names': val_images_names,
        'train_images_path': train_images_path,
        'val_images_path': val_images_path,
        'test_images_path': test_images_path
    }

# ---------------------------------------------------------------------------------------------------------------------
def aggregate_ground_truths(
    dataset_rootdir: str,
    label_file_path: str, 
    img_dir: str, 
    lane_labels_dir: str, 
    verbose: bool = True) \
        -> Tuple[List[Dict[str, Union[str, List[str], np.ndarray, List[np.ndarray]]]], 
                 Dict[str, Dict[str, bool]]]:
    """ The label file is a list of dict (JSON) and is structured as fllows:
        - name : image file name
        - attributes : a dictionary having the following attributes
                       - weather : type of weather e.g clear
                       - scene : city, rural etc
                       - timeofday : 
                       - timestamp :
        - labels : a LIST of dictionary where each entry corrosponds to a single object in the image and has the following attributes
                   - category : e.g traffic light, car, etc
                   - attributes : a dictionary having the following attributes
                                  -  occluded
                                  -  truncated
                                  -  trafficLightColor
                   - box2d : a dictionary with the attributes : x1, y1, x2, y2
                   < if the catagory is drivable area> the attributes are as follows
                   - poly2d : a list of polygons 
                   < if the category is lane> the attributes are different
                   -  poly2d : vertices
                   -  attributes : 'laneDirection': 'parallel'
                                 : 'laneStyle': 'solid'
                                 : 'laneType': 'road curb'
    """
    print('Load JSON file .. please wait')
    label_file_path_ = os.path.join(dataset_rootdir, label_file_path)
    with open(label_file_path_, 'r') as file: all_data = json.load(file)

    selected_labels = []   # selected in the sense that not all attributes are considered, some object, lane , drivable-area attributes are omitted

    # used mainly to get a list of all the label names ( datatset verification and validation purpose )
    all_obj_labels = {}
    all_traffic_light_labels = {}
    all_laneDirection_labels = {}
    all_laneStyle_labels = {}
    all_laneType_labels = {}
    all_weather = {}
    all_scene = {}
    all_timeofday = {}
    
    for i, data in enumerate(all_data):   # for each image
        objCategory = []         # object class
        trafficLight = []        # if the object class is traffic light, what color is the traffic light
        boundingBox2D = []       # object box atrributes
        drivableArea = []        # drivalble area list of polygon coordinates
        laneSegments = []        # lane segments
        laneDirections = []      # lane directions for each lanes 
        laneStyles = []          # lane styles for each lanes
        laneTypes = []           # lane types for each lanes

        for label in data['labels']:

            if label['category'] not in ['drivable area', 'lane']:
                objCategory.append(label['category'])
                trafficLight.append(label['attributes']['trafficLightColor'])
                boundingBox2D.append( np.array([ label['box2d']['x1'], label['box2d']['y1'], 
                                                 label['box2d']['x2'], label['box2d']['y2']], dtype=np.float32) )
                all_obj_labels[label['category']] = True
                all_traffic_light_labels[label['attributes']['trafficLightColor']] = True
                
            elif label['category'] == 'drivable area':
                for polygon in label['poly2d']:      # polygon['vertices'] is a list of list ( a list of x, y coordinates where the x, y coordinates are in a list)
                    polygon = np.array(polygon['vertices'], dtype=np.float32)
                    drivableArea.append(polygon)

            elif label['category'] == 'lane':
                laneDirections.append( label['attributes']['laneDirection'] )
                laneStyles.append( label['attributes']['laneStyle'] )
                laneTypes.append( label['attributes']['laneType'] )
                for segments in label['poly2d']:
                    segments = np.array(segments['vertices'], dtype=np.float32)
                    laneSegments.append(segments)    

                all_laneDirection_labels[label['attributes']['laneDirection']] = True   
                all_laneStyle_labels[label['attributes']['laneStyle']] = True
                all_laneType_labels[label['attributes']['laneType']] = True  

        if len(boundingBox2D) > 0: box2d_nparray = np.stack(boundingBox2D, axis=0)
        else: box2d_nparray = np.zeros(shape=(0, 4))

        # get the image and lane imag path
        img_path = os.path.join(img_dir, data['name'])
        lane_img_path = os.path.join(lane_labels_dir, f"{data['name'].split('.')[0]}.png")

        # scene attributes
        all_weather[data['attributes']['weather']] = True
        all_scene[data['attributes']['scene']] = True
        all_timeofday[data['attributes']['timeofday']] = True

        selected_labels.append( { 
            'img_path' : img_path,
            'weather' : data['attributes']['weather'],
            'scene' : data['attributes']['scene'],
            'timeofday' : data['attributes']['timeofday'],
            'lane_img_path' : lane_img_path,
            'objCategory': objCategory, 
            'trafficLight': trafficLight,
            'boundingBox2D': box2d_nparray,
            'drivableArea': drivableArea,
            'laneSegments': laneSegments,
            'laneDirections': laneDirections,
            'laneStyles': laneStyles,
            'laneTypes': laneTypes })
        
        if verbose:
            if i % 2000 == 0 : print(f'annotations from {i+1}/{len(all_data)} aggregated')
    
    print(f'annotations from {len(all_data)}/{len(all_data)} aggregated : Aggregation COMPLETE')

    label_names = {
        'all_laneDirection_labels': all_laneDirection_labels,
        'all_laneStyle_labels': all_laneStyle_labels,
        'all_laneType_labels': all_laneType_labels,
        'all_obj_labels': all_obj_labels,
        'all_traffic_light_labels': all_traffic_light_labels,
        'all_weather': all_weather,
        'all_scene': all_scene,
        'all_timeofday': all_timeofday
    }

    return selected_labels, label_names