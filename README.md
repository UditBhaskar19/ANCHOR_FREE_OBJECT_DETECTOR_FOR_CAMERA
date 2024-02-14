

# Anchor Free Object Detection 

## Introduction
This project is about the development of an **Anchor free 2D object detection** model using **PyTorch**, 
that aims to provide a comprehensive guide for enthusiasts, researchers, and practitioners in the domain. 
Here the object detection model is trained from scratch, incorporating a **ImageNet re-trained backbone from PyTorch**. The model is trained using a modest system configuration ( NVIDIA RTX A2000 4 GB Laptop GPU ), thus enabling users with low computational resources to train object detection models that gives resonably good performance.
An easy to understand and extend codebase is developed in this project.
The following are the key highlights:
   - Training a 2D object detection Model in PyTorch from scratch by utilizing 
     Imagenet dataset pre-trained backbone from PyTorch.
   - Development of an easy to understand and well documented codebase.
   - Implementation of a method for tuning the detection threshold parameters.
   - Utilizing training samples from two publicly available datasets: KITTI and BDD, 
     so as to provide a technique to merge samples from multiple training datasets,
     enabling users to utilize a diverse range of data for model generalization.

<br>

<div align="center">

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/1_model_archi.PNG)

**Anchor Free Network Architecture.**

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/2_detections.PNG)

**Detected Bounding Boxes (BDD).**

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/video_inference/kitti/gif/0007.gif)

**Detections in video (KITTI).**

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/3_performance.PNG)

**Detection Rate vs False Positives per image at different detection thresholds (ROC Curve for BDD).**

<br>

</div>

<br>




<details>
<summary>

## Table of Contents <a name="t0"></a>

</summary>

<ul>

<li><a href="#Project-Folder-Structure">Project Folder Structure</a></li>
<li><a href="#Exploratory-Data-Analysis">Exploratory Data Analysis</a>
   <ol>
      <li><a href="#Scene-and-Label-Instance">Scene and Label Instance</a></li>
      <li><a href="#Bounding-box-distribution">Bounding box distribution</a></li>
      <li><a href="#Wrong-annotations">Wrong annotations</a></li>
      <li><a href="#Dataset-Modification">Dataset Modification</a></li>
   </ol>
</li>
<li><a href="#Model-Architecture">Model Architecture</a> 
   <ol>
       <li><a href="#Concept-Level-Architecture">Concept Level Architecture</a></li> 
       <li><a href="#Backbone-for-Feature-Computation">Backbone for Feature Computation</a></li> 
       <li><a href="#Neck-for-Feature-Aggregation">Neck for Feature Aggregation</a></li> 
       <li><a href="#Head-for-Dense-Object Detection">Head for Dense Object Detection</a></li> 
   </ol> 
</li>
<li><a href="#Ground-Truth-Generation">Ground Truth Generation</a>
   <ol>
       <li><a href="#Bounding-Box-Offsets">Bounding Box Offsets</a></li> 
       <li><a href="#Centerness-Score">Centerness Score</a></li> 
       <li><a href="#Objectness-and-Object-Class">Objectness and Object Class</a></li>
   </ol> 
</li>
<li><a href="#Training">Training</a>  
   <ol>
       <li><a href="#Augmentation">Augmentation</a></li>
       <li><a href="#Loss-Functions">Loss Functions</a></li> 
       <li><a href="#Optimization-method">Optimization method</a></li>
   </ol> 
</li>
<li><a href="#Performance-Evaluation">Performance Evaluation</a></li>
<li><a href="#Conclusion">Conclusion</a></li>
<li><a href="#Reference">Reference</a></li>

</ul>
</details>

<br>


## Project Folder Structure
```bash
AnchorFree2DObjectDetection
│───hyperparam                   # Statistical data of the Bounding Box offsets
|───labels                       # aggregated GT labels data of KITTI and BDD dataset
│───mAP                          # module to compute mAP ( https://github.com/Cartucho/mAP.git )
│───model_weights                # model weights data after training
│───tensorboard                  # data folder for loss visualization in tensorboard.
│───modules                      # main modules 
      │───augmentation           # scripts for image augmentation functions            
      │───dataset_utils          # scripts for data analysis and dataset generation
      │───evaluation             # scripts for detector evaluation and threshold determination   
      │───first_stage            # scripts for defining the model and ground truth generation function for dense object detection
      │───hyperparam             # scripts for computing the bounding box offsets statistics from training data    
      │───loss                   # loss functions
      │───neural_net             # scripts for defining various neural net blocks             
            │───backbone               # model backbone blocks
            │───bifpn                  # BIFPN blocks for model neck            
            │───fpn                    # FPN blocks for model neck
            │───head                   # blocks for model head            
            │   common.py              # common model building blocks
            │   constants.py           # constants for model construction  
      │───plot                   # contains plotting functions
      │───pretrained             # scripts for loading the pre-trained backbone from pytorch            
      │───proposal               # scripts for proposal generation
      │───second-stage           # <work under progress> scripts for defining the model and ground truth generation function for second stage object detection              
│───tests                                    # folder for testing and validation scripts
│   config_dataset.py                        # parameters and constants for dataset 
│   config_neuralnet_stage1.py               # model design parameters
│   script1_create_datasets.py               # aggregate gt labels and save it inside the 'labels' folder
│   script2_gen_hyperparam.py                # aggregate and save the box offsets and its statistics inside the 'hyperparam' folder
│   script3_train_model.ipynb                # notebook to train the model 
│   script4_inference_bdd.ipynb              # run inference on the bdd dataset images
│   script4_inference_kitti.ipynb            # run inference on the kitti dataset images         
│   script5_compute_mAP_bdd.ipynb            # compute mean average precison (mAP) on the kitti dataset
│   video_inference_bdd.py                   # run inference on the bdd dataset video
│   video_inference_kitti.py                 # run inference on the kitti dataset frame sequence video
│   write_detection_to_video_bdd.py          # run inference and save results as a video for bdd inside the 'video_inference' folder
│   write_detection_to_video_kitti.py        # run inference and save results as a video for kitti inside the 'video_inference' folder                
```
[TOC](#t0)

<br>




## Exploratory Data Analysis
To have good performance from a trained object detection model, the training dataset needs to be large, diverse, balanced and the annotation has to be correct. BDD dataset is adequately large to train a resonably good performing model. Below are the data analysis conducted to get an insight about the quality of the dataset where good quality means that the training dataset has to be diverse and balanced.

### Scene and Label Instance
![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/4_eda_class_count.PNG)

<div align="center">

*Number of instances of different classes and scenes.* 
</div>

<br>

**Observations**
<ul>
   <li>There is a huge intra-class as well as inter-clss imbalance in the dataset (depends on how we are considering the intra and inter class).</li>
   <li>The intra-class imbalance is present in the number of instances of traffic light, where there is much less number of yellow traffic lights. The red and green instances are resonably balanced.</li>
   <li>The intra-class imbalance is also observed in the number of instances of road vehicles, where the car class has huge number of instances than other classes like truck and bus.</li>
   <li>The inter-class imbalance can be seen in the number of instances of vehicles and non-vehicles, where the car class has huge number of instances than other classes like person, rider, train etc.</li>
</ul>

[TOC](#t0)
<br>

### Bounding box distribution
![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/5_box_distribution.png)

<div align="center">

*Annotated bounding box dimension scatter plot.*
</div>

<br>

**Observations**
<ul>
   <li>From the plot we can observe that there are some boxes that are potentially incorrect annotations. These either have extreme aspect ratio or the area is too small</li>
</ul>

[TOC](#t0)
<br>

### Wrong annotations
If we select those boxes from the previous scatter plot that has some **extreme aspect ratio** or the **area is very small**, we would be able to identfy annotation errors. Some of them can be categorized as follows.
<ul>
<li> 

**Box area too small**

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/6_box_area_too_small.PNG) 

</li>
<li> 

**Extreme Box Aspect Ratio**

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/6_box_aspect_ratio_extreme.PNG) 

</li>
<li> 

**Incorrect Class**

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/6_incorrect_class.PNG) 

</li>


[TOC](#t0)
<br>

### Dataset Modification
Based on the above analysis the training samples and the dataset annotations are modified to 
<ul>
   <li>Simplify the development of object detection model in version 1 by reducing the number of classes and removing the highly imbalanced and irrelevant classes.</li> 
   <li>Reduce the number of wrong and low quality annotations. </li>
</ul>

<br>

The modifications are as follows:
<ul>
<li>

**Car**, **bus**, **truck** are merged as **vehicle**; **person** and **rider** are merged as **person**. The remaining classes are part of negative class.</li>
<li>Select boxes that satisfy the below conditions:
<ul>
<li> Box width &ge; 5 pixels </li>
<li> Box heighth &ge; 5 pixels </li>
<li> 0.1 &le; Box aspect ratio &le; 10 </li>
</ul></li>
</ul>

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/6_dataset_modifications.PNG)

<br>

**Relevant Scripts**

<table>
<tr><td>

|                         SCRIPT                       |               LINK                 |
|:----------------------------------------------------:|:------------------------------------------:|
|    1_1_eda_vis_anno_data.ipynb                       |          | 
|    1_2_eda_plot_label_count_distrib.ipynb            |                                  |
|    1_3_eda_bbox_distrib.ipynb                        |                             |
|    1_4_eda_vis_different_obj_categories.ipynb        |                  |
|    1_5_eda_identifying_anno_errors.ipynb             |          | 
|    2_1_eda_vis_remapped_anno_data.ipynb              |                                |
|    2_2_eda_plot_remapped_label_count_distrib.ipynb   |                             |
|    2_3_eda_remapped_bbox_distrib.ipynb               |                 |
|    2_4_eda_vis_remapped_obj_categories.ipynb         |          | 
|    2_5_eda_identifying_outliers.ipynb                |                                  |

</td></tr> 
</table>


[TOC](#t0)
<br>






## Model Architecture

### Concept Level Architecture

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/7_high_level_archi.PNG)

### Backbone for Feature Computation

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/7_backbone_archi.PNG)

### Neck for Feature Aggregation

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/7_bifpn.PNG)

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/7_bifpn_formulas.PNG)

### Head for Dense Object Detection

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/7_head_archi.PNG)


### Architecture Summary

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/7_summary.PNG)


[TOC](#t0)

<br>






## Ground Truth Generation
Each of the anchors corrospond to an object hypothesis where the network shall learn to predict 4 values : **box offsets**, **centerness score**, **objectness score**, and **classification score** from the image. The groundtruth for training is computed as follows.

### Bounding Box Offsets

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/8_box_offsets.PNG)

### Centerness Score

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/8_centerness.PNG)

### Objectness and Object Class

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/8_one_hot.PNG)


[TOC](#t0)

<br>





## Training

### Augmentation 
Augmentation is performed during training. The augmentation process is depicted as follows

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/9_augment1.PNG)

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/9_augment2.PNG)

<br>

### Loss Functions

<div align="center">

<table>
<tr><td>

|                 TASK                 |    LOSS FUNCTION                           |
|:------------------------------------:|:------------------------------------------:|
|    Class Prediction                  |      Class Weighted Cross Entrophy Loss    | 
|    Objectness Prediction             |      Focal Loss                            |
|    Box Offset Regression             |      Smooth L1 Loss                        |
|    Centerness Score Regression       |      Binary Cross Entrophy Loss            |

</td></tr> 
</table>

</div>

<br>

### Optimization Method
Either **SGD with momentum** or **AdamW** oprimization method can be used. Refer to these scripts for more details: [script1](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/modules/first_stage/set_parameters_for_training.py)
[script2](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/script3_train_model.ipynb)

<br>

[TOC](#t0)

<br>

## Performance Evaluation
[Back to TOC](#t0)

<br>

## Conclusion
[Back to TOC](#t0)

<br>

## Reference
[Back to TOC](#t0)

