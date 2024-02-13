

# Anchor Free Object Detection 

## Introduction
This project is about the development of a **2D object detection** model using **PyTorch**, 
aiming to provide a comprehensive guide for enthusiasts, researchers, and practitioners in the domain. 
Here the object detection model is trained from scratch, incorporating a **pre-trained backbone from the Imagenet dataset**. 
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

***Anchor Free Network Architecture.*** 

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/2_detections.PNG)

***Detected Bounding Boxes.***

<br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/3_performance.PNG)

**Detection Rate vs False Positives per image at different detection thresholds (ROC Curve).**

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
       <li>Concept Level Architecture</li> 
       <li>Backbone for Feature Computation</li> 
       <li>Neck for Feature Aggregation</li> 
       <li>Head for Dense Object Detection</li> 
   </ol> 
</li>
<li><a href="#Ground-Truth-Generation">Ground Truth Generation</a>
   <ol>
       <li>Bounding Box Offsets</li> 
       <li>Centerness Score</li> 
       <li>Objectness and Object Class</li>
   </ol> 
</li>
<li><a href="#Training">Training</a>  
   <ol>
       <li> Loss Functions </li> 
       <li> Optimization method </li>
   </ol> 
</li>
<li><a href="#Performance-Evaluation">Performance Evaluation</a></li>
<li><a href="#Video-Inference">Video Inference</a></li>
   <ol>
       <li> BDD Dataset </li> 
       <li> KITTI Dataset </li>
   </ol>
<li><a href="#Conclusion">Conclusion</a></li>
<li><a href="#Reference">Reference</a></li>

</ul>
</details>

<br>


## Project Folder Structure
```bash
AnchorFree2DObjectDetection
│───hyperparam                   # Bounding Box offset statistics folder
|───labels                       # aggregated GT labels folder of KITTI and BDD dataset
│───mAP                          # module to compute mAP ( https://github.com/Cartucho/mAP.git )
│───model_weights                # model weights folder after training
│───tensorboard                  # folder for tensorboard data visualization data
│───modules                      # main modules 
      │───augmentation           # scripts for image augmentation functions            
      │───dataset_utils          # scripts for data analysis and dataset generation functions
      │───evaluation             # scripts for detector evaluation and threshold determination functions     
      │───first_stage            # scripts for defining the model and ground truth generation function for dense object detection
      │───hyperparam             # scripts for computing the offsets and their statistics from training data    
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
      │───second-stage           # <work under progress >scripts for defining the model and ground truth generation function for second stage object detection              
│───tests                                    # folder for testing and validation scripts
│   config_dataset.py                        # parameters and constants for dataset 
│   config_neuralnet_stage1.py               # model design parameters
│   script1_create_datasets.py               # aggregate gt labels and save it inside the 'labels' folder
│   script2_gen_hyperparam.py                # aggregate and save the box offsets and its statistics inside the 'hyperparam' folder
│   script3_train_model.ipynb                # notebook to train the model 
│   script4_inference_bdd.ipynb              # run inference on the bdd dataset
│   script4_inference_kitti.ipynb            # run inference on the kitti dataset          
│   script5_compute_mAP_bdd.ipynb            # compute mean average precison (mAP) on the kitti dataset
│   video_inference_bdd.py                   # run inference on the bdd dataset video
│   video_inference_kitti.py                 # run inference on the kitti dataset frame sequence video
│   write_detection_to_video_bdd.py          # run inference and save results on a video for bdd inside the 'video_inference' folder
│   write_detection_to_video_kitti.py        # run inference and save results on a video for kitti inside the 'video_inference' folder                
```
[Back to TOC](#t0)

<br>

## Exploratory Data Analysis

[Back to TOC](#t0)
<br>

## Model Architecture
[Back to TOC](#t0)

<br>

## Ground Truth Generation
[Back to TOC](#t0)

<br>

## Training
[Back to TOC](#t0)

<br>

## Performance Evaluation
[Back to TOC](#t0)

<br>

## Video Inference
[Back to TOC](#t0)

<br>

## Conclusion
[Back to TOC](#t0)

<br>

## Reference
[Back to TOC](#t0)








<br><br><br><br><br><br><br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/kitti_video_infer1.gif)

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/kitti_video_infer2.gif)

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/kitti_video_infer3.gif)

**Detection from Video Clip.**

<br>


