

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

***Detection Rate vs False Positives per image at different detection thresholds (ROC Curve).***

<br>

</div>

<br>



<!-- TABLE OF CONTENTS -->
<details>
<summary>

## Table of Contents

</summary>

<ul>
   <li><a href="#project-folder-structure">Project Folder Structure></a></li>
   
   <li><a href="#project-structure">Project Structure</a></li>

</ul>
</details>

## Project Folder Structure

<br>

### Project Structure
```bash
HybridNets
│   backbone.py                   # Model configuration
|   export.py                     # UPDATED 10/2022: onnx weight with accompanying .npy anchors
│   hubconf.py                    # Pytorch Hub entrypoint
│   hybridnets_test.py            # Image inference
│   hybridnets_test_videos.py     # Video inference
│   train.py                      # Train script
│   train_ddp.py                  # DistributedDataParallel training (Multi GPUs)
│   val.py                        # Validate script
│   val_ddp.py                    # DistributedDataParralel validating (Multi GPUs)
```




<details>
<summary>

## Table of Contents <a name="t0"></a>

</summary>

<ul>

<li><a href="#project-folder-structureee">Project Folder Structureee</a></li>
<li> Exploratory Data Analysis 
   <ol>
       <li> Wrong Annotations </li> 
   </ol> 
</li>
<li> Model Architecture 
   <ol>
       <li> Concept Level Architecture </li> 
       <li> Backbone for Feature Computation </li> 
       <li> Neck for Feature Aggregation </li> 
       <li> Head for Dense Object Detection </li> 
   </ol> 
</li>
<li> Ground-Truth Generation 
   <ol>
       <li> Bounding Box Offsets </li> 
       <li> Centerness Score </li> 
       <li> Objectness and Object Class </li>
   </ol> 
</li>
<li> Training 
   <ol>
       <li> Loss Functions </li> 
       <li> Optimization method </li>
   </ol> 
</li>
<li> Performance Evaluation </li>
<li> Video Inference  </li>
   <ol>
       <li> BDD Dataset </li> 
       <li> KITTI Dataset </li>
   </ol>
<li> Conclusion  </li>
<li> Reference  </li>

</ul>
</details>

<br>


## 1. Project Folder Structureee <a name="#project_folder_structureee"></a>
[Back to TOC](#t0)

<br>

## 2. Exploratory Data Analysis <a name="t2"></a>
[Back to TOC](#t0)

<br>

## 3. Model Architecture <a name="t3"></a>
[Back to TOC](#t0)

<br>

## 4. Ground-Truth Generation <a name="t4"></a>
[Back to TOC](#t0)

<br>

## 5. Training <a name="t5"></a>
[Back to TOC](#t0)

<br>

## 6. Performance Evaluation <a name="t6"></a>
[Back to TOC](#t0)

<br>

## 7. Video Inference <a name="t7"></a>
[Back to TOC](#t0)

<br>

## 8. Conclusion <a name="t8"></a>
[Back to TOC](#t0)

<br>

## 9. Reference <a name="t9"></a>
[Back to TOC](#t0)


### Project Structure
```bash
HybridNets
│   backbone.py                   # Model configuration
|   export.py                     # UPDATED 10/2022: onnx weight with accompanying .npy anchors
│   hubconf.py                    # Pytorch Hub entrypoint
│   hybridnets_test.py            # Image inference
│   hybridnets_test_videos.py     # Video inference
│   train.py                      # Train script
│   train_ddp.py                  # DistributedDataParallel training (Multi GPUs)
│   val.py                        # Validate script
│   val_ddp.py                    # DistributedDataParralel validating (Multi GPUs)
│
├───encoders                      # https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/encoders
│       ...
│
├───hybridnets
│       autoanchor.py             # Generate new anchors by k-means
│       dataset.py                # BDD100K dataset
│       loss.py                   # Focal, tversky (dice)
│       model.py                  # Model blocks
│
├───projects
│       bdd100k.yml               # Project configuration
│
├───ros                           # C++ ROS Package for path planning
│       ...
│
└───utils
    |   constants.py
    │   plot.py                   # Draw bounding box
    │   smp_metrics.py            # https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/metrics/functional.py
    │   utils.py                  # Various helper functions (preprocess, postprocess, eval...)
```



<br><br><br><br><br><br><br>

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/kitti_video_infer1.gif)

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/kitti_video_infer2.gif)

![](https://github.com/UditBhaskar19/ANCHOR_FREE_OBJECT_DETECTOR_FOR_CAMERA/blob/main/AnchorFree2DObjectDetection/_readme_artifacts/kitti_video_infer3.gif)

**Detection from Video Clip.**

<br>















# Static Environment Representation from Radar Measurements
[Detailed Design Document](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P1_static_environment_representation/1_radar_static_environment_representation.pdf) <br>
[Python Code](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P1_static_environment_representation/python) <br>
[Result Videos](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/tree/main/P1_static_environment_representation/result_gifs)




## Introduction
**Static environment modelling** is a key component of autonomous navigation. Unfortunately due to various **Radar** specific phenomologies like **clutter**, **missed-detection** and **sparsity** of the point cloud, the raw radar point cloud cannot be used like a lidar point cloud. So in this project the radar data is first upsampled by random sampling. After which the upsampled data is represented in the form of a Regular Grid. **The Grid is defined in the vehicle frame**. Similar to occupancy grid mapping, a log-odds update scheme with a degrading factor is applied for each of the valid grid cells. Here the valid grid cells are those cells whose log-odds value is above a certain threshold. Each of the valid grid cells are characterized by sample position and log-odd value **$(x_m, y_m, l_m)$**. It turns out that this scheme results in low log-odds value for false / clutter detections, hence those can be filtered out by thresholding the log-odds. Finally we show some applications of this modelled environment, which are free-space and road boundary points using basic methods. More sophisticated methods for these applications can be designed which will be a part of a different project.  




## Table of Contents <a name="t0"></a>

   - [Sensor Setup and Layout](#t1)
   - [Inputs Considered and Required Outputs](#t2)
   - [Radar Scan Visualization in Ego Vehicle frame](#t3)
   - [High Level Design](#t4)
   - [Sequence Diagram](#t5)
   - [Module Architecture](#t6)
   - [Grid Fusion](#t7)
   - [Visualization](#t8)

<br>




### 1. Sensor Setup and Layout <a name="t1"></a>
In this project [RadarScenes](https://radar-scenes.com/) dataset is used for validating and generating results. The sensors are not synchronized and the sensor layout doesnot have a full 360&deg; coverage. Nonetheless the dataset is considered here because it is one of the few datasets publickly available that has raw radar point cloud measurements.
<br><br>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/0_sensor_setups.PNG)

<br>

[Back to TOC](#t0)
<br>




### 2. Inputs Considered and Required Outputs <a name="t2"></a>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/1_inputs_outputs.PNG)

<br>

[Back to TOC](#t0)
<br>




### 3. Radar Scan Visualization in Ego Vehicle frame <a name="t3"></a>
The below animation is a brief sequence of radar frames. It can be observed that most of the range-rate is pointed along the line joining the radar and the measurement location (radial axis) . Most of these arrows corrospond to the stationary measurements. The arrows that appears to be of drastically different size corrosponds to measurements from dynamic objects. In this project we use the stationary measuremnets. A method for selecting stationary measurements has been discussed in this [repo](https://github.com/UditBhaskar19/EGO_MOTION_ESTIMATION/tree/main/2_egomotion_radar_polar)

[Animation for longer sequence of radar frames](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/all_radar_meas_long_seq.gif)
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/all_radar_meas_short_seq.gif)

<br>

[Back to TOC](#t0)
<br>




### 4. High Level Design <a name="t4"></a>
   - **Radar $i$ Static Environment Grid Estimation $( i={1,2,3,4} )$** <a name="t41"></a> : A list of valid grid cells are estimated locally corrosponding to each of the radars. Depending on the sensor internal and mounting parameters, a part of the environment might be detected more accurately by one sensor, than the other. In such cases it was found that estimating the cell states locally for each of the radars, and then fusing them centrally gives a more consistent result.<br>
   - **Temporal Allignment** : Since each of the radar has its own grid state estimator, and the radars are operating asynchronously, before grid fusion step we have to represent the cell states of all the radars in the same reference frame. This allignment is achieved in the Temporal Allignment block where we do ego-motion compensation for all the cell states. This step can be optional if the sensors are synchronized. <br>
   - **Grid Fusion** : Finally we combine the local grid state estimates into a single grid <br><br>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/4_architecture.PNG)

<br>

[Back to TOC](#t0)
<br>




### 5. Sequence Diagram <a name="t5"></a>
The below diagram explains the temporal sequence of the grid cell state estimation and fusion.<br>
![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/4_seq_diag.PNG)

[Back to TOC](#t0)
<br>




### 6. Module Architecture <a name="t6"></a>
The components in each of the Radar $i$ [Static Environment Grid Estimation](#t41) $( i={1,2,3,4} )$ block is as follows

   - **Stationary Measurement Identification** : The stationary measurements are identified. First the predicted range-rate for stationarity case at each measurement (x,y) location is computed. If the measurement range-rate and the predicted range-rate is 'close' within a certain margin, then that measurement is considered for further processing. Vehicle odometry is utilized for computing the predicted range-rate. <br>

   - **Clutter Removal by RANSAC** : After an preliminary selection of the stationary measurements, Random Sample Consensus (RANSAC) is used to remove clutter measurements. <br>

   - **Convert Measurement from polar to cartesian** : The selected measurements are converted from polar to cartesian coordinates. <br>

   - **Coordinate Transformation Sensor frame to Vehicle Frame** : Here the measurements are coordinate transformed from sensor frame to vehicle frame. <br>

   - **Compute Measurement Grid** : The measurements are first upsampled by random sampling, the probability (weight) and the corrosponding log-odds is computed for each of the samples. Samples with unique cell IDs are selected. If multiple samples have the same cell ID, the sample that has the largest weight is selected. The sample position and log-odds $(x_i, y_i, l_i)$ is passed as the output. Below are the key steps written formally for sampling and weight computation. Let $(x_k, y_k)$ be a measurement.<br>
   
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **For each measurement generate samples**

   $$
   \begin{pmatrix}
      x_1 \\ 
      y_1
   \end{pmatrix},
   \begin{pmatrix}
      x_2 \\ 
      y_2
   \end{pmatrix} ... 
   \begin{pmatrix}
      x_n \\ 
      y_n
   \end{pmatrix} \sim Normal \ ( 
      \begin{pmatrix}
      x_k \\ 
      y_k
      \end{pmatrix},
      \Sigma_k )
   $$                      
   
   <br>

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Compute probability (weight) for each samples**

   $$
   dist = 
   \begin{pmatrix} 
   x_j - x_k \\ 
   y_j - y_k 
   \end{pmatrix}
   $$

   $$p_{jk} = exp(  -\dfrac{dist^T \Sigma_k^{-1} dist}{2}   )$$ <br>

   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Compute log-odds**

   $$p_{jk} = 0.5 + 0.5 * p_{jk}$$

   $$l_{jk} = log_e( \dfrac{ p_{jk} }{ 1 - p_{jk} } )$$


   - **Predict Grid States** : Before we can do grid cell state update, the grid cell state is predicted from the previous time $(t-1)$ to the current time $(t)$. Ego vehicle localization information at time $(t-1)$ & $(t)$ is utilized for cell state prediction. This prediction step ensures that the measurements at time $(t)$ and the previous cell states at time $(t-1)$ are in same ego vehicle frame at current time $(t)$. The ego vehicle localiztion info is w.r.t some arbitrary origin. The prediction equations for each grid cell $i$ are listed below.

   $$
   T_{prev} =
   \begin{pmatrix}
   cos(&theta;_{t-1}^{loc}) &  -sin(&theta;_{t-1}^{loc})   &   px_{t-1}^{loc} \\
   sin(&theta;_{t-1}^{loc}) &   cos(&theta;_{t-1}^{loc})   &   py_{t-1}^{loc} \\
   0 & 0 & 1
   \end{pmatrix}
   $$

   $$
   T_{curr} =
   \begin{pmatrix}
   cos(&theta;_{t}^{loc}) &  -sin(&theta;_{t}^{loc})   &   px_{t}^{loc} \\
   sin(&theta;_{t}^{loc}) &  cos(&theta;_{t}^{loc})   &   py_{t}^{loc} \\
   0 & 0 & 1
   \end{pmatrix}
   $$

   $$
   T = T_{curr}^{-1}T_{prev} = 
   \begin{pmatrix}
   R_{2x2} &  t_{2x1} \\
   O_{1x2} & 1
   \end{pmatrix}
   $$


   $$
   \begin{pmatrix}
   x_{pred}^i \\ 
   y_{pred}^i
   \end{pmatrix} = 
   R_{2x2} * 
   \begin{pmatrix} x_{prev}^i \\ 
   y_{prev}^i \end{pmatrix} + t_{2x1}
   $$

   <br>

   - **Update Grid State** : The grid cell measurements and the predicted grid cell states are gated and updated. Since the grid is rectangular with uniformly sized cells. Each grid cell can be indexed like an image leading to efficient gating and state updates. different rules for state update is applied depending on whether the cells are **gated**, **not gated**, **inside active sensor FOV** or **outside active sensor FOV**.

   ![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/update_rules.PNG)


   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <ins>The state update equations are listed below</ins>. <br><br><br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Un-Gated Measurement Grid Cell IDs**
            $$x_{upd}^i = x_{meas}^i$$
            $$y_{upd}^i = y_{meas}^i$$
            $$l_{upd}^i = a_{0} * l_{meas}^i$$  <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Gated Grid Cell IDs**
            $$x_{upd}^i = w_x * x_{meas}^i + ( 1 - w_x ) * x_{pred}^i$$
            $$y_{upd}^i = w_y * y_{meas}^i + ( 1 - w_y ) * y_{pred}^i$$
            $$l_{upd}^i = a_1 * l_{pred}^i + l_{meas}^i$$
            $$0 <= w_x <= 1$$
            $$0 <= w_y <= 1$$ <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Un-Gated Predicted Grid Cell within active sensor FOV**
            $$x_{upd}^i = x_{pred}^i$$
            $$y_{upd}^i = y_{pred}^i$$
            $$l_{upd}^i = a_2 * l_{pred}^i$$ <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Un-Gated Predicted Grid Cell outside active sensor FOV**
            $$x_{upd}^i = x_{pred}^i$$
            $$y_{upd}^i = y_{pred}^i$$
            $$l_{upd}^i = a_3 * l_{pred}^i$$
<br>

![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/readme_artifacts/4_module_arc.PNG)
<br>

[Back to TOC](#t0)
<br>




### 7. Grid Fusion <a name="t7"></a> 
Finally the Local Grid Cell state estimates are fused. The cell $(x_i, y_i)$ coordinates are combined by weighted averaging. The log-odds are summed. Only the valid local grid cell states are combined. The invalid cell state has 0 log-odd value. The weight for the invalid states is considered to be 0.

$$
x_{fus}^i = \sum_{s=1}^{4} w_{radar_s} * x_{radar_s}^i
$$

$$
y_{fus}^i = \sum_{s=1}^{4} w_{radar_s} * y_{radar_s}^i
$$

$$
l_{fus}^i = \sum_{s=1}^{4} l_{radar_s}^i
$$

$$
\sum_{s=1}^{4} w_{radar_s} = 1
$$

<br>

[Back to TOC](#t0)
<br>



### 8. Visualization <a name="t8"></a>
In this section we show videos of the results.

   1. **Radar Dense Point Cloud by Grid based measurement filtering** <br>
   [Longer Animation Sequence Link](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/result_gifs/radar_dense_point_cloud_long_seq.gif)
   ![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/result_gifs/radar_dense_point_cloud_short_seq.gif)

   2. **Free Space Computation** <br>
   [Longer Animation Sequence Link](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/result_gifs/free_space_computation_radar_long_seq.gif)
   ![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/result_gifs/free_space_computation_radar_short_seq.gif)

   3. **Road Boundary Point Extraction** <br>
   [Longer Animation Sequence Link](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/result_gifs/road_boundary_long_seq.gif)
   ![](https://github.com/UditBhaskar19/ENVIRONMENT_REPRESENTATION_USING_RADAR/blob/main/P1_static_environment_representation/result_gifs/road_boundary_short_seq.gif)


[Back to TOC](#t0)