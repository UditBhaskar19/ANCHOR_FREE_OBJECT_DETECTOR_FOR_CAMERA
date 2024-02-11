# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Aggregate box offset values from training data and compute statistics       
# ---------------------------------------------------------------------------------------------------------------------
import sys, os
module_rootdir = '.'
dataset_rootdir = '../'
label_rootdir = module_rootdir
sys.path.append(module_rootdir)
from modules.hyperparam.bdd_aggregate_gt_transforms import hyperparam_out_dir as bdd_hyperparam_out_dir
from modules.hyperparam.bdd_aggregate_gt_transforms import aggregate_deltas as bdd_aggregate_deltas
from modules.hyperparam.kitti_aggregate_gt_transforms import hyperparam_out_dir as kitti_hyperparam_out_dir
from modules.hyperparam.kitti_aggregate_gt_transforms import aggregate_deltas as kitti_aggregate_deltas

num_samples = 10000
write_deltas = True
write_deltas_statistics = True
write_label_instance_count = True

# =======================================> BDD DATASET <==============================================
if not os.path.exists(bdd_hyperparam_out_dir): 
    os.makedirs(bdd_hyperparam_out_dir, exist_ok=True)

class_instance_count, \
aggregated_transforms, \
mean, std = bdd_aggregate_deltas(
    num_samples, 
    label_rootdir, 
    dataset_rootdir,
    bdd_hyperparam_out_dir, 
    write_deltas = write_deltas,
    write_deltas_statistics = write_deltas_statistics,
    write_label_instance_count = write_label_instance_count)

print('========> Printing BDD Dataset gt transforms statistic <==========')
print('mean: ', mean)
print('std: ', std)
print(class_instance_count)

# =======================================> KITTI DATASET <==============================================
if not os.path.exists(kitti_hyperparam_out_dir): 
    os.makedirs(kitti_hyperparam_out_dir, exist_ok=True)

class_instance_count, \
aggregated_transforms, \
mean, std = kitti_aggregate_deltas(
    num_samples, 
    label_rootdir, 
    dataset_rootdir,
    kitti_hyperparam_out_dir, 
    write_deltas = write_deltas,
    write_deltas_statistics = write_deltas_statistics,
    write_label_instance_count = write_label_instance_count)

print('========> Printing KITTI Dataset gt transforms statistic <==========')
print('mean: ', mean)
print('std: ', std)
print(class_instance_count)