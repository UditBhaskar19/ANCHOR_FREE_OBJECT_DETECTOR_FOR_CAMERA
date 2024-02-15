# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Classes that encapsulates the datasets used during training         
# ---------------------------------------------------------------------------------------------------------------------
from modules.dataset_utils.bdd_dataset_utils.remapped_bdd_utils import load_ground_truths
from modules.dataset_utils.bdd_dataset_and_dataloader import BerkeleyDeepDriveDataset
from modules.dataset_utils.kitti_objdetect_dataloader import KittiTrackingDataset_2DObjectDetection
from modules.dataset_utils.kitti_dataset_utils.kitti_remap_utils import load_all_sequence_groundtruths_json
from modules.first_stage.get_parameters import bdd_parameters, kitti_parameters
import numpy as np
from torch.utils.data import DataLoader

# =============================================================================================================================
def infinite_loader(loader):
    """ Get an infinite stream of batches from a data loader """
    while True:
        yield from loader

# =============================================================================================================================
class BASE_CLASS_dataset:
    def __init__(
        self, 
        batch_size, 
        shuffle_dataset: bool = False):

        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.dataset_train = None
        self.dataset_val = None
        self.train_loader = None
        self.val_loader = None
        self.param_obj = None

    def set_dataloader(self):
        train_args = dict(batch_size=self.batch_size, shuffle=self.shuffle_dataset, collate_fn=self.dataset_train.collate_fn)
        val_args = dict(batch_size=self.batch_size, shuffle=self.shuffle_dataset, collate_fn=self.dataset_val.collate_fn)
        train_loader = DataLoader(self.dataset_train, **train_args)
        val_loader = DataLoader(self.dataset_val, **val_args)

        self.train_loader = infinite_loader(train_loader)
        self.val_loader = val_loader

    def get_training_sample(self):
        images, labels = next(self.train_loader)
        return images, labels, self.param_obj
    
    def disable_augmentation(self):
        self.dataset_train.set_prob_augment(0.0)
        train_args = dict(batch_size=self.batch_size, shuffle=self.shuffle_dataset, collate_fn=self.dataset_train.collate_fn)
        train_loader = DataLoader(self.dataset_train, **train_args)
        self.train_loader = infinite_loader(train_loader)

# ==============================================> BDD DATASET & DATALOADER <===================================================
class BDD_dataset(BASE_CLASS_dataset):
    def __init__(
        self,
        label_rootdir: str,
        dataset_rootdir: str,
        batch_size: int,
        num_samples_val: int, 
        bdd_param_obj: bdd_parameters,
        device: str,
        shuffle_dataset: bool = False,
        perform_augmentation_train: bool = False,
        augmentation_prob_train: float = 0.0):

        super().__init__(batch_size, shuffle_dataset)
        self.param_obj = bdd_param_obj

        # get the gt labels for training
        gt_labels_train = load_ground_truths(
            label_rootdir,
            self.param_obj.sel_train_labels_file, 
            dataset_rootdir,
            self.param_obj.train_images_dir, 
            verbose = False)

        # init train data-loader
        image_new_shape = (
            self.param_obj.IMG_D, 
            self.param_obj.IMG_RESIZED_H, 
            self.param_obj.IMG_RESIZED_W)
        
        self.dataset_train = BerkeleyDeepDriveDataset(
            gt_labels_train, 
            image_new_shape,
            device, subset = -1,
            augment = perform_augmentation_train,
            prob_augment = augmentation_prob_train)

        # get the gt labels for validation
        gt_labels_val = load_ground_truths(
            label_rootdir,
            self.param_obj.sel_val_labels_file, 
            dataset_rootdir,
            self.param_obj.val_images_dir, 
            verbose = False)

        # init val data-loader
        self.dataset_val = BerkeleyDeepDriveDataset(
            gt_labels_val, 
            image_new_shape,
            device, num_samples_val,
            augment = False)
        
        self.set_dataloader()

# ==============================================> KITTI DATASET & DATALOADER <=================================================
class KITTI_dataset(BASE_CLASS_dataset):
    def __init__(
        self,
        label_rootdir: str,
        dataset_rootdir: str,
        batch_size: int,
        num_samples_val: int, 
        kitti_param_obj: kitti_parameters,
        device: str,
        shuffle_dataset: bool = False,
        perform_augmentation_train: bool = False,
        augmentation_prob_train: float = 0.0):

        super().__init__(batch_size, shuffle_dataset)
        self.param_obj = kitti_param_obj

        # get the gt labels for training
        gt_labels_train, _, _ = load_all_sequence_groundtruths_json(
            self.param_obj.kitti_train_sequences_folders, 
            self.param_obj.kitti_remapped_label_file_path, 
            label_rootdir, 
            dataset_rootdir)
        
        # init train data-loader
        image_new_shape = (
            self.param_obj.IMG_D, 
            self.param_obj.IMG_RESIZED_H, 
            self.param_obj.IMG_RESIZED_W)
        
        self.dataset_train = KittiTrackingDataset_2DObjectDetection(
            gt_labels = gt_labels_train, 
            image_new_shape = image_new_shape,
            device = device, 
            subset = -1,
            shuffle_dataset = shuffle_dataset,
            augment = perform_augmentation_train,
            prob_augment = augmentation_prob_train)
        
        # get the gt labels for validation
        gt_labels_val, _, _ = load_all_sequence_groundtruths_json(
            self.param_obj.kitti_val_sequences_folders, 
            self.param_obj.kitti_remapped_label_file_path, 
            label_rootdir, 
            dataset_rootdir)

        # init val data-loader
        self.dataset_val = KittiTrackingDataset_2DObjectDetection(
            gt_labels = gt_labels_val, 
            image_new_shape = image_new_shape,
            device = device, 
            subset = num_samples_val,
            shuffle_dataset = True,
            augment = False)
        
        self.set_dataloader()

# =============================================> SELECT DATASET WITH EVERY ITERATION <=========================================
class DATSET_Selector:
    def __init__(
        self, 
        bdd_dataset_obj: BDD_dataset, 
        kitti_dataset_obj: KITTI_dataset,
        max_training_iter: int,
        bdd_dataset_weight: float):
        
        self.max_training_iterations = max_training_iter
        self.bdd_dataset = bdd_dataset_obj
        self.kitti_dataset = kitti_dataset_obj
        self.print_dataset_info = False

        # 1 -> choose from bdd_dataset, 2 -> choose from kitti dataset
        uniform_random_numbers = np.random.uniform(low=0.0, high=1.0, size=max_training_iter)
        self.dataset_idx =  np.where(uniform_random_numbers <= bdd_dataset_weight, 1, 2)

    def get_training_sample(self, iter):
        if self.dataset_idx[iter] == 1:
            if self.print_dataset_info: print('BDD Dataset')
            images, labels, param_obj = self.bdd_dataset.get_training_sample()
        elif self.dataset_idx[iter] == 2:
            if self.print_dataset_info: print('KITTI Dataset')
            images, labels, param_obj = self.kitti_dataset.get_training_sample()
        else: raise ValueError('wrong option.')
        return images, labels, param_obj
    
    def disable_augmentation(self):
        self.bdd_dataset.disable_augmentation()
        self.kitti_dataset.disable_augmentation()

    def disable_augmentation_kitti(self):
        self.kitti_dataset.disable_augmentation()

    def disable_augmentation_bdd(self):
        self.bdd_dataset.disable_augmentation()
