# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : model optimization function
# --------------------------------------------------------------------------------------------------------------
import numpy as np
import time, os, torch
from torch.nn.utils import clip_grad_norm_
from config_neuralnet_stage1 import (
    model_weights_main_dir, weights_name,
    CLS_LOSS_WT, OBJ_LOSS_WT, CTR_LOSS_WT, BOX_LOSS_WT)

# --------------------------------------------------------------------------------------------------------------
def create_weight_dir():
    curr_weight_dir = str(round(time.time() * 1000))
    if not os.path.exists(model_weights_main_dir): os.mkdir(model_weights_main_dir)
    weight_dir = os.path.join(model_weights_main_dir, curr_weight_dir)
    if not os.path.exists(weight_dir): os.mkdir(weight_dir)
    return weight_dir

def save_model_weights(detector, weight_dir):
    weight_path = os.path.join(weight_dir, weights_name)
    torch.save(detector.state_dict(), weight_path)

# --------------------------------------------------------------------------------------------------------------
def train_model(
    detector,
    optimizer,
    lr_scheduler,
    dataloader,
    tb_writer,
    max_iters,
    log_period: int = 20,
    val_period: int = 1000,
    iter_start_offset: int = 0):

    loss_tracker = LossTracker()        # track losses for tensorboard visualization
    weight_dir = create_weight_dir()    # create directory to save model weights 

    for iter_train in range(iter_start_offset, max_iters):
        # ==============================================================================================
        # change augmentation prob according to 'iter_train'
        # if iter_train == 0.95*max_iters: 
        #     dataloader.disable_augmentation()

        # compute loss and perform backpropagation
        detector.train()
        images, labels, param_obj = dataloader.get_training_sample(iter_train)       
        gt_boxes = labels['bbox_batch']            
        gt_class = labels['obj_class_label']
        
        detector.reinit_const_parameters(param_obj)
        losses = detector(images, gt_boxes, gt_class)
        total_loss \
            = CLS_LOSS_WT * losses['loss_cls'] + \
              BOX_LOSS_WT * losses['loss_box'] + \
              CTR_LOSS_WT * losses['loss_ctr'] + \
              OBJ_LOSS_WT * losses['loss_obj']
        optimizer.zero_grad()
        total_loss.backward()
        # max_grad_norm = 1.0  # Set the maximum gradient norm value
        # clip_grad_norm_(detector.parameters(), max_grad_norm, error_if_nonfinite=False)
        optimizer.step()
        lr_scheduler.step()    

        # ==============================================================================================      
        # validate model, print loss on console, save the model weights
        if total_loss > 0:
            loss_tracker.append_training_loss_for_tb(total_loss, losses)

        if iter_train % log_period == 0:   # Print losses periodically on the console
            loss_str = f"[Iter {iter_train}][loss: {total_loss:.5f}]"
            for key, value in losses.items():
                loss_str += f"[{key}: {value:.5f}]"
            print(loss_str)
            if total_loss > 0:
                loss_tracker.loss_history.append(total_loss.item())

        if iter_train % val_period == 0:   # write to tb,  and run validation 
            print('-'*100)
            print('saving model')
            save_model_weights(detector, weight_dir)

            # bdd dataset validation
            print('performing validation BDD Dataset')
            print('-'*100)
            detector.reinit_const_parameters(dataloader.bdd_dataset.param_obj)
            with torch.no_grad():
                for iter_val, (images, labels) in enumerate(dataloader.bdd_dataset.val_loader):
                    detector.eval()       
                    gt_boxes = labels['bbox_batch']            
                    gt_class = labels['obj_class_label']
                    losses = detector(images, gt_boxes, gt_class)
                    total_loss = sum(losses.values())
                    if total_loss > 0:
                        loss_tracker.append_bdd_validation_loss_for_tb(total_loss, losses)     

                    if iter_val % log_period == 0:   # Print losses periodically on the console
                        loss_str = f"[Iter {iter_val}][loss: {total_loss:.5f}]"
                        for key, value in losses.items():
                            loss_str += f"[{key}: {value:.5f}]"
                        print(loss_str)

            # kitti dataset validation
            print('performing validation KITTI Dataset')
            print('-'*100)
            detector.reinit_const_parameters(dataloader.kitti_dataset.param_obj)
            with torch.no_grad():
                for iter_val, (images, labels) in enumerate(dataloader.kitti_dataset.val_loader):
                    detector.eval()       
                    gt_boxes = labels['bbox_batch']            
                    gt_class = labels['obj_class_label']
                    losses = detector(images, gt_boxes, gt_class)
                    total_loss = sum(losses.values())
                    if total_loss > 0:
                        loss_tracker.append_kitti_validation_loss_for_tb(total_loss, losses)     

                    if iter_val % log_period == 0:   # Print losses periodically on the console
                        loss_str = f"[Iter {iter_val}][loss: {total_loss:.5f}]"
                        for key, value in losses.items():
                            loss_str += f"[{key}: {value:.5f}]"
                        print(loss_str)

            # ==============================================================================================
            # write the loss to tensor board
            total_train_loss_tb, cls_train_loss_tb, box_train_loss_tb, \
            ctr_train_loss_tb, obj_train_loss_tb = loss_tracker.compute_avg_training_loss()

            kitti_val_loss_tb, kitti_cls_val_loss_tb, kitti_box_val_loss_tb, \
            kitti_ctr_val_loss_tb, kitti_obj_val_loss_tb = loss_tracker.compute_avg_val_loss_kitti()

            bdd_val_loss_tb, bdd_cls_val_loss_tb, bdd_box_val_loss_tb, \
            bdd_ctr_val_loss_tb, bdd_obj_val_loss_tb = loss_tracker.compute_avg_val_loss_bdd()

            print('train_losses_tb    : ', total_train_loss_tb)          
            print('bdd_val_losses_tb  : ', bdd_val_loss_tb)  
            print('kitti_val_losses_tb: ', kitti_val_loss_tb)             
            print("end of validation : Resuming Training")
            print("-"*100)

            tb_writer.add_scalars('Total_Loss', 
                                  {'train':total_train_loss_tb, 
                                   'kitti_val':kitti_val_loss_tb,
                                   'bdd_val':bdd_val_loss_tb,}, iter_train)

            tb_writer.add_scalars('Classification_Loss', 
                                  {'train':cls_train_loss_tb, 
                                   'kitti_val':kitti_cls_val_loss_tb,
                                   'bdd_val':bdd_cls_val_loss_tb}, iter_train)

            tb_writer.add_scalars('Box_Loss', 
                                  {'train':box_train_loss_tb, 
                                   'kitti_val':kitti_box_val_loss_tb,
                                   'bdd_val':bdd_box_val_loss_tb}, iter_train)

            tb_writer.add_scalars('Centerness_Loss', 
                                  {'train':ctr_train_loss_tb, 
                                   'kitti_val':kitti_ctr_val_loss_tb,
                                   'bdd_val':bdd_ctr_val_loss_tb}, iter_train)

            tb_writer.add_scalars('Objectness_Loss', 
                                  {'train':obj_train_loss_tb, 
                                   'kitti_val':kitti_obj_val_loss_tb,
                                   'bdd_val':bdd_obj_val_loss_tb}, iter_train)
            
            # lr_scheduler.step(train_losses_tb)   update the 'ReduceLROnPlateau' scheduler if implemented

            # reset train_losses_tb
            loss_tracker.reset_training_loss_for_tb()
            loss_tracker.reset_bdd_validation_loss_for_tb()
            loss_tracker.reset_kitti_validation_loss_for_tb()

# --------------------------------------------------------------------------------------------------------------
class LossTracker:
    def __init__(self):
        # training losses
        self.loss_history = []      # Keep track of training loss for plotting.
        self.train_losses_tb = []   # train_losses for tensor board visualization
        self.cls_train_losses_tb = []  # classification train_losses for tensor board visualization
        self.box_train_losses_tb = []  # box regression train_losses for tensor board visualization
        self.ctr_train_losses_tb = []  # centerness train_losses for tensor board visualization
        self.obj_train_losses_tb = []  # objectness train_losses for tensor board visualization

        # kitti val losses
        self.kitti_val_losses_tb = []
        self.kitti_cls_val_losses_tb = []  
        self.kitti_box_val_losses_tb = []  
        self.kitti_ctr_val_losses_tb = []   
        self.kitti_obj_val_losses_tb = []   

        # bdd val losses
        self.bdd_val_losses_tb = []
        self.bdd_cls_val_losses_tb = []  
        self.bdd_box_val_losses_tb = []  
        self.bdd_ctr_val_losses_tb = []   
        self.bdd_obj_val_losses_tb = []

    def append_training_loss_for_tb(self, total_loss, losses):
        self.train_losses_tb.append(total_loss.item()) 
        self.cls_train_losses_tb.append(losses['loss_cls'].item())
        self.box_train_losses_tb.append(losses['loss_box'].item())
        self.ctr_train_losses_tb.append(losses['loss_ctr'].item())
        self.obj_train_losses_tb.append(losses['loss_obj'].item())

    def append_kitti_validation_loss_for_tb(self, total_loss, losses):
        self.kitti_val_losses_tb.append(total_loss.item()) 
        self.kitti_cls_val_losses_tb.append(losses['loss_cls'].item())
        self.kitti_box_val_losses_tb.append(losses['loss_box'].item())
        self.kitti_ctr_val_losses_tb.append(losses['loss_ctr'].item())
        self.kitti_obj_val_losses_tb.append(losses['loss_obj'].item())

    def append_bdd_validation_loss_for_tb(self, total_loss, losses):
        self.bdd_val_losses_tb.append(total_loss.item()) 
        self.bdd_cls_val_losses_tb.append(losses['loss_cls'].item())
        self.bdd_box_val_losses_tb.append(losses['loss_box'].item())
        self.bdd_ctr_val_losses_tb.append(losses['loss_ctr'].item())
        self.bdd_obj_val_losses_tb.append(losses['loss_obj'].item())

    def reset_training_loss_for_tb(self):
        self.train_losses_tb = []   
        self.cls_train_losses_tb = []  
        self.box_train_losses_tb = []  
        self.ctr_train_losses_tb = []  
        self.obj_train_losses_tb = [] 

    def reset_kitti_validation_loss_for_tb(self):
        self.kitti_val_losses_tb = []
        self.kitti_cls_val_losses_tb = []  
        self.kitti_box_val_losses_tb = []  
        self.kitti_ctr_val_losses_tb = []   
        self.kitti_obj_val_losses_tb = []   

    def reset_bdd_validation_loss_for_tb(self):
        self.bdd_val_losses_tb = []
        self.bdd_cls_val_losses_tb = []  
        self.bdd_box_val_losses_tb = []  
        self.bdd_ctr_val_losses_tb = []   
        self.bdd_obj_val_losses_tb = []

    def compute_avg_training_loss(self):
        total_train_loss_tb = np.mean(np.array(self.train_losses_tb))   
        cls_train_loss_tb = np.mean(np.array(self.cls_train_losses_tb))
        box_train_loss_tb = np.mean(np.array(self.box_train_losses_tb))
        ctr_train_loss_tb = np.mean(np.array(self.ctr_train_losses_tb))
        obj_train_loss_tb = np.mean(np.array(self.obj_train_losses_tb))
        return total_train_loss_tb, cls_train_loss_tb, box_train_loss_tb, ctr_train_loss_tb, obj_train_loss_tb
    
    def compute_avg_val_loss_kitti(self):
        kitti_val_loss_tb = np.mean(np.array(self.kitti_val_losses_tb))   
        kitti_cls_val_loss_tb = np.mean(np.array(self.kitti_cls_val_losses_tb))
        kitti_box_val_loss_tb = np.mean(np.array(self.kitti_box_val_losses_tb))
        kitti_ctr_val_loss_tb = np.mean(np.array(self.kitti_ctr_val_losses_tb))
        kitti_obj_val_loss_tb = np.mean(np.array(self.kitti_obj_val_losses_tb))
        return kitti_val_loss_tb, kitti_cls_val_loss_tb, kitti_box_val_loss_tb, kitti_ctr_val_loss_tb, kitti_obj_val_loss_tb  
    
    def compute_avg_val_loss_bdd(self):
        bdd_val_loss_tb = np.mean(np.array(self.bdd_val_losses_tb))   
        bdd_cls_val_loss_tb = np.mean(np.array(self.bdd_cls_val_losses_tb))
        bdd_box_val_loss_tb = np.mean(np.array(self.bdd_box_val_losses_tb))
        bdd_ctr_val_loss_tb = np.mean(np.array(self.bdd_ctr_val_losses_tb))
        bdd_obj_val_loss_tb = np.mean(np.array(self.bdd_obj_val_losses_tb))
        return bdd_val_loss_tb, bdd_cls_val_loss_tb, bdd_box_val_loss_tb, bdd_ctr_val_loss_tb, bdd_obj_val_loss_tb