import numpy as np
import pickle
import os

import torch.utils.data
from data.base_dataset import collate_fn
from data.task_grasp_loader import collate_fn as taskgrasp_collate_fn
import threading
import data.data_utils as d_utils
from torchvision import transforms
from data.data_specification import TASKS




def CreateDataset(opt):
    """loads dataset class"""

    if opt.dataset == 0:

        if opt.arch == 'vae' or opt.arch == 'gan':
            from data.grasp_sampling_data import GraspSamplingData
            dataset = GraspSamplingData(opt)
        else:
            from data.grasp_evaluator_data import GraspEvaluatorData
            dataset = GraspEvaluatorData(opt)
    else:

        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudGraspToTensor(),
                d_utils.PointcloudGraspScale(),
                d_utils.PointcloudGraspRotate(axis=np.array([1.0, 0.0, 0.0])),
                d_utils.PointcloudGraspRotatePerturbation(),
                d_utils.PointcloudGraspRotate(axis=np.array([0.0, 1.0, 0.0])),
                d_utils.PointcloudGraspRotatePerturbation(),
                d_utils.PointcloudGraspRotate(axis=np.array([0.0, 0.0, 1.0])),
                d_utils.PointcloudGraspRotatePerturbation(),
                d_utils.PointcloudGraspTranslate(),
                d_utils.PointcloudGraspJitter(),
                d_utils.PointcloudGraspRandomInputDropout(),
            ]
        )

        evaluator_transforms = transforms.Compose(
            [
                d_utils.PointcloudGraspToTensor(),
                # d_utils.PointcloudGraspScale(),
                # d_utils.PointcloudGraspRotate(axis=np.array([1.0, 0.0, 0.0])),
                # d_utils.PointcloudGraspRotatePerturbation(),
                # d_utils.PointcloudGraspRotate(axis=np.array([0.0, 1.0, 0.0])),
                # d_utils.PointcloudGraspRotatePerturbation(),
                # d_utils.PointcloudGraspRotate(axis=np.array([0.0, 0.0, 1.0])),
                # d_utils.PointcloudGraspRotatePerturbation(),
                # d_utils.PointcloudGraspTranslate(),
                # d_utils.PointcloudGraspJitter(),
                # d_utils.PointcloudGraspRandomInputDropout(),
            ]
        )
            
        _, _, _,name2wn = pickle.load(open(os.path.join(opt.dataset_root_folder, 'misc.pkl'),'rb'))

        with open('object_class.txt', 'w') as f:
            for item, key in name2wn.items():
                f.write("%s : %s\n" % (item, key))

        class_list = pickle.load(open(os.path.join(opt.dataset_root_folder, 'class_list.pkl'),'rb')) 

        with open('class_list.txt', 'w') as f:
            for item in class_list:
                f.write("%s\n" % item)

        with open('task.txt', 'w') as f:
            for item in TASKS:
                f.write("%s\n" % item)

        if opt.arch == 'vae' or opt.arch == 'gan':
            from data.task_grasp_loader import TaskGraspLoader

            dataset = TaskGraspLoader(
                opt,
                use_task1_grasps=True,
                tasks=TASKS, 
                class_list=class_list,
                transforms= train_transforms,
                map_obj2class=name2wn)
        else:
            from data.taskgrasp_evaluator import TaskGraspEvaluatorData

            dataset = TaskGraspEvaluatorData(
                opt,
                use_task1_grasps=True,
                tasks=TASKS, 
                class_list=class_list,
                transforms= evaluator_transforms,
                map_obj2class=name2wn)

    # if opt.arch == 'vae' or opt.arch == 'gan':
    #     if opt.dataset == 0:
    #         from data.grasp_sampling_data import GraspSamplingData
    #         dataset = GraspSamplingData(opt)
    #     else:
    #         from data.task_grasp_loader import TaskGraspLoader

    #         train_transforms = transforms.Compose(
    #             [
    #                 d_utils.PointcloudGraspToTensor(),
    #                 d_utils.PointcloudGraspScale(),
    #                 d_utils.PointcloudGraspRotate(axis=np.array([1.0, 0.0, 0.0])),
    #                 d_utils.PointcloudGraspRotatePerturbation(),
    #                 d_utils.PointcloudGraspRotate(axis=np.array([0.0, 1.0, 0.0])),
    #                 d_utils.PointcloudGraspRotatePerturbation(),
    #                 d_utils.PointcloudGraspRotate(axis=np.array([0.0, 0.0, 1.0])),
    #                 d_utils.PointcloudGraspRotatePerturbation(),
    #                 d_utils.PointcloudGraspTranslate(),
    #                 d_utils.PointcloudGraspJitter(),
    #                 d_utils.PointcloudGraspRandomInputDropout(),
    #             ]
    #         )
            
    #         _, _, _,name2wn = pickle.load(open(os.path.join(opt.dataset_root_folder, 'misc.pkl'),'rb'))

    #         with open('object_class.txt', 'w') as f:
    #             for item, key in name2wn.items():
    #                 f.write("%s : %s\n" % (item, key))

    #         class_list = pickle.load(open(os.path.join(opt.dataset_root_folder, 'class_list.pkl'),'rb')) 

    #         with open('class_list.txt', 'w') as f:
    #             for item in class_list:
    #                 f.write("%s\n" % item)

    #         with open('task.txt', 'w') as f:
    #             for item in TASKS:
    #                 f.write("%s\n" % item)


    #         dataset = TaskGraspLoader(
    #             opt,
    #             use_task1_grasps=True,
    #             tasks=TASKS, 
    #             class_list=class_list,
    #             transforms= train_transforms,
    #             map_obj2class=name2wn)
    # else:
    #     if opt.dataset == 0:
    #         from data.grasp_evaluator_data import GraspEvaluatorData
    #         dataset = GraspEvaluatorData(opt)
    #     else:
    #         from data.taskgrasp_evaluator import TaskGraspEvaluatorData

    #         train_transforms = transforms.Compose(
    #             [
    #                 d_utils.PointcloudGraspToTensor(),
    #                 d_utils.PointcloudGraspScale(),
    #                 d_utils.PointcloudGraspRotate(axis=np.array([1.0, 0.0, 0.0])),
    #                 d_utils.PointcloudGraspRotatePerturbation(),
    #                 d_utils.PointcloudGraspRotate(axis=np.array([0.0, 1.0, 0.0])),
    #                 d_utils.PointcloudGraspRotatePerturbation(),
    #                 d_utils.PointcloudGraspRotate(axis=np.array([0.0, 0.0, 1.0])),
    #                 d_utils.PointcloudGraspRotatePerturbation(),
    #                 d_utils.PointcloudGraspTranslate(),
    #                 d_utils.PointcloudGraspJitter(),
    #                 d_utils.PointcloudGraspRandomInputDropout(),
    #             ]
    #         )

    #         dataset = TaskGraspEvaluatorData(
    #             opt,
    #             use_task1_grasps=True,
    #             tasks=TASKS, 
    #             class_list=class_list,
    #             transforms= train_transforms,
    #             map_obj2class=name2wn)
    return dataset


class DataLoader:
    """multi-threaded data loading"""
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.num_objects_per_batch,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn= collate_fn)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
