import os
from cv2 import line
import numpy as np
import tqdm
import time

import torch.utils.data as data
import torch

from utils.splits import get_split_data, parse_line, get_ot_pairs_taskgrasp, get_task1_hits
from utils import utils
import copy



def pc_normalize(pc, grasp, pc_scaling=True):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    grasp[:3, 3] -= centroid

    if pc_scaling:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

        pc = np.concatenate([pc, np.ones([pc.shape[0], 1])], axis=1)
        scale_transform = np.diag([1 / m, 1 / m, 1 / m, 1])
        pc = np.matmul(scale_transform, pc.T).T
        pc = pc[:, :3]
        grasp = np.matmul(scale_transform, grasp)
    return pc, grasp

def collate_fn(batch):
    """ This function overrides defaul batch collate function and aggregates 
    the graph and point clound data across the batch into a single graph tensor """

    pc = torch.stack([torch.as_tensor(_[0]) for _ in batch], dim=0)
    pc_color = torch.stack([torch.as_tensor(_[1]) for _ in batch], dim=0)
    task_id = torch.stack([torch.tensor(_[2]) for _ in batch], dim=0)
    class_id = torch.stack([torch.tensor(_[3]) for _ in batch], dim=0)
    instance_id = torch.stack([torch.tensor(_[4]) for _ in batch], dim=0)
    grasp = torch.stack([torch.tensor(_[5]) for _ in batch], dim=0)
    grasp_pc = torch.stack([torch.as_tensor(_[6]) for _ in batch], dim=0)
    label = torch.stack([torch.tensor(_[7]) for _ in batch], dim=0)
    grasp_id = torch.stack([torch.tensor(_[8]) for _ in batch], dim=0)

    return pc, pc_color, task_id, class_id, instance_id, grasp, grasp_pc, label, grasp_id


class TaskGraspLoader(data.Dataset):
    def __init__(
        self, 
        opt, 
        use_task1_grasps=True,
        tasks=None, 
        class_list=None,
        map_obj2class=None,
        transforms=None):
        super().__init__()

        folder_dir = ""
        self.opt = opt
        self._pc_scaling = opt.pc_scaling
        self._split_mode = opt.split_mode
        self._split_idx = opt.split_idx
        self._split_version = opt.split_version
        self._transforms = transforms
        self._tasks = tasks
        self._num_tasks = len(self._tasks)
        

        task1_results_file = os.path.join(
            opt.dataset_root_folder, 'task1_results.txt')
        assert os.path.exists(task1_results_file)

        self._train = opt.phase
        self._map_obj2class = map_obj2class
        data_dir = os.path.join(opt.dataset_root_folder, "scans")

        data_txt_splits = {
            'test': 'test_split.txt',
            'train': 'train_split.txt',
            'val': 'val_split.txt'}

        if self._train not in data_txt_splits:
            raise ValueError("Unknown split arg {}".format(self._train))

        self._parse_func = parse_line
        lines = get_split_data(
            opt.dataset_root_folder,
            folder_dir,
            self._train,
            self._split_mode,
            self._split_idx,
            self._split_version,
            use_task1_grasps,
            data_txt_splits,
            self._map_obj2class,
            self._parse_func,
            get_ot_pairs_taskgrasp,
            get_task1_hits)

        # lines = lines[:10]
        # if(len(lines) > 10000):self._num_object_classes
        self._data = []
        self._pc = {}
        self._grasps = {}
        self._object_classes = class_list

        self._num_object_classes = len(self._object_classes)

        start = time.time()
        correct_counter = 0

        all_object_instances = []

        self._object_task_pairs_dataset = []
        self._data_labels = []
        self._data_label_counter = {0: 0, 1: 0}

        # if self._train == 'train':
        #     with open('data/taskgrasp_data/splits_final/o/0/val_split.txt', 'w') as f:
        #         for item in lines:
        #             f.write("%s" % item)

        for i in range(len(lines)):
            obj, obj_class, grasp_id, task, label = parse_line(lines[i])
            all_object_instances.append(obj)
            
        self._all_object_instances = list(set(all_object_instances))

        with open('instance.txt', 'w') as f:
            for item in self._all_object_instances:
                f.write("%s\n" % item)

        for obj in tqdm.tqdm(self._all_object_instances):
            
            pc_file = os.path.join(data_dir, obj, "fused_pc_clean.npy")

            if pc_file not in self._pc:
                if not os.path.exists(pc_file):
                    raise ValueError(
                        'Unable to find processed point cloud file {}'.format(pc_file))
                pc = np.load(pc_file)
                self._pc[pc_file] = pc
            
            grasp_set = []

            for grasp_id in range(opt.num_grasps_per_object):
                grasp_file = os.path.join(
                data_dir, obj, "grasps", str(grasp_id), "grasp.npy")
                if grasp_file not in self._grasps:
                    grasp = np.load(grasp_file)
                    grasp_set.append(grasp_file)
                    self._grasps[grasp_file] = grasp
            
            self._data.append((pc_file, grasp_set, obj))
            
        self._len = len(self._data)
        print('Loading files from {} took {}s; overall dataset size {}'.format(
        data_txt_splits[self._train], time.time() - start, self._len))


####################################################
             
        # for i in tqdm.trange(len(lines)):
        #     obj, obj_class, grasp_id, task, label = parse_line(lines[i])
        #     obj_class = self._map_obj2class[obj]
        #     all_object_instances.append(obj)
        #     self._object_task_pairs_dataset.append("{}-{}".format(obj, task))

        #     pc_file = os.path.join(data_dir, obj, "fused_pc_clean.npy")
        #     if pc_file not in self._pc:
        #         if not os.path.exists(pc_file):
        #             raise ValueError(
        #                 'Unable to find processed point cloud file {}'.format(pc_file))
        #         pc = np.load(pc_file)
        #         pc_mean = pc[:, :3].mean(axis=0)
        #         pc[:, :3] -= pc_mean
        #         self._pc[pc_file] = pc

        #     grasp_file = os.path.join(
        #         data_dir, obj, "grasps", str(grasp_id), "grasp.npy")
        #     if grasp_file not in self._grasps:
        #         grasp = np.load(grasp_file)
        #         self._grasps[grasp_file] = grasp

        #     self._data.append(
        #         (grasp_file, pc_file, obj, obj_class, grasp_id, task, label))
        #     self._data_labels.append(int(label))
        #     if label:
        #         correct_counter += 1
        #         self._data_label_counter[1] += 1
        #     else:
        #         self._data_label_counter[0] += 1

        # self._all_object_instances = list(set(all_object_instances))

        # with open('instance.txt', 'w') as f:
        #     for item in self._all_object_instances:
        #         f.write("%s\n" % item)
        
        # # with open('data/taskgrasp_data/splits_final/o/3/test_split.txt', 'w') as f:
        # #     for item in self._all_object_instances:
        # #         f.write("%s\n" % item)

        # self._len = len(self._data)
        # print('Loading files from {} took {}s; overall dataset size {}, proportion successful grasps {:.2f}'.format(
        #     data_txt_splits[self._train], time.time() - start, self._len, float(correct_counter / self._len)))

        # self._data_labels = np.array(self._data_labels)

##########################################################################

    def __getitem__(self, idx):

        pc_file, grasp_set, obj = self._data[idx]
        meta = {}

        pc = self._pc[pc_file]
        pc = utils.regularize_pc_point_count(
            pc, self.opt.npoints)
        pc_color = pc[:, 3:]
        pc = pc[:,:3]
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3] 

        

        output_grasps = []
        for grasp_file in grasp_set:
            grasp = self._grasps[grasp_file]

            output_grasps.append(grasp)
            
        gt_control_points = utils.transform_control_points_numpy(
            np.array(output_grasps), self.opt.num_grasps_per_object, mode='rt')[:,:, :3]
        
        pcs = []
        final_grasp_pcs = []
        for i in range(self.opt.num_grasps_per_object):
            
            pc_tmp = copy.deepcopy(pc)
            grasp_pc = gt_control_points[i]
            # Introduce noise
            latent = np.concatenate(
            [np.zeros(pc_tmp.shape[0]), np.ones(grasp_pc.shape[0])])
            latent = np.expand_dims(latent, axis=1)
            pc_tmp = np.concatenate([pc_tmp, grasp_pc], axis=0)

            pc_tmp = np.concatenate([pc_tmp, latent], axis=1) # adding latent space

            if self._transforms is not None:
                pc_tmp = self._transforms(pc_tmp)


            pc_tmp, grasp_pc = torch.split(pc_tmp,[pc_tmp.shape[0]-grasp_pc.shape[0], grasp_pc.shape[0]])
            pc_tmp = pc_tmp[:,:-1]
            grasp_pc = grasp_pc[:,:-1]
            pcs.append(pc_tmp.numpy())
            final_grasp_pcs.append(grasp_pc.numpy())

        
        meta['pc'] = np.array(pcs).astype('float32')
        meta['pc_color'] = np.array([pc_color] * self.opt.num_grasps_per_object)[:, :, :3].astype('float32')
        meta['grasp_rt'] = np.array(output_grasps).reshape(
            len(output_grasps), -1)
        meta['target_cps'] = np.array(final_grasp_pcs)
        meta['obj'] = np.array([obj])

        return meta
        # grasp_pc = get_gripper_control_points()
        # grasp_pc = np.matmul(grasp, grasp_pc.T).T
        # grasp_pc = grasp_pc[:, :3]

        # latent = np.concatenate(
        #     [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        # latent = np.expand_dims(latent, axis=1)
        # pc = np.concatenate([pc, grasp_pc], axis=0)

        # latent = np.concatenate(
        #     [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        # latent = np.expand_dims(latent, axis=1)
        # pc = np.concatenate([pc, grasp_pc], axis=0)


        # #Normalize pc and grasp
        # pc, grasp = pc_normalize(pc, grasp, pc_scaling=self._pc_scaling)
        # pc = np.concatenate([pc, latent], axis=1) # adding latent space

        # if self._transforms is not None:
        #     pc = self._transforms(pc)

        # label = float(label)

        # pc, grasp_pc = torch.split(pc,[pc.shape[0]-grasp_pc.shape[0], grasp_pc.shape[0]])
        # pc = pc[:,:-1]
        # grasp_pc = grasp_pc[:,:-1]


        # return pc, pc_color, task_id, class_id, instance_id, grasp, grasp_pc, label, grasp_id

    def __len__(self):
        return self._len


    