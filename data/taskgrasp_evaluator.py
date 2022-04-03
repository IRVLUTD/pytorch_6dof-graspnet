import os
import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
import random
import tqdm
import time
try:
    from Queue import Queue
except:
    from queue import Queue


from utils.splits import get_split_data, parse_line, get_ot_pairs_taskgrasp, get_task1_hits
from utils import utils


class TaskGraspEvaluatorData(BaseDataset):
    def __init__(
        self,
        opt,
        use_task1_grasps=True,
        tasks=None, 
        class_list=None,
        map_obj2class=None,
        transforms=None):

        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        folder_dir = ""
        self._pc_scaling = opt.pc_scaling
        self._split_mode = opt.split_mode
        self._split_idx = opt.split_idx
        self._split_version = opt.split_version
        self._transforms = transforms
        self._tasks = tasks
        self._num_tasks = len(self._tasks)

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
        # if(len(lines) > 10000):
        #     lines = lines[:10000]

        self._data = []
        self._pc = {}
        self._grasps = {}

        start = time.time()
        correct_counter = 0

        all_object_instances = []

        self._object_task_pairs_dataset = []
        self._data_labels = []
        self._data_label_counter = {0: 0, 1: 0}

        for i in tqdm.trange(len(lines)):
            obj, obj_class, grasp_id, task, label = parse_line(lines[i])
            obj_class = self._map_obj2class[obj]
            all_object_instances.append(obj)
            self._object_task_pairs_dataset.append("{}-{}".format(obj, task))

            pc_file = os.path.join(data_dir, obj, "fused_pc_clean.npy")
            if pc_file not in self._pc:
                if not os.path.exists(pc_file):
                    raise ValueError(
                        'Unable to find processed point cloud file {}'.format(pc_file))
                pc = np.load(pc_file)
                pc_mean = pc[:, :3].mean(axis=0)
                pc[:, :3] -= pc_mean
                self._pc[pc_file] = pc

            grasp_file = os.path.join(
                data_dir, obj, "grasps", str(grasp_id), "grasp.npy")
            if grasp_file not in self._grasps:
                grasp = np.load(grasp_file)
                self._grasps[grasp_file] = grasp

            self._data.append(
                (grasp_file, pc_file, obj, obj_class, grasp_id, task, label))
            self._data_labels.append(int(label))
            if label:
                correct_counter += 1
                self._data_label_counter[1] += 1
            else:
                self._data_label_counter[0] += 1

        self._all_object_instances = list(set(all_object_instances))
        self._len = len(self._data)
        print('Loading files from {} took {}s; overall dataset size {}, proportion successful grasps {:.2f}'.format(
            data_txt_splits[self._train], time.time() - start, self._len, float(correct_counter / self._len)))

        self._data_labels = np.array(self._data_labels)

        # with open('instance.txt', 'w') as f:
        #     for item in self._all_object_instances:
        #         f.write("%s\n" % item)

    def set_ratios(self, ratio):
        if int(self.opt.num_grasps_per_object * ratio) == 0:
            return 1 / self.opt.num_grasps_per_object
        return ratio

    def __getitem__(self, index):

        grasp_file, pc_file, obj, obj_class, grasp_id, task, label = self._data[index]
        pc = self._pc[pc_file]
        pc = utils.regularize_pc_point_count(
            pc, self.opt.npoints)
        pc_color = pc[:, 3:]
        pc = pc[:,:3]
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc -= pc_mean

        grasp = self._grasps[grasp_file]
        task_id = self._tasks.index(task)
        instance_id = self._all_object_instances.index(obj)

        grasp_pc = utils.get_gripper_control_points_taskgrasp()
        grasp_pc = np.matmul(grasp, grasp_pc.T).T
        grasp_pc = grasp_pc[:, :3]

        latent = np.concatenate(
            [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        latent = np.expand_dims(latent, axis=1)
        pc = np.concatenate([pc, grasp_pc], axis=0)

        pc = np.concatenate([pc, latent], axis=1)

        if self._transforms is not None:
            pc = self._transforms(pc)

        # pc, grasp_pc = torch.split(pc,[pc.shape[0]-grasp_pc.shape[0], grasp_pc.shape[0]])

        pc = pc.cpu().detach().numpy()
        # grasp_pc = grasp_pc.cpu().detach().numpy()
        
        meta = {}
        meta['pc'] = np.array([pc])
        meta['pc_color'] = np.array([pc_color])
        meta['grasp_rt'] = np.array([grasp])
        meta['target'] = np.array([pc[pc.shape[0]-grasp_pc.shape[0]:, :3]])
        meta['labels'] = np.array([label])
        meta['obj'] = np.array([obj])
        meta['grasp_id'] = np.array([grasp_id])
        meta['task_id'] = np.array(torch.nn.functional.one_hot(torch.LongTensor([task_id]), self._num_tasks))
        return meta

    def __len__(self):
        return self._len