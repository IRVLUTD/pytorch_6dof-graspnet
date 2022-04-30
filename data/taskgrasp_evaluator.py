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


from utils.splits import get_split_data, parse_line, get_ot_pairs_taskgrasp, get_task1_hits, get_valid_ot_pairs
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
        self._object_task_grasp_dataset = {}
        self._all_object_instances = {} # converted to list later
        self._all_tasks = {}
        self._object_classes = class_list

        self._num_object_classes = len(self._object_classes)

        self._grasp_ratio = {}

        start = time.time()
        correct_counter = 0

        

        all_object_instances = []
        all_tasks = []
        

        for i in tqdm.trange(len(lines)):
            obj, obj_class, grasp_id, task, label = parse_line(lines[i])

            all_object_instances.append(obj)
            if obj not in self._all_object_instances:
                self._all_object_instances[obj] = None
            all_tasks.append(task)

            if obj not in self._all_tasks:
                self._all_tasks[task] = None

            self._object_task_grasp_dataset["{}-{}-{}".format(obj, task, grasp_id)] = label
        
        self._all_object_instances = list(self._all_object_instances)
        # self._all_tasks = list(set(all_tasks))

        # if self._train == "train":
        #     with open('instance.txt', 'w') as f:
        #         for item in self._all_object_instances:
        #             f.write("%s\n" % item)


        # with open(self._train  + '_ratio.txt', 'w') as f:
        #     pass

        # Get valid object task pair
        # task1_results_file = os.path.join(opt.dataset_root_folder, 'task1_results.txt')
        # assert os.path.exists(task1_results_file)
        # object_task_pairs = get_object_task_pairs(task1_results_file)

        task1_results_file = os.path.join(
            opt.dataset_root_folder, 'task1_results.txt')
        assert os.path.exists(task1_results_file)

        valid_ot_pair = get_valid_ot_pairs(task1_results_file)


        
        for obj in tqdm.tqdm(self._all_object_instances):

            obj_class = self._map_obj2class[obj]

            pc_file = os.path.join(data_dir, obj, "fused_pc_clean.npy")
            if pc_file not in self._pc:
                if not os.path.exists(pc_file):
                    raise ValueError(
                        'Unable to find processed point cloud file {}'.format(pc_file))
                pc = np.load(pc_file)
                pc_mean = pc[:, :3].mean(axis=0)
                pc[:, :3] -= pc_mean
                self._pc[pc_file] = pc

            grasp_set = []

            for grasp_id in range(opt.num_grasps_per_object):
                grasp_file = os.path.join(
                    data_dir, obj, "grasps", str(grasp_id), "grasp.npy")
                if not os.path.exists(grasp_file):
                    raise ValueError(
                        'Unable to find grasp point cloud file {}'.format(grasp_file))
                if grasp_file not in self._grasps:
                    grasp = np.load(grasp_file)
                    grasp_set.append(grasp_file)
                    self._grasps[grasp_file] = grasp
            
            for task in self._all_tasks:

                data_label_counter = {0: 0, 1: 0}

                if task in valid_ot_pair[obj]:

                    self._data.append((pc_file, grasp_set, obj, obj_class, task))

                    for grasp_id in range(self.opt.num_grasps_per_object):
            
                        key = "{}-{}-{}".format(obj, task, grasp_id)
                        if key in self._object_task_grasp_dataset:
                            
                            label = self._object_task_grasp_dataset[key]

                            if label:
                                data_label_counter[1] += 1
                            else:
                                data_label_counter[0] += 1
                        
                    # with open(self._train + '_ratio.txt', 'a') as f:
                    #     if data_label_counter[0] + data_label_counter[1] != 0 :
                    #         f.write("%s:\t%s-\t%s\n" % (obj, task, float(data_label_counter[1] / (data_label_counter[0] + data_label_counter[1]))))
                    #     else:
                    #         f.write("%s:\t%s- Data not exist\n" % (obj, task))
                        


        self._len = len(self._data)

        ### For testing
        # self._len = 3

        print('Loading files from {} took {}s; overall dataset size {}'.format(
            data_txt_splits[self._train], time.time() - start, self._len))


        # with open('instance.txt', 'w') as f:
        #     for item in self._all_object_instances:
        #         f.write("%s\n" % item)

    def set_ratios(self, ratio):
        if int(self.opt.num_grasps_per_object * ratio) == 0:
            return 1 / self.opt.num_grasps_per_object
        return ratio

    def __getitem__(self, index):
        pc_file, grasp_set, obj, obj_class, task = self._data[index]

        pc = self._pc[pc_file]
        pc = utils.regularize_pc_point_count(
            pc, self.opt.npoints)
        pc_color = pc[:, 3:]
        pc = pc[:,:3]


        output_pcs = []
        output_grasps = []
        output_grasps_cp = []
        output_labels = []
        output_obj = []
        output_class_id = []
        task_ids = []

        class_id = self._object_classes.index(obj_class)

        for grasp_id in range(self.opt.num_grasps_per_object):
            
            key = "{}-{}-{}".format(obj, task, grasp_id)
            if key in self._object_task_grasp_dataset:
                grasp = self._grasps[grasp_set[grasp_id]]
                label = self._object_task_grasp_dataset[key]
                task_id = self._tasks.index(task)

                output_grasps.append(grasp)
                output_obj.append(obj)
                output_labels.append(label)
                task_ids.append(task_id)
                output_class_id.append(class_id)

        # for grasp_file in grasp_set:
        #     grasp = self._grasps[grasp_file]
        #     output_grasps.append(grasp)
    

        gt_control_points = utils.transform_control_points_numpy(
            np.array(output_grasps), self.opt.num_grasps_per_object, mode='rt')[:,:, :3]


        total_cp_points = gt_control_points.shape[0] * gt_control_points.shape[1]
        latent = np.concatenate(
        [np.zeros(pc.shape[0]), np.ones(total_cp_points)])
        latent = np.expand_dims(latent, axis=1)
        pc = np.concatenate([pc, gt_control_points.reshape(total_cp_points, 3)], axis=0)

        pc = np.concatenate([pc, latent], axis=1) # adding latent space

        if self._transforms is not None:
            pc = self._transforms(pc)
        
        pc, grasps_pc = torch.split(pc,[pc.shape[0]-total_cp_points, total_cp_points])


        grasps_pc = grasps_pc.numpy()[:, :3]
        grasps_pc = grasps_pc.reshape(gt_control_points.shape[0], gt_control_points.shape[1], 3)
        pc = pc.numpy()[:, :3]

        # for grasp_id in range(self.opt.num_grasps_per_object):
            
        #     key = "{}-{}-{}".format(obj, task, grasp_id)
        #     if key in self._object_task_grasp_dataset:

        #         pc_tmp = copy.deepcopy(pc)
        #         grasp_pc = gt_control_points[grasp_id]
        #         task_id = self._tasks.index(task)

        #         label = self._object_task_grasp_dataset[key]

        #         latent = np.concatenate(
        #             [np.zeros(pc_tmp.shape[0]), np.ones(grasp_pc.shape[0])])
        #         latent = np.expand_dims(latent, axis=1)
        #         pc_tmp = np.concatenate([pc_tmp, grasp_pc], axis=0)

        #         pc_tmp = np.concatenate([pc_tmp, latent], axis=1)

        #         if self._transforms is not None:
        #             pc_tmp = self._transforms(pc_tmp)

        #         # pc, grasp_pc = torch.split(pc,[pc.shape[0]-grasp_pc.shape[0], grasp_pc.shape[0]])

        #         pc_final = pc_tmp.cpu().detach().numpy()
        #         # grasp_pc = grasp_pc.cpu().detach().numpy()
        #         output_pcs.append(pc_final)
        #         output_obj.append(obj)
        #         output_grasps_cp.append(pc_final[pc_final.shape[0]-grasp_pc.shape[0]:, :3])
        #         output_labels.append(label)
        #         task_ids.append(task_id)
                

        meta = {}
        meta['pc'] = np.array([pc] * len(output_grasps))
        meta['pc_color'] = np.array([pc_color] * len(output_grasps))
        meta['grasp_rt'] = np.array(output_grasps)
        meta['target_cps'] = np.array(grasps_pc)
        meta['labels'] = np.array(output_labels)
        meta['obj'] = np.array(output_obj)
        meta['task_id'] = np.array(torch.LongTensor(task_ids))
        meta['class_id'] = np.array(output_class_id)
        return meta

    def __len__(self):
        return self._len