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
        ratio_positive=0.3, 
        ratio_hardnegative=0.4):

        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        folder_dir = ""
        self._train = opt.phase

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

        all_object_instances = []

        self._object_task_pairs_dataset = []
        self._data_labels = []
        self._data_label_counter = {0: 0, 1: 0}

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
        
        
        
        
        
        self.root = opt.dataset_root_folder
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.collision_hard_neg_queue = {}
        #self.get_mean_std()
        opt.input_nc = self.ninput_channels
        self.ratio_positive = self.set_ratios(ratio_positive)
        self.ratio_hardnegative = self.set_ratios(ratio_hardnegative)

    def set_ratios(self, ratio):
        if int(self.opt.num_grasps_per_object * ratio) == 0:
            return 1 / self.opt.num_grasps_per_object
        return ratio

    def __getitem__(self, index):
        path = self.paths[index]
        if self.opt.balanced_data:
            data = self.get_uniform_evaluator_data(path)
        else:
            data = self.get_nonuniform_evaluator_data(path)

        gt_control_points = utils.transform_control_points_numpy(
            data[1], self.opt.num_grasps_per_object, mode='rt')

        meta = {}
        meta['pc'] = data[0][:, :, :3]
        meta['grasp_rt'] = gt_control_points[:, :, :3]
        meta['labels'] = data[2]
        meta['quality'] = data[3]
        meta['pc_pose'] = data[4]
        meta['cad_path'] = data[5]
        meta['cad_scale'] = data[6]
        return meta

    def __len__(self):
        return self.size

    def get_uniform_evaluator_data(self, path, verify_grasps=False):
        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(
            path)

        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self.opt.batch_size
        output_cad_scales = np.asarray([cad_scale] * self.opt.batch_size,
                                       np.float32)

        num_positive = int(self.opt.batch_size * self.opt.ratio_positive)
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps,
                                                      pos_qualities)
        num_hard_negative = int(self.opt.batch_size *
                                self.opt.ratio_hardnegative)
        num_flex_negative = self.opt.batch_size - num_positive - num_hard_negative
        negative_clusters = self.sample_grasp_indexes(num_flex_negative,
                                                      neg_grasps,
                                                      neg_qualities)
        hard_neg_candidates = []
        # Fill in Positive Examples.

        for clusters, grasps, qualities in zip(
            [positive_clusters, negative_clusters], [pos_grasps, neg_grasps],
            [pos_qualities, neg_qualities]):
            for cluster in clusters:
                selected_grasp = grasps[cluster[0]][cluster[1]]
                selected_quality = qualities[cluster[0]][cluster[1]]
                hard_neg_candidates += utils.perturb_grasp(
                    selected_grasp,
                    self.collision_hard_neg_num_perturbations,
                    self.collision_hard_neg_min_translation,
                    self.collision_hard_neg_max_translation,
                    self.collision_hard_neg_min_rotation,
                    self.collision_hard_neg_max_rotation,
                )

        if verify_grasps:
            collisions, heuristic_qualities = utils.evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if path not in self.collision_hard_neg_queue or len(
                self.collision_hard_neg_queue[path]) < num_hard_negative:
            if path not in self.collision_hard_neg_queue:
                self.collision_hard_neg_queue[path] = []
            #hard negatives are perturbations of correct grasps.
            collisions, heuristic_qualities = utils.evaluate_grasps(
                hard_neg_candidates, obj_mesh)

            hard_neg_mask = collisions | (heuristic_qualities < 0.001)
            hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
            np.random.shuffle(hard_neg_indexes)
            for index in hard_neg_indexes:
                self.collision_hard_neg_queue[path].append(
                    (hard_neg_candidates[index], -1.0))
            random.shuffle(self.collision_hard_neg_queue[path])

        # Adding positive grasps
        for positive_cluster in positive_clusters:
            #print(positive_cluster)
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)

        # Adding hard neg
        for i in range(num_hard_negative):
            grasp, quality = self.collision_hard_neg_queue[path][i]
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        self.collision_hard_neg_queue[path] = self.collision_hard_neg_queue[
            path][num_hard_negative:]

        # Adding flex neg
        if len(negative_clusters) != num_flex_negative:
            raise ValueError(
                'negative clusters should have the same length as num_flex_negative {} != {}'
                .format(len(negative_clusters), num_flex_negative))

        for negative_cluster in negative_clusters:
            selected_grasp = neg_grasps[negative_cluster[0]][
                negative_cluster[1]]
            selected_quality = neg_qualities[negative_cluster[0]][
                negative_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(0)

        #self.change_object(cad_path, cad_scale)
        for iter in range(self.opt.num_grasps_per_object):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                pc, camera_pose, _ = self.change_object_and_render(
                    cad_path,
                    cad_scale,
                    thread_id=torch.utils.data.get_worker_info().id
                    if torch.utils.data.get_worker_info() else 0)
                output_pcs.append(pc)
                output_pc_poses.append(utils.inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)

        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales

    def get_nonuniform_evaluator_data(self, path, verify_grasps=False):

        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(
            path)

        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self.opt.num_grasps_per_object
        output_cad_scales = np.asarray(
            [cad_scale] * self.opt.num_grasps_per_object, np.float32)

        num_positive = int(self.opt.num_grasps_per_object *
                           self.ratio_positive)
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps,
                                                      pos_qualities)
        num_negative = self.opt.num_grasps_per_object - num_positive
        negative_clusters = self.sample_grasp_indexes(num_negative, neg_grasps,
                                                      neg_qualities)
        hard_neg_candidates = []
        # Fill in Positive Examples.
        for positive_cluster in positive_clusters:
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)
            hard_neg_candidates += utils.perturb_grasp(
                selected_grasp,
                self.collision_hard_neg_num_perturbations,
                self.collision_hard_neg_min_translation,
                self.collision_hard_neg_max_translation,
                self.collision_hard_neg_min_rotation,
                self.collision_hard_neg_max_rotation,
            )

        if verify_grasps:
            collisions, heuristic_qualities = utils.evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if path not in self.collision_hard_neg_queue or self.collision_hard_neg_queue[
                path].qsize() < num_negative:
            if path not in self.collision_hard_neg_queue:
                self.collision_hard_neg_queue[path] = Queue()
            #hard negatives are perturbations of correct grasps.
            random_selector = np.random.rand()
            if random_selector < self.ratio_hardnegative:
                #print('add hard neg')
                collisions, heuristic_qualities = utils.evaluate_grasps(
                    hard_neg_candidates, obj_mesh)
                hard_neg_mask = collisions | (heuristic_qualities < 0.001)
                hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
                np.random.shuffle(hard_neg_indexes)
                for index in hard_neg_indexes:
                    self.collision_hard_neg_queue[path].put(
                        (hard_neg_candidates[index], -1.0))
            if random_selector >= self.ratio_hardnegative or self.collision_hard_neg_queue[
                    path].qsize() < num_negative:
                for negative_cluster in negative_clusters:
                    selected_grasp = neg_grasps[negative_cluster[0]][
                        negative_cluster[1]]
                    selected_quality = neg_qualities[negative_cluster[0]][
                        negative_cluster[1]]
                    self.collision_hard_neg_queue[path].put(
                        (selected_grasp, selected_quality))

        # Use negative examples from queue.
        for _ in range(num_negative):
            #print('qsize = ', self._collision_hard_neg_queue[file_path].qsize())
            grasp, quality = self.collision_hard_neg_queue[path].get()
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        for iter in range(self.opt.num_grasps_per_object):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                pc, camera_pose, _ = self.change_object_and_render(
                    cad_path,
                    cad_scale,
                    thread_id=torch.utils.data.get_worker_info().id
                    if torch.utils.data.get_worker_info() else 0)
                #self.change_object(cad_path, cad_scale)
                #pc, camera_pose, _ = self.render_random_scene()

                output_pcs.append(pc)
                output_pc_poses.append(utils.inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)
        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales
