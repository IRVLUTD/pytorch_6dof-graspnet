import os
import sys
import pickle
import omegaconf
import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from collections import defaultdict
import matplotlib.pyplot as plt

import torch.nn.functional as F
from data import DataLoader

from data.data_specification import TASKS, TASKS_SG14K
from models import create_model
from options.test_options import TestOptions
from options.train_options import TrainOptions
from utils.splits import get_ot_pairs_taskgrasp
from utils.utils import intialize_dataset, mkdir
from utils.visualization_utils import draw_scene

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../'))

DEVICE = "cuda"

# def visualize_batch(pc, grasps):
#     """ Visualizes all the data in the batch, just for debugging """

#     for i in range(pc.shape[0]):
#         pcc = pc[i, :, :3]
#         grasp = grasps[i, :, :]
#         draw_scene(pcc, [grasp, ])


# def visualize_batch_wrong(pc, grasps, labels, preds):
#     """ Visualizes incorrect predictions """

#     for i in range(pc.shape[0]):
#         if labels[i] != preds[i]:
#             print('labels {}, prediction {}'.format(labels[i], preds[i]))
#             pcc = pc[i, :, :3]
#             grasp = grasps[i, :, :]
#             draw_scene(pcc, [grasp, ])

def main(opt, save=False, visualize=False, experiment_dir=None):

    _, _, _, name2wn = pickle.load(
        open(os.path.join(
            opt.dataset_root_folder, 
            'misc.pkl'),'rb'))
    
    class_list = pickle.load(
        open(os.path.join(
            opt.dataset_root_folder, 
            'class_list.pkl'),'rb')) if opt.use_class_list else list(
        name2wn.values())

    opt.serial_batches = True  # no shuffle
    datasetIterator = DataLoader(opt)
    model = create_model(opt)

    all_preds = []
    all_probs = []
    all_labels = []
    all_data_vis = {}
    all_data_pc = {}

    # Our data annotation on MTurk happens in 2 stages, see paper for more details

    # Only considering Stage 2 grasps
    task1_results_file = os.path.join(
        opt.dataset_root_folder, 'task1_results.txt')
    assert os.path.exists(task1_results_file)

    if opt.dataset == 1:
        object_task_pairs = get_ot_pairs_taskgrasp(task1_results_file)
        TASK2_ot_pairs = object_task_pairs['True'] + \
            object_task_pairs['Weak True']
        TASK1_ot_pairs = object_task_pairs['False'] + \
            object_task_pairs['Weak False']
    else:
        raise ValueError('Unknown class {}'.format(opt.dataset))

    all_preds_2 = []
    all_probs_2 = []
    all_labels_2 = []

    all_preds_2_v2 = defaultdict(dict)
    all_probs_2_v2 = defaultdict(dict)
    all_labels_2_v2 = defaultdict(dict)

    # Only considering Stage 1 grasps
    all_preds_1 = []
    all_probs_1 = []
    all_labels_1 = []

    print('Running evaluation on Test set')
    with torch.no_grad():
        for data in tqdm(datasetIterator):

            model.set_input(data)
            probs, preds, _, _ = model.test()

            try:
                preds = preds.cpu().numpy()
                probs = probs.cpu().numpy()
                labels = data['labels']

                all_preds += list(preds)
                all_probs += list(probs)
                all_labels += list(labels)
            except TypeError:
                all_preds.append(preds.tolist())
                all_probs.append(probs.tolist())
                all_labels.append(labels.tolist()[0])

            tasks = data['task_id']
            obj = data['obj']
            for i in range(tasks.shape[0]):
                task = tasks[i]
                task = TASKS[task]
                obj_instance_name = obj[i]
                ot = "{}-{}".format(obj_instance_name, task)

                try:
                    pred = preds[i]
                    prob = probs[i]
                    label = labels[i]
                except IndexError:
                    # TODO: This is very hacky, fix it
                    pred = preds.tolist()
                    prob = probs.tolist()
                    label = labels.tolist()[0]

                if ot in TASK2_ot_pairs:
                    all_preds_2.append(pred)
                    all_probs_2.append(prob)
                    all_labels_2.append(label)

                    try:
                        all_preds_2_v2[obj_instance_name][task].append(pred)
                        all_probs_2_v2[obj_instance_name][task].append(prob)
                        all_labels_2_v2[obj_instance_name][task].append(label)

                    except KeyError:
                        all_preds_2_v2[obj_instance_name][task] = [pred, ]
                        all_probs_2_v2[obj_instance_name][task] = [prob, ]
                        all_labels_2_v2[obj_instance_name][task] = [label, ]

                elif ot in TASK1_ot_pairs:
                    all_preds_1.append(pred)
                    all_probs_1.append(prob)
                    all_labels_1.append(label)
                elif ot in ROUND1_GOLD_STANDARD_PROTOTYPICAL_USE:
                    all_preds_2.append(pred)
                    all_probs_2.append(prob)
                    all_labels_2.append(label)

                    try:
                        all_preds_2_v2[obj_instance_name][task].append(pred)
                        all_probs_2_v2[obj_instance_name][task].append(prob)
                        all_labels_2_v2[obj_instance_name][task].append(label)

                    except KeyError:
                        all_preds_2_v2[obj_instance_name][task] = [pred, ]
                        all_probs_2_v2[obj_instance_name][task] = [prob, ]
                        all_labels_2_v2[obj_instance_name][task] = [label, ]

                else:
                    raise Exception('Unknown ot {}'.format(ot))

            if visualize or save:

                pc = data['pc']
                grasps = data['grasp_rt']
                classes = data["class_id"]
                pc_color = data['pc_color']

                # Uncomment the following for debugging
                # visualize_batch(pc, grasps)
                # visualize_batch_wrong(pc, grasps, labels, preds)

                for i in range(pc.shape[0]):
                    pc_i = pc[i, :, :]
                    # pc_i = pc_i[np.where(pc_i[:, 3] == 0), :3].squeeze(0)
                    pc_color_i = pc_color[i, :, :3]
                    pc_i = np.concatenate([pc_i, pc_color_i], axis=1)
                    grasp = grasps[i, :, :]
                    task = tasks[i]
                    task = TASKS[task]
                    obj_instance_name = obj[i]
                    obj_class = classes[i]
                    obj_class = class_list[obj_class]

                    try:
                        pred = preds[i]
                        prob = probs[i]
                        label = labels[i]
                    except IndexError:
                        pred = preds.tolist()
                        prob = probs.tolist()
                        label = labels.tolist()[0]

                    ot = "{}-{}".format(obj_instance_name, task)
                    grasp_datapt = (grasp, prob, pred, label)
                    if ot in all_data_vis:
                        all_data_vis[ot].append(grasp_datapt)
                        all_data_pc[ot] = pc_i
                    else:
                        all_data_vis[ot] = [grasp_datapt, ]
                        all_data_pc[ot] = pc_i

    # Stage 1+2 grasps
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    random_probs = np.random.uniform(low=0, high=1, size=(len(all_probs)))
    results = {
        'preds': all_preds,
        'probs': all_probs,
        'labels': all_labels,
        'random': random_probs}

    # Only Stage 2 grasps
    all_preds_2 = np.array(all_preds_2)
    all_probs_2 = np.array(all_probs_2)
    all_labels_2 = np.array(all_labels_2)
    random_probs_2 = np.random.uniform(low=0, high=1, size=(len(all_probs_2)))
    results_2 = {
        'preds': all_preds_2,
        'probs': all_probs_2,
        'labels': all_labels_2,
        'random': random_probs_2}

    # Only Stage 1 grasps
    all_preds_1 = np.array(all_preds_1)
    all_probs_1 = np.array(all_probs_1)
    all_labels_1 = np.array(all_labels_1)
    random_probs_1 = np.random.uniform(low=0, high=1, size=(len(all_probs_1)))
    results_1 = {
        'preds': all_preds_1,
        'probs': all_probs_1,
        'labels': all_labels_1,
        'random': random_probs_1}

    # Only Stage 2 grasps
    random_probs_2 = np.random.uniform(low=0, high=1, size=(len(all_probs_2)))
    results_2_v2 = {
        'preds': all_preds_2_v2,
        'probs': all_probs_2_v2,
        'labels': all_labels_2_v2,
        'random': random_probs_2}

    if save:
        mkdir(os.path.join(experiment_dir, 'results'))
        pickle.dump(
            results,
            open(
                os.path.join(
                    experiment_dir,
                    'results',
                    "results.pkl"),
                'wb'))

        mkdir(os.path.join(experiment_dir, 'results1'))
        pickle.dump(
            results_1,
            open(
                os.path.join(
                    experiment_dir,
                    'results1',
                    "results.pkl"),
                'wb'))

        mkdir(os.path.join(experiment_dir, 'results2'))
        pickle.dump(
            results_2,
            open(
                os.path.join(
                    experiment_dir,
                    'results2',
                    "results.pkl"),
                'wb'))

    if save or visualize:
        mkdir(os.path.join(experiment_dir, 'results2_ap'))
        pickle.dump(
            results_2_v2,
            open(
                os.path.join(
                    experiment_dir,
                    'results2_ap',
                    "results.pkl"),
                'wb'))

        # TODO - Write separate script for loading and visualizing predictions
        # mkdir(os.path.join(experiment_dir, 'visualization_data'))
        # pickle.dump(
        #     all_data_vis,
        #     open(
        #         os.path.join(
        #             experiment_dir,
        #             'visualization_data',
        #             "predictions.pkl"),
        #         'wb'))

    if visualize:

        mkdir(os.path.join(experiment_dir, 'visualization'))
        mkdir(os.path.join(experiment_dir, 'visualization', 'task1'))
        mkdir(os.path.join(experiment_dir, 'visualization', 'task2'))

        print('saving ot visualizations')
        for ot in all_data_vis.keys():

            if ot in TASK1_ot_pairs:
                save_dir = os.path.join(
                    experiment_dir, 'visualization', 'task1')
            elif ot in TASK2_ot_pairs:
                save_dir = os.path.join(
                    experiment_dir, 'visualization', 'task2')
            else:
                continue

            pc = all_data_pc[ot]
            grasps_ot = all_data_vis[ot]
            grasps = [elem[0] for elem in grasps_ot]
            probs = np.array([elem[1] for elem in grasps_ot])
            preds = np.array([elem[2] for elem in grasps_ot])
            labels = np.array([elem[3] for elem in grasps_ot])

            grasp_color = np.stack(
                [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])], axis=1)
            draw_scene(pc, grasps, grasp_color=list(grasp_color), max_grasps=len(
                grasps), save_dir=os.path.join(save_dir, '{}_gt.png'.format(ot)))

            grasp_color = np.stack(
                [np.ones(preds.shape[0]) - preds, preds, np.zeros(preds.shape[0])], axis=1)
            draw_scene(pc, grasps, grasp_color=list(grasp_color), max_grasps=len(
                grasps), save_dir=os.path.join(save_dir, '{}_pred.png'.format(ot)))

            grasp_color = np.stack(
                [np.ones(probs.shape[0]) - probs, probs, np.zeros(probs.shape[0])], axis=1)
            draw_scene(pc, grasps, grasp_color=list(grasp_color), max_grasps=len(
                grasps), save_dir=os.path.join(save_dir, '{}_probs.png'.format(ot)))


if __name__ == "__main__":

    opt = TestOptions().parse()

    opt.name = opt.model_name

    intialize_dataset(opt.dataset)

    experiment_dir = os.path.join(opt.checkpoints_dir, opt.name)

    main(
        opt,
        save=opt.save,
        visualize=opt.visualize,
        experiment_dir=experiment_dir)
