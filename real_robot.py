import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import numpy as np
import cv2
import torch
import time
import datetime
import os
import random
import threading
import argparse

import torch
import numpy as np
import cv2

from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
from lwrf_infer import LwrfInfer
from policies import Explorer, Coordinator
def main(arg):
    nonlocal_variables = {'executing_action': False,
                          'primitive_action': None,
                          'seeking_target': False,
                          'best_push_pix_ind': None,
                          'push_end_pix_yx': None,
                          'margin_occupy_ratio': None,
                          'clutter_degree': None,
                          'margin_occupy_norm': None,
                          'best_grasp_pix_ind': None,
                          'best_pix_ind': None,
                          'target_grasped': False}
    timestamp = time.time()
    timestamp_value = datetime.datetime.fromtimestamp(timestamp)
    logging_directory = os.path.join(os.path.abspath('logs'), 'testing/release', timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
    print('Creating data logging session: %s' % logging_directory)
    logger = Logger(logging_directory)
    lwrf_model = LwrfInfer(use_cuda=False, save_path=logger.lwrf_results_directory)
    future_reward_discount = args.future_reward_discount
    is_testing = args.is_testing
    load_ckpt = args.load_ckpt
    force_cpu = args.force_cpu
    critic_ckpt_file = os.path.abspath(args.critic_ckpt) if load_ckpt else None
    coordinator_ckpt_file = os.path.abspath(args.coordinator_ckpt) if load_ckpt else None
    trainer = Trainer(future_reward_discount, is_testing, load_ckpt, critic_ckpt_file, force_cpu)
    coordinator = Coordinator(save_dir=logger.coordinator_directory, ckpt_file=coordinator_ckpt_file)
    #cam_intrinsics = np.asarray([[598.39, 0, 317.9536], [0, 598.4988, 317.78], [0, 0, 1]])
    #cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
    cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
    #workspace_limits = np.asarray([[-0.914, -0.496], [-0.434, -0.016], [-0.0001, 0.4]])
    workspace_limits = np.asarray([[-0.914, -0.496], [-0.284, 0.134], [-0.0001, 0.4]])
    heightmap_resolution = args.heightmap_resolution
    # cam_trans = np.eye(4, 4)
    # cam_trans[0:3, 3] = np.asarray(cam_position)
    # cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
    # cam_rotm = np.eye(4, 4)
    # cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
    #cam_pose = np.asarray([[0,1,0,-0.65], [1,0,0,-0.05], [0,0,-1,0.8],[0,0,0,1]])
    cam_pose = np.asarray([[-1, 0, 0, -0.65], [0, 1, 0, -0.05], [0, 0, -1, 0.8], [0, 0, 0, 1]])
    #cam_pose = np.asarray([[1, 0, 0, -0.65], [0, -1, 0, -0.05], [0, 0, 1, 0.8], [0, 0, 0, 1]])
    #cam_pose = np.asarray([[0, -1, 0, -0.65], [1, 0, 0, -0.05], [0, 0, 1, 0.8], [0, 0, 0, 1]])
    #cam_pose = np.asarray([[-1, 0, 0, -0.500268], [0, 1,0, -0.1078056], [0, 0, -1, 0.80374099], [0, 0, 0, 1]])
    # cam_pose=np.linalg.inv(cam_pose)
    # print(cam_pose)
    color_images_directory='/home/tongjiayuan/tjy/AUBO-python/camera/realsense-py/data/color/001116.color.png' #%006d.color.png
    depth_images_directory='/home/tongjiayuan/tjy/AUBO-python/camera/realsense-py/data/depth/001116.depth.npy'

    color_img = cv2.imread(color_images_directory)
    depth_img = np.load(depth_images_directory)

    print(depth_img[320][240])

    depth_img = depth_img/1000
    print(depth_img[320][240])
    color_img= cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("color", color_img)
    cv2.waitKey(1000)

    # 读取到的深度信息/1000 为真实的深度信息，单位为m
    # truth_depth = depth_image[x, y]/1000
    # 如果深度信息为0, 则说明没有获取到

    segment_results = lwrf_model.segment(color_img)
    print(segment_results)
    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)

    color_heightmap, depth_heightmap, seg_mask_heightmaps = utils.get_heightmap(
        color_img, depth_img, segment_results['masks'], cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution)
    cv2.imshow("color", color_heightmap)
    cv2.waitKey(2000)

    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
    mask_heightmaps = utils.process_mask_heightmaps(segment_results, seg_mask_heightmaps)
    print(mask_heightmaps)
    if len(mask_heightmaps['names']) == 0 and False:
        nonlocal_variables['seeking_target'] = True
        target_mask_heightmap = np.ones_like(valid_depth_heightmap)
    else:
        target_id = random.randint(0, len(mask_heightmaps['names']) - 1)
        target_id=0
        target_name = mask_heightmaps['names'][target_id]
        target_mask_heightmap = mask_heightmaps['heightmaps'][target_id]
        print('lwrf segments:', mask_heightmaps['names'])
        print('Target name:', target_name)
    logger.save_heightmaps(trainer.iteration, color_heightmap, depth_heightmap, target_mask_heightmap)
    nonlocal_variables['margin_occupy_ratio'], nonlocal_variables['margin_occupy_norm'] = utils.check_grasp_margin(
        target_mask_heightmap, depth_heightmap)
    grasp_fail_count = [0]
    push_predictions, grasp_predictions, state_feat = trainer.forward(
        color_heightmap, valid_depth_heightmap, target_mask_heightmap, is_volatile=True)
    nonlocal_variables['best_push_pix_ind'], nonlocal_variables['push_end_pix_yx'] = utils.get_push_pix(
        grasp_predictions, trainer.model.num_rotations)
    nonlocal_variables['best_grasp_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
    if True:
        # print(nonlocal_variables['best_push_pix_ind'],nonlocal_variables['push_end_pix_yx'])
        # print(nonlocal_variables['best_grasp_pix_ind'])
        # nonlocal_variables['best_push_pix_ind']=(0,104,88)
        push_pred_vis = trainer.get_push_prediction_vis(grasp_predictions, color_heightmap,
                                                        nonlocal_variables['best_push_pix_ind'],
                                                        nonlocal_variables['push_end_pix_yx'])
        logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
        cv2.imwrite('visualization.push.png', push_pred_vis)
        # cv2.imshow("color", push_pred_vis)
        # cv2.waitKey(7000)
        grasp_pred_vis = trainer.get_grasp_prediction_vis(grasp_predictions, color_heightmap,
                                                          nonlocal_variables['best_grasp_pix_ind'])
        logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
        cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
        cv2.imshow("color", grasp_pred_vis)
        cv2.waitKey(7000)
        if nonlocal_variables['seeking_target']:
            print('Seeking target in testing mode')
            nonlocal_variables['primitive_action'] = 'push'
            best_push_conf = np.max(push_predictions)
            print('Primitive confidence scores: %f (push)' % (best_push_conf))

            if nonlocal_variables['primitive_action'] == 'push':
                nonlocal_variables['best_pix_ind'] = nonlocal_variables['best_push_pix_ind']
                predicted_value = np.max(push_predictions)
        else:
            # Determine whether grasping or pushing should be executed based on network predictions
            best_push_conf = np.max(push_predictions)
            best_grasp_conf = np.max(grasp_predictions)
            print('Primitive confidence scores: %f (push), %f (grasp)' % (best_push_conf, best_grasp_conf))

            # Actor
            if not is_testing:
                print('Greedy deterministic policy ...')
                motion_type = 1 if best_grasp_conf > best_push_conf else 0
            else:
                print('Coordination policy ...')
                syn_input = [best_push_conf, best_grasp_conf, nonlocal_variables['margin_occupy_ratio'],
                             nonlocal_variables['margin_occupy_norm'], grasp_fail_count[0]]
                motion_type = coordinator.predict(syn_input)
                #motion_type=0
            explore_actions = np.random.uniform() < 0
            if explore_actions:
                print('Exploring actions, explore_prob: %f' % 0)
                motion_type = 1 - 0

            nonlocal_variables['primitive_action'] = 'push' if motion_type == 0 else 'grasp'

            if nonlocal_variables['primitive_action'] == 'push':
                grasp_fail_count[0] = 0
                nonlocal_variables['best_pix_ind'] = nonlocal_variables['best_push_pix_ind']
                predicted_value = np.max(push_predictions)
            elif nonlocal_variables['primitive_action'] == 'grasp':
                nonlocal_variables['best_pix_ind'] = nonlocal_variables['best_grasp_pix_ind']
                predicted_value = np.max(grasp_predictions)

            # Save predicted confidence value
            trainer.predicted_value_log.append([predicted_value])
            logger.write_to_log('predicted-value', trainer.predicted_value_log)

        # Compute 3D position of pixel
        print('Action: %s at (%d, %d, %d)' % (
        nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0],
        nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))

        best_rotation_angle = nonlocal_variables['best_pix_ind'][0] * (360.0 / trainer.model.num_rotations)
        best_pix_x = nonlocal_variables['best_pix_ind'][2]
        best_pix_y = nonlocal_variables['best_pix_ind'][1]
        primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                              best_pix_y * heightmap_resolution + workspace_limits[1][0],
                              valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]
        primitive_position = [-0.65 * 2-primitive_position[0],primitive_position[1],primitive_position[2]]
        primitive_position=[-0.65-(primitive_position[1])+0.05,-0.05+primitive_position[0]+0.65,primitive_position[2]]
        print(primitive_position)
        print(best_rotation_angle)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store',
                        default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234)
    parser.add_argument('--force_cpu', dest='force_cpu', action='store_true', default=False)

    # --------------- Object options ---------------
    parser.add_argument('--config_file', dest='config_file', action='store',
                        default='simulation/preset/exploration-03.txt')  # exploration-00,coordination-00

    # ------------- Algorithm options -------------
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store',
                        default=0.5)
    parser.add_argument('--stage_epoch', dest='stage_epoch', type=int, action='store', default=1000)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=True)
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=True)
    parser.add_argument('--test_target_seeking', dest='test_target_seeking', action='store_true', default=True)
    parser.add_argument('--max_motion_onecase', dest='max_motion_onecase', type=int, action='store', default=20,
                        help='maximum number of motions per test trial')
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=20,
                        help='number of repeated test trials')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_ckpt', dest='load_ckpt', action='store_true', default=True)
    parser.add_argument('--critic_ckpt', dest='critic_ckpt', action='store',
                        default='/home/tongjiayuan/tjy/copy0730/grasping-invisible-master (3)/logs/2021-06-15.16_17_20/critic_models/critic-003500.pth')
    parser.add_argument('--coordinator_ckpt', dest='coordinator_ckpt', action='store',
                        default='/home/tongjiayuan/tjy/copy0730/grasping-invisible-master (3)/logs/2021-06-15.16_17_20/coordinator_models/coordinator-004500.pth')
    #default='/home/tongjiayuan/tjy/copy0730/grasping-invisible-master (3)/logs/2021-06-15.16_17_20/critic_models/critic-005000.pth')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False)
    parser.add_argument('--logging_directory', dest='logging_directory', action='store',
                        default='/home/tongjiayuan/tjy/invisible1/logs/2021-12-01.20:35:18')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True)

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)