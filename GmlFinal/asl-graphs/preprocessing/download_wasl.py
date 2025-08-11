######################### Download Dataset ##############################
import kagglehub
import os
import subprocess
import cv2
import mediapipe as mp
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import random

import numpy as np
import pickle

from utils import *
from reorg import *

if __name__ == '__main__':

    # Download latest version
    
    path = '/home/tkg5kq/.cache/kagglehub/datasets/'
    if not os.path.exists(path + 'risangbaskoro/wlasl-processed/versions/5/'):
        path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
        print(f"Path to dataset files: {path}")
    else:
        path = path + 'risangbaskoro/wlasl-processed/versions/5/'
        print(f'PATH: {path}')


    # new dataset name
    org_dir = 'videos_org'
    org_path = os.path.join(path, org_dir)

    print(org_path)
    directory = 'wlasl_10'
    resized_dir = os.path.join(path, 'resized_' + directory)
    frame_dir = os.path.join(path, 'frames_' + directory)
    seed = 100
    random.seed(seed)

    if not os.path.exists(org_path):
        threshold = 0
        file = org_dir
        json_file = os.path.join(path, 'WLASL_v0.3.json')
        reorg(json_file, path, threshold, file)
    else:
        print(f'Reorg path exists: {org_path}')

    print('Creating resize dir....')
    if not os.path.exists(resized_dir) and not os.path.exists(frame_dir):
        print(resized_dir)
        # Level 2 because the format is root/class/mp4
        # outpath is the resized dir
        # video path is the org_path
        subprocess.run(f'python preprocessing/video_resize.py {org_path} {resized_dir} --level 2 -se 256'.split(' '))
    else:
        print(f'Resize Directory {"exists" if os.path.exists(resized_dir) else "does not exist"} and Frame Directory {"exists" if os.path.exists(frame_dir) else "does not exist"}')

    print('Creating frame dir....')
    if not os.path.exists(frame_dir):
        print(frame_dir)
        subprocess.run(f'python preprocessing/video2image.py {resized_dir} {frame_dir} --level 2 --lib opencv --prefix image_%04d.jpg'.split(' '))
    else:
        print(f'Frame Directory {"exists" if os.path.exists(frame_dir) else "does not exist"}')


    # mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions.drawing_styles
    # mp_holistic = mp.solutions.holistic

    # # num_classes = 10
    # # image_height = 64
    # # image_width = 80
    # # classes = ['love', 'thankyou', 'face', 'no', 'thin', 'man', 'woman', 'cousin', 'deaf', 'no', 'who', 'help'] 
    # # if False:
    # #     frames, rgb_frame, results_list, results, frame_path, holistic, idx = print_frames(classes, mp_holistic)


    # print("Create poses files...")
    # frame_dir = os.path.join(path, 'frames_wlasl_10')
    # poses_path = os.path.join(path, 'poses')
    # # Create poses files
    # if not os.path.exists(poses_path) and os.path.exists(frame_dir):
    #     os.makedirs(poses_path, exist_ok=True)
    #     print(f'Creating pose files at {poses_path}...')
    #     frames = sorted(os.listdir(frame_dir))
    #     for count, class_name in tqdm(enumerate(frames), total=len(frames)):
    #         print(f'\t{count}: {class_name}')
    #         class_path = f'{frame_dir}/{class_name}'

    #         os.makedirs(f'{poses_path}/{class_name}', exist_ok=True)

    #         video_name_list = os.listdir(class_path)

    #         for video in tqdm(video_name_list):
    #             video_path = f'{class_path}/{video}'

    #             frame_list = os.listdir(video_path)
    #             os.makedirs(f'{poses_path}/{class_name}/{video}', exist_ok=True)

    #             with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #                 for frame in frame_list:
    #                     frame_path = f'{class_path}/{video}/{frame}'
    #                     extracted_path = f'{poses_path}/{class_name}/{video}/{frame.split(".")[0].split("_")[-1]}'
    #                     bgr_frame, results = mediapipe_detection(frame_path, holistic) #hands.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    #                     if not results:
    #                         continue
    #                     landmarks = extract_landmarks(results)
    #                     # rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    #                     np.save(extracted_path, landmarks)
    # else:
    #     print(f'POSES DIR exists: {poses_path}')


    # graphs_path = os.path.join(path, 'graphs')

    # import sys
    # if not os.path.exists(graphs_path) or len(os.listdir(graphs_path)) < len(os.listdir(frame_dir)):
    #     os.makedirs(graphs_path, exist_ok=True)
    #     print(f'Creating pose files at {graphs_path}...')

    #     frames = sorted(os.listdir(frame_dir))[len(os.listdir(graphs_path))-1:-1]
    #     for count, class_name in tqdm(enumerate(frames), total=len(frames)):
    #         print(f'\t{count}: {class_name}')
    #         class_path = f'{frame_dir}/{class_name}'

    #         os.makedirs(f'{graphs_path}/{class_name}', exist_ok=True)

    #         video_name_list = os.listdir(class_path)

    #         for video in video_name_list:
    #             video_path = f'{class_path}/{video}'

    #             frame_list = os.listdir(video_path)
    #             os.makedirs(f'{graphs_path}/{class_name}/{video}', exist_ok=True)

    #             with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #                 for frame in frame_list:
    #                     frame_path = f'{class_path}/{video}/{frame}'
    #                     extracted_path = f'{graphs_path}/{class_name}/{video}/{frame.split(".")[0].split("_")[-1]}'
    #                     pose_graph = create_pose_graph(mp_holistic, frame_path, with_hands=True)
    #                     # bgr_frame, results = mediapipe_detection(cv2.imread(frame_path), holistic) #hands.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    #                     # landmarks = extract_landmarks(results)
    #                     # rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    #                     # np.save(extracted_path, landmarks)
    #                     if pose_graph is None:
    #                         continue

    #                     # save graph object to file
    #                     pickle.dump(pose_graph, open(extracted_path, 'wb'))
