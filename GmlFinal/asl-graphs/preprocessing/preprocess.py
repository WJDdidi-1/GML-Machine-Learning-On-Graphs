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
import threading

import numpy as np
import pickle
import sys

from utils import *
from reorg import *

def get_poses(poses_path, frame_dir, mp_holistic):
    os.makedirs(poses_path, exist_ok=True)
    print(f'Creating pose files at {poses_path}...')
    frames = sorted(os.listdir(frame_dir))
    for count, class_name in tqdm(enumerate(frames), total=len(frames)):
        print(f'\t{count}: {class_name}')
        class_path = f'{frame_dir}/{class_name}'
    
        os.makedirs(f'{poses_path}/{class_name}', exist_ok=True)
    
        video_name_list = os.listdir(class_path)
    
        for video in tqdm(video_name_list):
            video_path = f'{class_path}/{video}'
    
            frame_list = os.listdir(video_path)
            os.makedirs(f'{poses_path}/{class_name}/{video}', exist_ok=True)
    
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for frame in frame_list:
                    frame_path = f'{class_path}/{video}/{frame}'
                    extracted_path = f'{poses_path}/{class_name}/{video}/{frame.split(".")[0].split("_")[-1]}'
                    bgr_frame, results = mediapipe_detection(frame_path, holistic) #hands.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
                    if not results:
                        continue
                    landmarks = extract_landmarks(results)
                    # rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    np.save(extracted_path, landmarks)
    return landmarks

def extract_frame(class_path, video, frame, graphs_path, class_name, holistic, mp_holistic):
    frame_path = f'{class_path}/{video}/{frame}'
    extracted_path = f'{graphs_path}/{class_name}/{video}/{frame.split(".")[0].split("_")[-1]}'
    pose_graph = create_pose_graph(holistic, mp_holistic, frame_path, with_hands=True)
    
    if pose_graph is None:
        return
    
    # save graph object to file
    pickle.dump(pose_graph, open(extracted_path, 'wb'))


def get_graphs(graphs_path, frame_dir, mp_holistic, start, end):
    os.makedirs(graphs_path, exist_ok=True)
    print(f'Creating graph files at {graphs_path}...')
    
    frames = sorted(os.listdir(frame_dir))
    if start > 0 or end > -1:
        frames = frames[start:end]
    
    for count, class_name in tqdm(enumerate(frames), total=len(frames)):
        print(f'\t{count}: {class_name}')
        class_path = f'{frame_dir}/{class_name}'
    
        os.makedirs(f'{graphs_path}/{class_name}', exist_ok=True)
    
        video_name_list = sorted(os.listdir(class_path))
    
        for video in video_name_list:
            video_path = f'{class_path}/{video}'
    
            frame_list = os.listdir(video_path)
            os.makedirs(f'{graphs_path}/{class_name}/{video}', exist_ok=True)
    
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for frame in tqdm(frame_list, total=len(frame_list)):
                    extract_frame(class_path, video, frame, graphs_path, class_name, holistic, mp_holistic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses', type=bool, default=True)
    parser.add_argument('--graphs', type=bool, default=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    path = '/Users/felixwjd/Desktop/GmlFinal/asl-graphs'

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    print("Create poses files...")
    frame_dir = os.path.join(path, 'frames_wlasl_10')

    if args.poses:
        # Create poses files
        poses_path = os.path.join(path, 'poses')
        if not os.path.exists(poses_path) and os.path.exists(frame_dir) and sys.argv[2]:
            get_poses(poses_path, frame_dir, mp_holistic)

    if args.graphs:
        graphs_path = os.path.join(path, 'graphs')
        if not os.path.exists(graphs_path) or len(os.listdir(graphs_path)) < len(os.listdir(frame_dir)):
            get_graphs(graphs_path, frame_dir, mp_holistic, args.start, args.end)

