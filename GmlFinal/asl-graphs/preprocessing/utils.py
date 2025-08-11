from tensorflow.keras.callbacks import TensorBoard

import os
from tensorflow.python.client import device_lib
from dotmap import DotMap
import yaml
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch import nn
import torch
import cv2

import networkx as nx

import numpy as np

FACE_LIST_LEN = 468
LEFT_LIST_LEN = 21
RIGHT_LIST_LEN = 21
POSE_LIST_LEN = 33
TOTAL_LIST_LEN = FACE_LIST_LEN*3 + LEFT_LIST_LEN*3 + RIGHT_LIST_LEN*3 + POSE_LIST_LEN*4

def mediapipe_detection(file_name, model):
    try:
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results  = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except:
        print(f'Error: File {file_name} does not exist')
        return None, None
    return image, results

def draw_styled_landmarks(image, results, mp_drawing, mp_holistic):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_landmarks(results):
    pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(POSE_LIST_LEN*4)
    face = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.face_landmarks.landmark]).flatten()  if results.face_landmarks else np.zeros(FACE_LIST_LEN*3)
    left = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(LEFT_LIST_LEN*3)
    right = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(RIGHT_LIST_LEN*3)
    return np.concatenate([pose, face, left, right])


def print_frames(classes, mp_holistic, num_classes=10, frames_path = '/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/frames_wlasl_10'):
    plt.figure(figsize=(20,20))
    random_range = random.sample(range(len(classes)), num_classes)
    
    frames = []
    results_list = []
    print('Printing frames...')
    for counter, selected_class_name in enumerate(classes):
        print(f'{counter}: {selected_class_name}')
        video_files_names_list = os.listdir(os.path.join(frames_path, selected_class_name))
    
        selected_video_file_name = random.choice(video_files_names_list)
    
        video_frame_path = os.path.join(frames_path, selected_class_name, selected_video_file_name)
        video_frame_list = os.listdir(video_frame_path)
    
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            idx = 0
            # print(video_frame_list[0])
            frame_path = os.path.join(frames_path, selected_class_name, selected_video_file_name, video_frame_list[0])
            # print(frame_path)
            # bgr_frame = cv2.imread(frame_path)
            bgr_frame, results = mediapipe_detection(frame_path, holistic)
            frames.append(bgr_frame)
            results_list.append(results)
    
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        cv2.putText(rgb_frame, selected_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        plt.subplot(5, 4, counter+1); plt.imshow(rgb_frame); plt.axis('off')
    
    print('Saving figure...')
    plt.savefig('frame_examples.png')

    image = print_graphs(frames, rgb_frame, results_list, mp_drawing, mp_holistic)

    face_list_len = len(results.face_landmarks.landmark) if results.face_landmarks else 0
    left_list_len = len(results.left_hand_landmarks.landmark) if results.left_hand_landmarks else 0
    right_list_len = len(results.right_hand_landmarks.landmark) if results.right_hand_landmarks else 0
    pose_list_len = len(results.pose_landmarks.landmark) if results.pose_landmarks else 0

    print(face_list_len, left_list_len, right_list_len, pose_list_len)

    for landmark in results.pose_landmarks.landmark:
        print(landmark.x)
        print(landmark.y)
        print(landmark.z)
        print(landmark.visibility)
        break
    
    landmarks = extract_landmarks(results)
    print(len(landmarks), TOTAL_LIST_LEN)
    print(landmarks)

    return frames, rgb_frame, results_list, results, frame_path, holistic, idx

def print_graphs(frames, rgb_frame, results_list, mp_drawing, mp_holistic):
    rgb_frame = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    plt.imshow(rgb_frame)
    plt.axis('off')
    plt.show()
    
    image = deepcopy(rgb_frame)
    
    draw_styled_landmarks(image, results_list[0], mp_drawing, mp_holistic)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return image

def create_pose_graph(holistic, mp_holistic, save_path, with_hands=True):
    image, results = mediapipe_detection(save_path, holistic)
    
    if results is None or results.pose_landmarks is None:
        return None

    ignore_face = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ignore_hands = [18, 20, 22, 17, 19, 21]
                    # 25, 27, 29, 31, 14, 16, 23]
    left_wrist = 15
    right_wrist = 16

    max_i = 0
    pose_G = nx.DiGraph()
    # first add nodes from landmarks
    for i, landmark in enumerate(results.pose_landmarks.landmark):
        if i not in ignore_face and i not in ignore_hands:
            if i == left_wrist or i == right_wrist:
                continue
            if landmark.visibility < 0.2:
                pose_G.add_node(i, pos=(0, 0, 0), vis=False)
            else:
                pose_G.add_node(i, pos=(landmark.x, landmark.y, landmark.z), vis=True)
    max_i = i+1

    for connection in mp_holistic.POSE_CONNECTIONS:
        if connection[0] in pose_G.nodes() and connection[1] in pose_G.nodes():
            if pose_G.nodes[connection[0]]['vis'] and pose_G.nodes[connection[1]]['vis']:
                pose_G.add_edge(connection[0], connection[1])

    # nx.draw(pose_G, with_labels=True, node_color='skyblue', edge_color='gray')

    if with_hands:
        if results.left_hand_landmarks is not None:
            left_elbow = 13
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                if i == 0:
                    pose_G.add_edge(left_elbow, max_i)
                    # print(landmark.x, landmark.y, landmark.z)
                # if i != 0:
                pose_G.add_node(max_i+i, pos=(landmark.x, landmark.y, landmark.z), vis=True)

            # hand has 21 nodes so the nodes are now 33+0, 33+1, 33+2, ... 33+20
            # because the pose has 32 nodes 
            for connection in mp_holistic.HAND_CONNECTIONS:
                # if connection[0] == 0:
                #     pose_G.add_edge(15, connection[1]+max_i)
                # elif connection[1] == 0:
                #     pose_G.add_edge(connection[0]+max_i, 15)
                # else:
                pose_G.add_edge(connection[0]+max_i, connection[1]+max_i)
            max_i += i+1
        else:
            for i in range(LEFT_LIST_LEN):
                pose_G.add_node(max_i+i, pos=(0,0,0), vis=False)

            # hand has 21 nodes so the nodes are now 33+0, 33+1, 33+2, ... 33+20
            # because the pose has 32 nodes 
            for connection in mp_holistic.HAND_CONNECTIONS:
                if connection[0]+max_i in pose_G.nodes() and connection[1]+max_i in pose_G.nodes():
                    if pose_G.nodes[connection[0]+max_i]['vis'] and pose_G.nodes[connection[1]+max_i]['vis']:
                        pose_G.add_edge(connection[0]+max_i, connection[1]+max_i)
            max_i += i+1
        
        if results.right_hand_landmarks is not None:
            right_elbow = 14
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                if i == 0:
                    pose_G.add_node(max_i+i, pos=(landmark.x, landmark.y, landmark.z))
                    pose_G.add_edge(right_elbow, max_i+i)
                else:
                    # print(landmark.x, landmark.y, landmark.z)
                    pose_G.add_node(max_i+i, pos=(landmark.x, landmark.y, landmark.z))

            # hand has 53 nodes so the nodes are now 53+0, 53+1, 53+2, ... 53+20
            # because the pose has 32 nodes and left hand had 21
            for connection in mp_holistic.HAND_CONNECTIONS:
                # if connection[0] == 0:
                #     pose_G.add_edge(16, connection[1]+max_i)
                # elif connection[1] == 0:
                #     pose_G.add_edge(connection[1]+max_i, 16)
                # else:
                pose_G.add_edge(connection[0]+max_i, connection[1]+max_i)
        else:
            for i in range(RIGHT_LIST_LEN):
                pose_G.add_node(max_i+i, pos=(0,0,0), vis=False)

            # hand has 21 nodes so the nodes are now 33+0, 33+1, 33+2, ... 33+20
            # because the pose has 32 nodes 
            for connection in mp_holistic.HAND_CONNECTIONS:
                if connection[0]+max_i in pose_G.nodes() and connection[1]+max_i in pose_G.nodes():
                    if pose_G.nodes[connection[0]+max_i]['vis'] and pose_G.nodes[connection[1]+max_i]['vis']:
                        pose_G.add_edge(connection[0]+max_i, connection[1]+max_i)
            max_i += i+1

    pose_G = pose_G.to_undirected()
    # if len(pose_G) != 56:
    #     import pdb; pdb.set_trace()
    return pose_G