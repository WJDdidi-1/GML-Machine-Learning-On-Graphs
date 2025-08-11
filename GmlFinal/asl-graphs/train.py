import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_gnn as tfgnn

import cv2
import mediapipe as mp
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import random

from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense # fully connected layer
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os

from copy import deepcopy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense # fully connected layer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
from dotmap import DotMap
import yaml
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch import nn
import torch

from wlasl_dataloader import *
from preprocessing.utils import *

# from torch_geometric.nn import GCNConv
# from spektral.layers import GCNConv
from tensorflow_gnn.models.gcn.gcn_conv import GCNConv
import tensorflow_gnn as tfgnn






def plot_confusion_matrix(y_true, y_pred, classes, name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.xticks(rotation=90)
    plt.savefig(f'{name}.png')
    plt.clf()


def create_model(classes, frame_num, total_list_len):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(frame_num,total_list_len)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(np.array(classes).shape[0], activation='softmax'))
    return model

# def create_graph_model(classes, frame_num, total_list_len):
#     model = Sequential()
#     model.add(GCNConv(64, activation='relu')) #([features, adjacency_matrix])
#     model.add(GCNConv(128, activation='relu')) #([model, adjacency_matrix])
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(len(classes)))
#     return model
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow.keras import layers, Model

def update_graph_features(inputs):
    original_graph, updated_features = inputs
    # Replace the node features in the "nodes" set.
    # We store the new features under the default key tfgnn.HIDDEN_STATE.
    return original_graph.replace_features(node_sets={"nodes": {tfgnn.HIDDEN_STATE: updated_features}})
        

def create_frame_model(classes, frame_num, total_list_len, graph_spec):
    input_graph = tf.keras.Input(type_spec=graph_spec)

    graph = input_graph

    # GCNConv layers (Note: requires feature keys to be defined)
    graph = GCNConv(64, activation='relu', node_feature="pos")(
        graph, edge_set_name="edges"
    )

    # graph = tf.keras.layers.Lambda(update_graph_features)([input_graph, graph])
    # graph = GCNConv(128, activation='relu', node_feature=tfgnn.HIDDEN_STATE)(
    #     graph, edge_set_name="edges"
    # )

    # Final dense layers on node features
    # node_states = graph.node_sets["nodes"][tfgnn.HIDDEN_STATE]
    # graph = layers.Dense(128, activation='relu')(graph)

    # x = layers.Dense(64, activation='relu')(x)
    # outputs = layers.Dense(len(classes))(x)

    return Model(inputs=input_graph, outputs=graph, name="frame_model")

def create_graph_model(classes, frame_num, total_list_len, graph_spec):
    frame_inputs = [
        tf.keras.Input(type_spec=graph_spec, name=f"frame_input_{i}")
        for i in range(20)
    ]

    frame_model = create_frame_model(classes, frame_num, total_list_len, graph_spec)

    # Process each frame individually.
    frame_outputs = [frame_model(frame_input) for frame_input in frame_inputs]
    
    # Average the 20 frame outputs.
    # averaged = layers.Average(name="frame_average")(frame_outputs)
    
    # Final dense layers.
    # x = layers.Dense(64, activation='relu')(averaged)
    # outputs = layers.Dense(len(classes))(x)
    
    return Model(inputs=frame_inputs, outputs=frame_outputs, name="graph_model")

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos])
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def calculate_total_len(pose_type):
    if pose_type == 'no_pose':
        return FACE_LIST_LEN*3 + LEFT_LIST_LEN*3 + RIGHT_LIST_LEN*3 
    elif pose_type == 'no_face':
        return LEFT_LIST_LEN*3 + RIGHT_LIST_LEN*3 + POSE_LIST_LEN*4 
    elif pose_type == 'no_hands':
        return FACE_LIST_LEN*3 + POSE_LIST_LEN*4 
    elif pose_type == 'only_hands':
        return FACE_LIST_LEN*3 + LEFT_LIST_LEN*3 + RIGHT_LIST_LEN*3 + POSE_LIST_LEN*4 
    else:
        return TOTAL_LIST_LEN

def train_tensor(classes, sample_num, total_list_len, config, X_train, y_train, tb_callback, X_test, y_test):
    model = create_model(classes, sample_num, total_list_len)
    if config.load_file != '':
        model.load_weights(config.load_file)
    
    # Set up optimization
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Run and train the model
    model.fit(X_train, y_train, epochs=config.epochs, callbacks=[tb_callback])
    
    # Make predictions
    res = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).astype(int).tolist()
    yhat = np.argmax(res, axis=1).astype(int).tolist()
    accuracy = accuracy_score(ytrue, yhat)
    print(f'Accuracy Score: {accuracy*100}%')
    
    # Print the confusion matrix
    plot_confusion_matrix(ytrue, yhat, np.array(classes), 'confusion')
    
    print(f'{(np.sum([classes[np.argmax(res[idx])] == classes[np.argmax(y_test[idx])] for idx in range(len(res))])/len(res))*100:02f}%')
    return res, accuracy

def train_graphs(graph_spec, classes, sample_num, total_list_len, config, X_train, y_train, tb_callback, X_test, y_test):

    model = create_frame_model(classes, sample_num, total_list_len, graph_spec)
    if config.load_file != '':
        model.load_weights(config.load_file)

    # Set up optimization
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Run and train the model
    # import pdb; pdb.set_trace()
    x_list = []
    for x in X_train:
        for i, e in enumerate(x):
            
            e.name = f"frame_input_{i}"
            x_list.append(e)
            break
    # X_train = X_train.reshape(-1, 56, 56)
    # y_train = np.repeat(y_train, 20, 0)
    import pdb; pdb.set_trace()
    model.fit(x_list, y_train, epochs=config.epochs, callbacks=[tb_callback])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='config/asl.yaml')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config = DotMap(config)
    seed = int(config.seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  # For GPU operations
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    random.seed(seed)
    np.random.seed(seed) 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the class map
    classes, label_map = load_classes(config.config_file)

    # Set up Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    if config.debug:
        frames, rgb_frame, results_list, results, frame_path, holistic, idx = print_frames(classes, mp_holistic)


    frames_path = os.path.join(config.path, 'frames_wlasl_10')
    poses_path = os.path.join(config.path, 'poses')
    graphs_path = os.path.join(config.path, 'graphs')
    # Create poses files
    if not os.path.exists(poses_path):
        os.makedirs(poses_path, exist_ok=True)
        print(f'Creating pose files at {poses_path}...')
        for count, class_name in enumerate(os.listdir(frames_path)):
            print(f'\t{count}: {class_name}')
            class_path = f'{frames_path}/{class_name}'

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
                        bgr_frame, results = mediapipe_detection(cv2.imread(frame_path), holistic) #hands.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
                        landmarks = extract_landmarks(results)
                        # rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        np.save(extracted_path, landmarks)

    # Create a label map
    print(f'Label Map: {label_map}')

    total_list_len = calculate_total_len(config.pose_type)
    print(f'Total possible len: {TOTAL_LIST_LEN}, small total len: {total_list_len}')


    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    if config.train_type == 'lstm':
        sequences, labels, res = collect_pose_data(classes, poses_path, total_list_len, config)
        
        y = to_categorical(labels).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(np.array(sequences), y, test_size=config.test_size, random_state=config.seed)

        print(np.array(sequences).shape)
        print(np.array(labels).shape)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        res, accuracy = train_tensor(classes, config.sample_num, total_list_len, config, X_train, y_train, tb_callback, X_test, y_test)
    elif config.train_type == 'graphs':
        sequences, labels = collect_graph_data(classes, graphs_path, total_list_len, config)
        
        y = to_categorical(labels).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(np.array(sequences), y, test_size=config.test_size, random_state=config.seed)

        print(np.array(sequences).shape)
        print(np.array(labels).shape)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        
        res, accuracy = train_graphs(graph_spec, classes, config.sample_num, total_list_len, config, X_train, y_train, tb_callback, X_test, y_test)
        print(accuracy)

if __name__ == '__main__':
    main()