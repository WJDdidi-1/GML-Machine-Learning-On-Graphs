from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import pickle
from torch_geometric.utils import from_networkx
import networkx
import random
import torch
from torch_geometric.data import Data


def collect_pose_data(classes, poses_path, total_list_len, config, label_map):
    sequences, labels = [], []
    pose_lens = []
    for count, class_name in enumerate(classes):
        class_path = f'{poses_path}/{class_name}'
    
        video_name_list = os.listdir(class_path)
    
        for vid_cnt, video in enumerate(video_name_list):
            if (vid_cnt) > 7:
                break
            window = []
            video_path = f'{class_path}/{video}'
    
            poses_list = os.listdir(video_path)
    
    
            window = np.empty((len(poses_list), total_list_len))
            for pose_cnt, pose in enumerate(poses_list):
    
                pose_path = f'{video_path}/{pose}'
                res = np.load(pose_path)
    
                if config.pose_type == 'no_pose':
                    window[pose_cnt] = res[POSE_LIST_LEN*4:]
                elif config.pose_type == 'no_face':
                    window[pose_cnt] = np.concatenate((res[0:POSE_LIST_LEN*4], res[POSE_LIST_LEN*4+FACE_LIST_LEN*3:]))
                elif config.pose_type == 'no_hands':
                    window[pose_cnt] = res[0:-1*(LEFT_LIST_LEN*3+RIGHT_LIST_LEN*3)]
                elif config.pose_type == 'only_hands':
                    window[pose_cnt] = res[-1*(LEFT_LIST_LEN*3+RIGHT_LIST_LEN*3):]
                else:
                    window[pose_cnt] = res
    
            for _ in range(config.samples_per_video):
                # print(pose_cnt, np.array(window).shape)
                random_sequence = sorted(random.sample(range(pose_cnt+1), 20))
                # print(random_sequence)
                new_window = window[random_sequence, :]
                sequences.append(new_window)
                labels.append(label_map[class_name])
    return sequences, labels, res

# ChatGPT
import networkx as nx
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

def nx_to_graph_tensor(G: nx.Graph) -> tfgnn.GraphTensor:
    # Step 1: Create a consistent ordering of nodes.
    nodes = sorted(G.nodes())
    num_nodes = len(nodes)

    # Step 2: Collect node features. Here we assume each node has a "feat" attribute.
    node_feats = []
    for node in nodes:
        feat = G.nodes[node].get("pos")
        if feat is None:
            raise ValueError(f"Node {node} is missing the 'feat' attribute.")
        node_feats.append(feat)
    node_feats = np.array(node_feats, dtype=np.float32)

    # Step 3: Map original node IDs to new indices.
    node_id_to_index = {node: idx for idx, node in enumerate(nodes)}

    # Step 4: Build edge index arrays.
    sources, targets = [], []
    for source, target in G.edges():
        sources.append(node_id_to_index[source])
        targets.append(node_id_to_index[target])
    sources = np.array(sources, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    # Step 5: Create the NodeSet.
    node_set = tfgnn.NodeSet.from_fields(
        sizes=tf.constant([num_nodes], dtype=tf.int32),
        features={"pos": tf.convert_to_tensor(node_feats)}
    )

    # Create the EdgeSet using an Adjacency object.
    adjacency = tfgnn.Adjacency.from_indices(
        source=("nodes", tf.convert_to_tensor(sources)),
        target=("nodes", tf.convert_to_tensor(targets))
    )
    edge_set = tfgnn.EdgeSet.from_fields(
        sizes=tf.constant([len(sources)], dtype=tf.int32),
        adjacency=adjacency,
        features={}  # If you have edge features, add them here.
    )

    # Step 6: Build the GraphTensor.
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={"nodes": node_set},
        edge_sets={"edges": edge_set}
    )
    return graph_tensor

# # Example usage:
# if __name__ == "__main__":
#     # Create a simple NetworkX graph.
#     G = nx.Graph()
#     # Add nodes with a dummy 4-dimensional feature.
#     G.add_node(1, feat=[0.1, 0.2, 0.3, 0.4])
#     G.add_node(2, feat=[0.5, 0.6, 0.7, 0.8])
#     G.add_node(3, feat=[0.9, 1.0, 1.1, 1.2])
#     # Add some edges.
#     G.add_edge(1, 2)
#     G.add_edge(2, 3)

#     # Convert to a GraphTensor.
#     graph_tensor = nx_to_graph_tensor(G)
#     print(graph_tensor)


def collect_graph_data(classes, graphs_path, config, label_map):
    sequences, labels = [], []
    pose_lens = []
    for count, class_name in enumerate(classes):
        class_path = os.path.join(graphs_path, class_name)
    
        video_name_list = os.listdir(class_path)
    
        for vid_cnt, video in enumerate(video_name_list):
            if (vid_cnt) > 7:
                break
            window = []
            video_path = os.path.join(class_path, video)
    
            poses_list = os.listdir(video_path)
    
    
            graphs = []
            for pose_cnt, pose in enumerate(poses_list):
    
                final_path = os.path.join(video_path, pose)
                # load graph object from file
                G = pickle.load(open(final_path, 'rb'))
                nx_flag = False
                x = torch.tensor([list(G.nodes()[node].get("pos")) for node in G.nodes()], dtype=torch.float)

                edges = []
                for source, destination in G.edges():
                    # torso, legs, left hand, right hand
                    # 11-14 -> 0-3, 23-32 -> 4-13, 33-53 -> 14-34, 54-74 -> 35-55
                    if source >= 11 and source <= 14:
                        source = source - 11
                        destination = destination - 11
                    elif source >= 23 and source <= 74:
                        source = source - 19
                        destination = destination - 19
                    edges.append((source, destination))
                edge_index=torch.tensor(list(edges))

                # import pdb; pdb.set_trace()
                data = Data(x=x, edge_index=edge_index.t().contiguous(), y=count)
                if nx_flag:
                    graphs.append(nx_to_graph_tensor(G))
                else:
                    graphs.append(data)
    
            window = graphs #= np.array(graphs) #[networkx.to_numpy_array(g).astype(np.float32) for g in graphs])
            # else:
            #     import pdb; pdb.set_trace()
            #     window = np.array([torch.tensor(networkx.to_numpy_array(g).astype(np.float32)) for g in graphs])
            # features = np.array([np.array([node[1]['pos'] for node in g.nodes(data=True)]) for g in graphs])

            for _ in range(config.samples_per_video):
                # print(pose_cnt, np.array(window).shape)
                # random_sequence = sorted(random.sample(range(pose_cnt+1), 20))
                random_sequence = sorted(np.random.choice(range(pose_cnt+1), size=20, replace=True))
                
                # print(random_sequence)
                new_window = [window[rs] for rs in random_sequence]
                sequences.append(new_window)
                labels.append(label_map[class_name])

    if nx_flag:
        return sequences, labels, graphs[-1].spec
    else:
        return sequences, labels
        
def load_classes(file_name):
    classes = []
    classes_dict = {}
    with open(file_name) as f:
        for i, line in enumerate(f):
            if i > 0:
                class_idx, class_name = line.strip().split(',')
                classes.append(class_name)
                classes_dict[class_name] = int(class_idx)-1

    return classes, classes_dict

class WLASL(Dataset):
    def __init__(self, poses_path, total_list_len, config, train=True):
        classes, label_map = load_classes(config.config_file)

        sequences, labels, res = collect_pose_data(classes, poses_path, total_list_len, config, label_map)

        y = to_categorical(labels).astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.array(sequences), y, test_size=config.test_size, random_state=config.seed)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, i):
        return self.X_train[i], self.y_train[i]