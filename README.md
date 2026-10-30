# GraphTALK — ASL recognition from skeleton graphs (GNN)

American Sign Language recognition from **hand/pose skeletons instead of RGB
pixels**: MediaPipe extracts keypoints per frame, the clip becomes a graph, and
a graph neural network classifies the sign. Skeleton input removes background
and lighting from the problem entirely, which is why this route is worth trying
on a 24-class WLASL subset.

Course project for *Machine Learning on Graphs* (UVA). The code is the artifact;
accuracy numbers from the report are not reproduced here, and this README does
not claim any.

## Pipeline

```
WLASL videos ──► preprocessing/download_wasl.py   (fetch clips)
             ──► preprocessing/extract.py          (frames → MediaPipe keypoints)
             ──► wlasl_dataloader.py               (windows of frames → PyG graphs)
             ──► gcn.py / train.py                 (GCN / temporal GNN)
             ──► test.py                           (validation, confusion matrix)
```

- `config/asl.yaml` — the run config: 64×80 frames, `pose_type: no_face` (drop
  face landmarks, keep pose + hands), 10 samples per video, 500 epochs,
  `train_type: graphs` (the `lstm` branch is also wired).
- `gcn.py` — `GCNConv` layers with `global_mean_pool`: the spatial-only path.
- `train.py` — the temporal path: `tensorflow_gnn` over the same graph inputs.
- `wlasl_dataloader.py` — builds `InMemoryDataset` windows from extracted
  keypoints; `vids_per_label.txt` is the class-count audit from preparation.
- `graph_pred.py`, `test.py`, `utils.py`, `lr_scheduler.py` — inference,
  validation/metrics, helpers, warmup schedules.

## Running it

```console
# 1. environment (full conda line in GmlFinal/asl-graphs/README.md)
$ conda create --name bound -y && conda activate bound
$ conda install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
$ python -m pip install 'numpy<2' pandas opencv-python mediapipe \
      torch-geometric tensorflow tensorflow-gnn pyyaml dotmap tqdm scikit-learn

# 2. data
$ python preprocessing/download_wasl.py      # WLASL clips
$ python preprocessing/extract.py            # frames → keypoints

# 3. train / evaluate
$ python train.py --config config/asl.yaml
$ python test.py
```

A CUDA GPU is required in practice: 500 epochs × 10 samples per clip is not a
laptop workload. `config/asl.yaml` still holds absolute paths from the machine
this ran on (`path:` and `config_file:` point at a Desktop folder) — edit those
two lines before a run.

## What is and is not in here

- **In here**: the pipeline, both model paths (spatial GCN, temporal GNN), the
  class lists, the preparation audit files, and the exploratory notebooks
  (`notebook/mediapipe_pose.ipynb`, `graph_pred.ipynb`, `new_pred.ipynb`).
- **Not in here**: WLASL videos and extracted frames (too large), checkpoints,
  and the report. No accuracy number is claimed, because none can be reproduced
  from this checkout without the dataset.

## Limits worth naming

- 24 classes with a small per-class sample count: a feasibility study, not a
  deployable recogniser.
- `train.py` and `gcn.py` sit on different stacks (TensorFlow-GNN vs
  PyTorch-Geometric). Comparing them was the point of the experiment, but the
  two entry points need different environments.
- MediaPipe keypoints are noisy under occlusion, and the graph drops confidence
  information.
- The repository root previously carried a README describing a project called
  GraphTALK with a `git clone …/yourusername/GraphTALK.git` placeholder; the
  code lives under `GmlFinal/asl-graphs/`, which is what this README documents.

## License

No license file, so the default applies: all rights reserved.
