# asl-graphs


## Install

```bash
conda create --name bound -y
conda activate bound
conda install pytorch==2.2.1 torchvision==0.17.1  pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -m pip install 'numpy<2'
python -m pip install pandas opencv-python randaugment tqdm wandb dotmap ftfy regex kornia omegaconf matplotlib captum ipython imageio scikit-learn grad-cam einops timm transformers
python -m pip install mediapipe
```

`for dir in */; do echo "$(find "$dir" -mindepth 1 -maxdepth 1 -type d | wc -l) $dir"; done | sort -n > vids_per_label.txt`