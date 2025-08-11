# GraphTALK: ASL Sign Language Recognition using Graph Neural Networks

## 项目背景
美国手语（ASL）识别在背景复杂、光照变化和手势多样性方面存在较大挑战。传统基于图像的识别方法容易受到背景干扰，且对手势动态变化的捕捉不足。

## 项目目标
利用图神经网络（GNN）对 Mediapipe 提取的骨骼点数据进行时序建模，从而在无需依赖 RGB 背景的情况下实现高精度 ASL 手语识别。

## 技术栈
- **编程语言**: Python
- **框架**: PyTorch Geometric
- **数据处理**: Mediapipe Pose & Hands
- **数据集**: WLASL 子集（24 类手势）

## 主要功能
- Mediapipe 提取骨骼关键点并生成图结构数据
- Temporal Graph Network 建模时序信息
- 分类 24 类 ASL 手势

## 你的贡献
- 设计并实现 GNN 模型结构
- 构建数据处理 pipeline（骨骼点 → 图结构 → 模型输入）
- 优化模型参数，提升分类精度与鲁棒性

## 成果
- 在 WLASL 子集上获得较高准确率，并在不同背景条件下保持稳定性能

## 使用方法
```bash
git clone https://github.com/yourusername/GraphTALK.git
cd GraphTALK
pip install -r requirements.txt
python train.py
