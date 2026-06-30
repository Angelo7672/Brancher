# Self-Supervised 3D Human Pose Estimation in Sports Aerial Videos
Human Pose Estimation (HPE) in unconstrained outdoor sports scenarios introduces critical challenges, including severe occlusions, viewpoint diversity, depth ambiguity, and a significant scarcity of annotated 3D aerial data. While Unmanned Aerial Vehicles (UAVs) offer a flexible platform for motion capture, drone-acquired footage suffers from dynamic camera motion and scale changes.  
This repository introduces **Brancher**, a self-supervised neural network architecture designed for single-person, single-view 3D HPE directly from monocular drone videos without relying on 3D ground-truth annotations during the aerial adaptation stage.

The full thesis is available [here](https://www.politesi.polimi.it/handle/10589/253213).

## 🚀 Key Contributions
- **Motion-Aware Representations:** A self-supervised framework initially pretrained on high-speed athletic videos to effectively learn complex sports dynamics.
- **Self-Supervised UAV Fine-Tuning:** An unsupervised domain adaptation strategy explicitly tailored to tackle domain shifts, scale variations, and viewpoint changes inherent to drone tracking.
- **Novel Drone Dataset:** Introduction of a custom video dataset explicitly recorded and curated for outdoor aerial 3D human pose estimation (the dataset is available [here](https://drive.google.com/drive/folders/1fnoo955REZ15jHVQDx3bwAQb0oTVZk_U?usp=sharing), ask for the permission on Google Drive).

## 🎥 Demo Video
Below you can see the qualitative reconstruction capabilities of Brancher on real-world sports aerial footages:
https://github.com/user-attachments/assets/YOUR_VIDEO_ID_HERE

## 📁 Repository Structure
- **baseline_2d_lift/**: Contains the pipeline for the 2D-to-3D lifting baseline model.
- **brancher/:** Houses the main contribution of this thesis. This includes the hybrid Spatial-Temporal Attention blocks of MotionAGFormer and the specialized evaluation outputs for the Pose, Rotation, and Uncertainty branches.  
- **keypoints_module/:** Stores the structured config files to initialize the 2D pose estimators utilized during the bounding-box inference stage.
- **preprocess/:** Includes automated scripts to crop human subjects from raw sports footage and structure the extracted 2D coordinates into normalized trajectories ready for training.  
- **slide_presentation/:** PowerPoint presentation used for the thesis defense.
- **utility/:** A collection of auxiliary plotting and visualization helpers designed to project the predicted root-relative 3D skeletons back onto the video frames for verification.  

## ⚙️ Installation

This project is based on Conda and OpenMMLab. The following steps describe how to set up the environment from scratch.

### 1. Create Conda environment

```bash
conda create -n AR17 python=3.11.6 -c conda-forge -c defaults
conda activate AR17
```

### 2. Install PyTorch with CUDA support

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 3. Install additional dependencies

```bash
conda install av -c conda-forge
```

### 4. Install OpenMIM and YOLO (Ultralytics)

```bash
pip install -U openmim
pip install ultralytics  # YOLO implementation
```

### 5. Install OpenMMLab ecosystem

```bash
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.1"
mim install "mmpretrain==1.2.0"
```

### 6. Update environment from configuration file

```bash
conda env update --file AR17/code/environment.yml
```

## How to Use
Download the trained model checkpoints from this [link](https://drive.google.com/drive/folders/1CqdH_hzRHsSfg3oL1oPGC3Gr_51m99Qs?usp=sharing) and changes the variable *weights_path* with your current path.

Then use this command to run inference using the pre-trained Brancher model:
```bash
python3 brancher/inference.py
```