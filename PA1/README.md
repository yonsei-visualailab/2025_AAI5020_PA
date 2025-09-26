# PA1: NetVLAD

- Due: 19th, October, 2025
- TA: Junmyeong Lee (june65@yonsei.ac.kr)

**NO PLAGIARISM, NO DELAY, DON'T USE AI SUPPORTER**

## Overview

### Before you start, make sure you understand the code flow by reading base.py

The provided code follows these main steps:
1. **VGG16-based NetVLAD Structure :**  

   *Fill the blank in:* `model.py`

2. **Learning with Ranking Loss :**  
   
   *Fill the blank in:* `train.py`

3. **Evaluation with Recall@K :**  
   *Fill the blank in:* `eval.py`

5. **Camera Pose Estimation :**  
   *Fill the blank in* `test.py`

## Data Download

You can download the dataset archives from the links below:

- **dataset.tar** (main dataset archive):  
  https://drive.google.com/open?id=1a5f2wX40fFcqGfxzEiA9aJWW8bnOwBlT&usp=drive_copy  
- **dataset.zip** (supplementary archive):  
  https://drive.google.com/open?id=1kdy6Q_ziso-LdYctOo6JqR3_zDEcY7dE&usp=drive_copy  

After downloading, extract both archives into the `dataset/` folder so that your project directory matches the structure below.

## Directory Structure
```
. 
├── dataset/
│ ├── camera/
│ ├── gt/
│ ├── index/
│ ├── query/
│ ├── index_test.txt
│ ├── ...
│ └── query_train.txt     
├── base.py               
├── eval.py    
├── model.py   
├── test.py       
└── train.py             
```

## Requirements

- **Python Version:** 3.7 or above
- **Libraries:**  
  - NumPy
  - pillow
  - tqdm
  - torch
  - torchvision

You can install the required libraries using pip:

```
pip install numpy pillow tqdm torch torchvision
```

## Usage
Run the main script from the command line. The script accepts the following

```
python base.py
```

```
python test.py
```

## Implementation Steps

1. Build a VGG16-based NetVLAD Structure :
   - Pass the input image through a VGG16 CNN backbone to obtain a high-dimensional feature map.
   - Perform soft-assignment of local features to cluster centroids.
   - Aggregate the residual vectors into a global descriptor.

2. Implement Learning with Ranking Loss :
   - Define a triplet loss that enforces:
        - Query vectors are closer to positive samples (same class/scene).
        - Query vectors are farther from negative samples (different class/scene).
   - The ranking loss is implemented using a margin-based formulation.

3. Evaluation with Recall@K :
   - Extract embeddings for query and database images.
   - Compute cosine similarity between embeddings.
   - For each query, check whether at least one true positive appears in the top-K retrieved results (K = 1, 5, 10).
   - Compute recall@K scores to evaluate retrieval performance.

4. Measure Performance of Camera Pose Estimation :
   - Each scene contains around 20–30 images.
   - Some query images have their camera poses removed.
   - Using retrieval and matching information, estimate the camera poses of these query images.
   - Evaluate how accurately the system recovers the missing poses.
   - You can add any ideas

## Output:

- NetVLAD model: Saved as `output/checkpoints/netvlad_final.pth`
- Evaluation Value: Saved the capture of result of task3 `output/task3_result.png`
- Camera Pose Estimation: Saved the result of task4 `output/result_pose_eval.txt`
- Code: base.py, eval.py, model.py, test.py, train.py → `output/`
- Report: Write report on your implementation `output/report`

All outputs are saved in the output/ directory.

You should up load the output file to LearnUS