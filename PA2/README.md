# PA2: Depth Completion

- **Due:** 19th, December, 2025  
- **TA:** Junmyeong Lee (june65@yonsei.ac.kr)

**NO PLAGIARISM, NO DELAY, DON'T USE AI SUPPORTER**

## Overview

### Before you start, make sure you understand the code flow by reading `main.py`.

The provided code follows these main steps:

1. **Initial Depth Generation (Hole Filling):**  
   *Fill the blank in:* `main.py` (line 18–23)

2. **Building a UNet Architecture:**  
   *Fill the blank in:* `main.py` (line 30–34)

3. **Depth-to-Normal Conversion:**  
   *Fill the blank in:* `main.py` (line 50–59)

4. **Per-Scene Optimization:**  
   *Fill the blank in:* `main.py` (line 65–76)

## Directory Structure

```
. 
data/
 ├── data_example/
 │    ├── gt.npy
 │    ├── normal.npy
 │    ├── rgb.png
 │    └── sparse_depth.npy
 └── data_submission/
      ├── normal.npy
      ├── rgb.png
      └── sparse_depth.npy
```


## Usage

Run the main script:

```
python main.py
```

This will run the entire pipeline:
- Hole filling  
- UNet forward pass  
- Depth-to-normal conversion  
- Per-scene optimization  
- Saving outputs  

## Implementation Steps

### 1. Implement Initial Depth via Hole Filling
- Apply convolution-based filtering to fill empty pixels.  
- Average only valid depth values within a kernel window.  
- Slide the kernel and iterate the process multiple times to obtain the initial dense depth.

### 2. Build a UNet Architecture
- Input channels: **RGB + sparse depth**  
- Output: **1-channel dense depth**  
- You may design a simple UNet; a complex model is not required.  
- Reference papers are allowed, but the architecture must be your own.

### 3. Depth-to-Normal Conversion
- Convert predicted depth into surface normals.  
- Ensure normals face the camera.  
- Normalize to unit vectors.  
- Apply proper padding.  
- Use PyTorch operations to ensure gradients flow during backpropagation.

### 4. Per-Scene Optimization
- Refine predicted depth for each scene using your defined loss functions.  
- You may add additional refinement techniques if desired.

## Output

All outputs must be saved under the `output/` directory:

- **Initial Depth (submission):** Include the image in the report  
- **Final Refined Depth (submission):** Include the image in the report
- **Final Refined Nomal (submission):** Include the image in the report  
- **Refined depth (submission) `.npy` file:**  
  ```
  output/refined_depth.npy
  ```
- **Code:** `main.py`  
  saved in  
  ```
  output/code/
  ```
- **Report:** Detailed explanation of your method  
  saved in  
  ```
  output/report/
  ```

Upload the entire `output/` folder to LearnUS.
