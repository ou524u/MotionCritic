# Motion Documentation

This document provides a detailed description of the structure and usage of motion `.npz` files obtained from the outputs of MDM and FLAME pretrained models, which were collected for annotation purposes. Each motion `.npz` file corresponds to the data representation behind a specific video with four choices (A, B, C, D), from which annotations need select. In the MDM model, the objective is to select the best motion, whereas in the FLAME model, the objective is to choose the worst motion. Annotation results are then used to create better-worse pairs and packed into dataset. Refer to `dataset.md` for the next part.

**Note:** Due to the lower data quality of FLAME compared to MDM, FLAME data was not utilized for training the critic model. Instead, a limited number of FLAME samples were selected for testing to evaluate the critic model's generalization ability.

## Structure of the Motion .npz Files

Each motion `.npz` file contains motion data structured as follows:

### Keys in the .npz File
- `['arr_0']`: This key holds a dictionary containing detailed motion data and prompts.

### Contents of `npz['arr_0']`
- **Key: `prompt`**
  - **Type**: `<class 'list'>`
  - **Description**: A list containing 4 prompts, each corresponding to one of the ABCD choices.

- **Key: `motion`**
  - **Type**: `<class 'list'>`
  - **Description**: A list containing 4 motions. Each motion is shaped `[batchsize=1, 25, 6, frame=60]`, representing the rot6d representation of SMPL motion. This representation is directly the output of MDM models and is convenient for rendering. To transform this representation into the format required by the critic model, the motions need to be transposed to `[1, 60, 25, 6]` and then processed using the `into_critic` function from `parsedata.py`.

## Usage

To render or transform the motion data into the format required by the critic model, follow these steps:

```python
import numpy as np
from MotionCritic.parsedata import into_critic
from MDMCritic.render.render import render_single

# Load the dataset
motion = np.load('path_to_your_file.npz', allow_pickle=True)

# Access the dictionary inside the npz file
motion_data = motion['arr_0'].item()

# Extract the list of motions
motion_list = motion_data['motion']

# Extract a choice within 4, and render it
motion_A = motion_list[0]
render_single(motion_A, device='cpu', comment='A', file_path='path_to_your_video.mp4')

# Transform each motion for the critic model
transformed_A = into_critic(motion_A.transpose(0, 3, 1, 2))
```

Replace `'path_to_your_file.npz'` with the actual path to your `.npz` file and `'path_to_your_video.mp4'` with the actual path to your `.mp4` file where the video will be saved.

This document provides a comprehensive overview of the structure and usage of the motion `.npz` files, ensuring effective utilization of the data for your tasks.