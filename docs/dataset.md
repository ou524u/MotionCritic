# Dataset Documentation

This dataset contains pre-processed better-worse motion pairs stored in a `.pth` file. Each pair in the dataset includes the keys `motion_better` and `motion_worse`, represented as tensors in the following format:

## Structure of Motion Pairs

- **Keys:**
  - `motion_better`: $\mathbf{x_i} \in \mathbb{R}^{60 \times 25 \times 3}$
  - `motion_worse`: $\mathbf{x_j} \in \mathbb{R}^{60 \times 25 \times 3}$

## Tensor Dimensions

- **L (Frames):** 60
- **J (Joints):** 25 (including 24 SMPL joints and one xyz root location)
- **D (Dimensions):** 3 (24 axis-angles, and 1 xyz coordinates)

### Example of a Motion Pair

Each motion pair includes two tensors:
1. `motion_better` ($\mathbf{x_i}$): A tensor representing a motion sequence that is considered better.
2. `motion_worse` ($\mathbf{x_j}$): A tensor representing a motion sequence that is considered worse.

### Data Format

Each tensor is structured as follows:
- **60 frames:** The motion sequence is divided into 60 frames.
- **25 joints:** Each frame contains information for 25 joints (24 SMPL joints + 1 root location).
- **3 dimensions:** Each joint is described using 3-dimensional xyz coordinates.

## Usage

To load the dataset, use the following example code in Python:

```python
import torch

# Load the dataset
data = torch.load('path_to_your_file.pth')

# Access a pair of motions
motion_pair = data['pair_index']  # replace 'pair_index' with the actual index
motion_better = motion_pair['motion_better']
motion_worse = motion_pair['motion_worse']
```

Replace `'path_to_your_file.pth'` with the actual path to your `.pth` file and `'pair_index'` with the index of the motion pair you want to access.

This format ensures the dataset is clearly understood and easy to work with for further processing and analysis.




