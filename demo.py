# -*- coding: utf-8 -*-
# @Software: PyCharm
# @description: 
# @Author: KevinW
# @Time: Jul 21, 2022

import numpy as np
import torch
from src.stereonet.model import StereoNet
from src.stereonet import utils as utils

# Load in the image pair as numpy uint8 arrays
# sample = {'left': utils.image_loader(path_to_left_rgb_image_file),
#           'right': utils.image_loader(path_to_right_rgb_image_file)
#           }

# Here just creating a random image
rng = np.random.default_rng()
sample = {'left': (rng.random((540, 960, 3))*255).astype(np.uint8),  # [height, width, channel],
          'right': (rng.random((540, 960, 3))*255).astype(np.uint8)  # [height, width, channel]
          }

# Transform the single image pair into a torch.Tensor then into a
# batch of shape [batch, channel, height, width]
transformers = [utils.ToTensor(), utils.PadSampleToBatch()]
for transformer in transformers:
    sample = transformer(sample)

# Load in the model from the trained checkpoint
# model = StereoNet.load_from_checkpoint(path_to_checkpoint)

# Here just instantiate the model with random weights
model = StereoNet()
#model = StereoNet.load_from_checkpoint(str(checkpoint_path))

# Set the model to eval and run the forward method without tracking gradients
model.eval()
with torch.no_grad():
    batched_prediction = model(sample)

# Remove the batch diemnsion and switch back to channels last notation
single_prediction = batched_prediction[0].numpy()  # [batch, ...] -> [...]
single_prediction = np.moveaxis(single_prediction, 0, 2)  # [channel, height, width] -> [height, width, channel]

single_prediction.shape

