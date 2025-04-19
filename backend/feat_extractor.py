import torch
from torch import nn
import numpy as np
from tqdm import tqdm
model_name = 'x3d_l'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
from torchinfo import summary
from torchvision.transforms.v2 import CenterCrop, Normalize
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from torch.utils.data import DataLoader
import os.path
import sys

device = "cpu"  # 'cuda'
model = model.eval()
model = model.to(device)

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 30
model_transform_params = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    },
    "x3d_l": {
        "side_size": 320,
        "crop_size": 320,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

transform_params = model_transform_params[model_name]

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)

transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x / 255.0),
            Permute((1, 0, 2, 3)),
            Normalize(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),
            Permute((1, 0, 2, 3))
        ]
    ),
)

clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second

del model.blocks[-1]

summary(model, (1, 3, 16, 320, 320))

if len(sys.argv) > 1:
    video_path = sys.argv[1]
    video_name = os.path.basename(video_path)[:-4]

    test_list = [(video_path, {'label': 0, 'video_label': video_name})]

    dataset = LabeledVideoDataset(
        labeled_video_paths=test_list,
        clip_sampler=UniformClipSampler(clip_duration),
        transform=transform,
        decode_audio=False,
    )

    loader = DataLoader(dataset, batch_size=1)

    label = None
    current = None

    for inputs in tqdm(loader):
        preds = model(inputs['video'].to(device)).detach().cpu().numpy()
        for i, pred in enumerate(preds):
            if inputs['video_label'][i] != label:
                if label is not None:
                    np.save(label + '.npy', current.squeeze())
                label = inputs['video_label'][i]
                current = pred[None, ...]
            else:
                current = np.max(np.concatenate((current, pred[None, ...]), axis=0), axis=0)[None, ...]

    if label is not None:
        np.save(label + '.npy', current.squeeze())

    print(f".npy file created: {video_name}.npy") #added print statement to show the name of the created file.

else:
    print("Please provide the .mp4 video file path as a command-line argument.")