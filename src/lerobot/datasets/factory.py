#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    # Phase 3: optionally mix in an external celebrity-identification dataset
    # for text-loss co-training. Opt-in via the policy's celebrity_mix_ratio.
    mix_ratio = float(getattr(cfg.policy, "celebrity_mix_ratio", 0.0) or 0.0)
    if mix_ratio > 0.0:
        from lerobot.datasets.celebrity_dataset import (
            CelebrityIdentificationDataset,
            MixedRobotCelebrityDataset,
        )

        celeb_name = getattr(cfg.policy, "celebrity_dataset_name", "tonyassi/celebrity-1000")
        # camera_key picks the first camera; mixed batches need an image key
        # that the policy's input_features include.
        cam_keys = list(dataset.meta.camera_keys)
        image_key = cam_keys[0] if cam_keys else "observation.image"
        # Peek at one robot sample to learn the exact shapes the dataloader's
        # default_collate will demand. Celebrity placeholders must match these
        # or torch.stack crashes on mismatched sizes.
        sample = dataset[0]
        sample_img = sample[image_key]
        # image tensor shape can be (C, H, W) or (T, C, H, W).
        if sample_img.ndim == 4:
            _, _, H, W = sample_img.shape
        else:
            _, H, W = sample_img.shape
        robot_image_size = (int(H), int(W))

        # Action shape: typically (chunk_size, action_dim). Use whatever the
        # robot dataset emits raw — the model pads to max_action_dim internally.
        sample_action = sample.get("action")
        if sample_action is not None and sample_action.ndim == 2:
            chunk_size, action_dim = sample_action.shape
        else:
            chunk_size = getattr(cfg.policy, "chunk_size", 50)
            action_dim = getattr(cfg.policy, "max_action_dim", 32)

        # State shape: (state_dim,) — also pulled from the actual sample.
        sample_state = sample.get("observation.state")
        if sample_state is not None and sample_state.ndim == 1:
            state_dim = int(sample_state.shape[0])
        else:
            state_dim = getattr(cfg.policy, "max_state_dim", 32)

        celeb_ds = CelebrityIdentificationDataset(
            dataset_name=celeb_name,
            streaming=False,
            chunk_size=int(chunk_size),
            action_dim=int(action_dim),
            state_dim=int(state_dim),
            image_camera_key=image_key,
            target_image_size=robot_image_size,
            all_camera_keys=cam_keys if cam_keys else None,
        )
        dataset = MixedRobotCelebrityDataset(
            robot_dataset=dataset,
            celebrity_dataset=celeb_ds,
            celebrity_mix_ratio=mix_ratio,
        )

    return dataset
