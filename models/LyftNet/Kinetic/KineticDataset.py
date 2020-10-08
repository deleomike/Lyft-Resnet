from l5kit.dataset.ego import EgoDataset
from l5kit.dataset.agent import AgentDataset
from pathlib import Path
import bisect

from l5kit.data import ChunkedDataset, get_agents_slice_from_frames, get_frames_slice_from_scenes
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer
from l5kit.dataset.select_agents import TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE, select_agents

from models.LyftNet.Kinetic.KineticSample import generate_kinetic_agent_sample

import numpy as np
from zarr import convenience
from functools import partial
from typing import Optional, Tuple, cast


# WARNING: changing these values impact the number of instances selected for both train and inference!
MIN_FRAME_HISTORY = 10  # minimum number of frames an agents must have in the past to be picked
MIN_FRAME_FUTURE = 1  # minimum number of frames an agents must have in the future to be picked


class KineticDataset(AgentDataset):

    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
        agents_mask: Optional[np.ndarray] = None,
        min_frame_history: int = MIN_FRAME_HISTORY,
        min_frame_future: int = MIN_FRAME_FUTURE,
    ):
        assert perturbation is None, "KineticDataset does not support perturbation (yet)"

        super(KineticDataset, self).__init__(cfg, zarr_dataset, rasterizer, perturbation, agents_mask,
                                             min_frame_history, min_frame_future)

        # build a partial so we don't have to access cfg each time
        self.sample_function = partial(
            generate_kinetic_agent_sample,
            raster_size=cast(Tuple[int, int], tuple(cfg["raster_params"]["raster_size"])),
            pixel_size=np.array(cfg["raster_params"]["pixel_size"]),
            ego_center=np.array(cfg["raster_params"]["ego_center"]),
            history_num_frames=cfg["model_params"]["history_num_frames"],
            history_step_size=cfg["model_params"]["history_step_size"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            future_step_size=cfg["model_params"]["future_step_size"],
            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=rasterizer,
            perturbation=perturbation,
        )

    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
            the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp

        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]
        data = self.sample_function(state_index, frames, self.dataset.agents, self.dataset.tl_faces, track_id)
        # 0,1,C -> C,0,1
        image = data["image"].transpose(2, 0, 1)

        target_positions = np.array(data["target_positions"], dtype=np.float32)
        target_velocities = np.array(data["target_velocities"], dtype=np.float32)
        target_accelerations = np.array(data["target_accelerations"], dtype=np.float32)
        target_yaws = np.array(data["target_yaws"], dtype=np.float32)

        history_positions = np.array(data["history_positions"], dtype=np.float32)
        history_velocities = np.array(data["history_velocities"], dtype=np.float32)
        history_accelerations = np.array(data["history_accelerations"], dtype=np.float32)
        history_yaws = np.array(data["history_yaws"], dtype=np.float32)

        timestamp = frames[state_index]["timestamp"]
        track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        return {
            "image": image,
            "target_positions": target_positions,
            "target_velocities": target_velocities,
            "target_accelerations": target_accelerations,
            "target_yaws": target_yaws,
            "target_availabilities": data["target_availabilities"],
            "history_positions": history_positions,
            "history_velocities": history_velocities,
            "history_accelerations": history_accelerations,
            "history_yaws": history_yaws,
            "history_availabilities": data["history_availabilities"],
            "world_to_image": data["world_to_image"],
            "track_id": track_id,
            "timestamp": timestamp,
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "extent": data["extent"],
        }