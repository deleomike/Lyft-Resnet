from l5kit.dataset import AgentDataset

from l5kit.data import ChunkedDataset, get_agents_slice_from_frames, get_frames_slice_from_scenes
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer
from l5kit.dataset.select_agents import TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE, select_agents

from models.LyftNet.KineticSample import generate_kinetic_agent_sample

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
        assert perturbation is None, "AgentDataset does not support perturbation (yet)"

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