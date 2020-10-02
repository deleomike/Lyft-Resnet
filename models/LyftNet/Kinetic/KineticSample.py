from typing import List, Optional, Tuple

import numpy as np

from l5kit.data import (
    TL_FACE_DTYPE,
    filter_agents_by_labels,
    filter_tl_faces_by_frames,
    get_agents_slice_from_frames,
    get_tl_faces_slice_from_frames,
)
from l5kit.data.filter import filter_agents_by_frames, filter_agents_by_track_id
from l5kit.geometry import rotation33_as_yaw, world_to_image_pixels_matrix
from l5kit.kinematic import Perturbation
from l5kit.rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer
from l5kit.sampling.slicing import get_future_slice, get_history_slice


def generate_kinetic_agent_sample(
    state_index: int,
    frames: np.ndarray,
    agents: np.ndarray,
    tl_faces: np.ndarray,
    selected_track_id: Optional[int],
    raster_size: Tuple[int, int],
    pixel_size: np.ndarray,
    ego_center: np.ndarray,
    history_num_frames: int,
    history_step_size: int,
    future_num_frames: int,
    future_step_size: int,
    filter_agents_threshold: float,
    rasterizer: Optional[Rasterizer] = None,
    perturbation: Optional[Perturbation] = None,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        raster_size (Tuple[int, int]): Desired output raster dimensions
        pixel_size (np.ndarray): Size of one pixel in the real world
        ego_center (np.ndarray): Where in the raster to draw the ego, [0.5,0.5] would be the center
        history_num_frames (int): Amount of history frames to draw into the rasters
        history_step_size (int): Steps to take between frames, can be used to subsample history frames
        future_num_frames (int): Amount of history frames to draw into the rasters
        future_step_size (int): Steps to take between targets into the future
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer (Optional[Rasterizer]): Rasterizer of some sort that draws a map image
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
to train models that can recover from slight divergence from training set data

    Raises:
        ValueError: A ValueError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """

    agent_velocity = None

    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = get_history_slice(state_index, history_num_frames, history_step_size, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, future_step_size)

    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()

    sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    agents = agents[agent_slice].copy()  # this is the minimum slice of agents we need
    history_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    future_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    history_agents = filter_agents_by_frames(history_frames, agents)
    future_agents = filter_agents_by_frames(future_frames, agents)

    try:
        tl_slice = get_tl_faces_slice_from_frames(history_frames[-1], history_frames[0])  # -1 is the farthest
        # sync interval with the traffic light faces array
        history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
        history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces[tl_slice].copy())
    except ValueError:
        history_tl_faces = [np.empty(0, dtype=TL_FACE_DTYPE) for _ in history_frames]

    if perturbation is not None:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid = cur_frame["ego_translation"][:2]

        # FIXME
        # Assume velocity was 0, but it probably shouldnt be
        agent_velocity = np.array([0, 0], dtype=np.float32)

        agent_yaw = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        try:
            agent = filter_agents_by_track_id(
                filter_agents_by_labels(cur_agents, filter_agents_threshold), selected_track_id
            )[0]
        except IndexError:
            raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")
        agent_centroid = agent["centroid"]
        agent_velocity = agent["velocity"]
        agent_yaw = float(agent["yaw"])
        agent_extent = agent["extent"]
        selected_agent = agent

    input_im = (
        None
        if not rasterizer
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )

    world_to_image_space = world_to_image_pixels_matrix(
        raster_size,
        pixel_size,
        ego_translation_m=agent_centroid,
        ego_yaw_rad=agent_yaw,
        ego_center_in_image_ratio=ego_center,
    )

    future_coords_offset, future_vel_offset, future_accel_offset,future_yaws_offset, future_availability = _create_targets_for_deep_prediction(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_centroid[:2], agent_velocity[:2],
        agent_yaw,
    )

    # history_num_frames + 1 because it also includes the current frame
    history_coords_offset, _, _, history_yaws_offset, history_availability = _create_targets_for_deep_prediction(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_centroid[:2], agent_velocity,
        agent_yaw,
    )

    return {
        "image": input_im,
        "target_positions": future_coords_offset,
        "target_velocities": future_vel_offset,
        "target_accelerations": future_accel_offset,
        "target_yaws": future_yaws_offset,
        "target_availabilities": future_availability,
        "history_positions": history_coords_offset,
        "history_yaws": history_yaws_offset,
        "history_availabilities": history_availability,
        "world_to_image": world_to_image_space,
        "centroid": agent_centroid,
        "yaw": agent_yaw,
        "extent": agent_extent,
    }


def _create_targets_for_deep_prediction(
    num_frames: int,
    frames: np.ndarray,
    selected_track_id: Optional[int],
    agents: List[np.ndarray],
    agent_current_centroid: np.ndarray,
    agent_current_velocity: np.ndarray,
    agent_current_yaw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).

    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_id (Optional[int]): agent track_id or AV (None)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_current_centroid (np.ndarray): centroid of the agent at timestep 0
        agent_current_yaw (float): angle of the agent at timestep 0

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, availabilities

    """
    # How much the coordinates differ from the current state in meters.
    coords_offset = np.zeros((num_frames, 2), dtype=np.float32)
    vel_offset = np.zeros((num_frames, 2), dtype=np.float32)
    accel_offset = np.zeros((num_frames, 2), dtype=np.float32)
    yaws_offset = np.zeros((num_frames, 1), dtype=np.float32)
    availability = np.zeros((num_frames,), dtype=np.float32)

    for i, (frame, agents) in enumerate(zip(frames, agents)):
        if selected_track_id is None:
            agent_centroid = frame["ego_translation"][:2]
            agent_yaw = rotation33_as_yaw(frame["ego_rotation"])
            agent_velocity = np.array([0, 0], dtype=np.float32)
        else:
            # it's not guaranteed the target will be in every frame
            try:
                agent = filter_agents_by_track_id(agents, selected_track_id)[0]
            except IndexError:
                availability[i] = 0.0  # keep track of invalid futures/history
                continue

            agent_centroid = agent["centroid"]
            agent_velocity = agent["velocity"]
            agent_yaw = agent["yaw"]

        coords_offset[i] = agent_centroid - agent_current_centroid
        vel_offset[i] = agent_velocity - agent_current_velocity
        # print(coords_offset[i], vel_offset[i])
        if coords_offset[i].all() == 0.0:
            accel_offset[i] = coords_offset[i]
        else:
            accel_offset[i] = (agent_velocity ** 2 - agent_current_velocity ** 2) / (2 * coords_offset[i])

        yaws_offset[i] = agent_yaw - agent_current_yaw
        availability[i] = 1.0
    return coords_offset, vel_offset, accel_offset, yaws_offset, availability