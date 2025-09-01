import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import numpy as np
import cv2
from tqdm import tqdm
import rerun as rr
import glob
from natsort import natsorted
import json

import hydra
from anymap.models import init_model
from anymap.inference import loss_of_one_batch_multi_view
from anymap.utils.geometry import (
    depthmap_to_world_frame,
    quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
    relative_transformation,
)

# Import dataset modules
from anymap.datasets.multi_view_motion.tapvid3d_pstudio import TAPVID3D_PStudio_MultiView_Motion
from anymap.datasets.multi_view_motion.tapvid3d_drivetrack import TAPVID3D_DriveTrack_MultiView_Motion
from anymap.datasets.multi_view_motion.tapvid3d_adt import TAPVID3D_ADT_MultiView_Motion
from anymap.datasets.multi_view_motion.kubric_eval import KubricEval
from anymap.datasets.multi_view_motion.dynamic_replica_eval import DynamicReplicaEval

# Import visualization utilities
from anymap.utils.viz import script_add_rerun_args
from anymap.utils.image import rgb
from anymap.utils.misc import seed_everything
from anymap.train.training import build_dataset

from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.utils.visualizer import Visualizer
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri

def init_dataset(args, sequence_name, data_norm_type):

    sequence_path = os.path.join(args.dataset_dir, sequence_name)

    if args.dataset == "tapvid3d_pstudio":
        dataset = TAPVID3D_PStudio_MultiView_Motion(
            num_views=args.num_of_views,
            split="val",
            ROOT=sequence_path,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type=data_norm_type,
            iterate_over_scenes=False,
        )
    elif args.dataset == "tapvid3d_drivetrack":
        dataset = TAPVID3D_DriveTrack_MultiView_Motion(
            num_views=args.num_of_views,
            split="val",
            ROOT=sequence_path,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type=data_norm_type,
            iterate_over_scenes=False,
        )
    elif args.dataset == "tapvid3d_adt":
        dataset = TAPVID3D_ADT_MultiView_Motion(
            num_views=args.num_of_views,
            split="val",
            ROOT=sequence_path,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type=data_norm_type,
            iterate_over_scenes=False,
        )
    elif args.dataset == "kubric_eval":
        dataset = KubricEval(
            num_views=args.num_of_views,
            split="val",
            ROOT=sequence_path,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type=data_norm_type,
            iterate_over_scenes=False,
        )
    elif args.dataset == "dynamic_replica_eval":
        dataset = DynamicReplicaEval(
            num_views=args.num_of_views,
            split="val",
            ROOT=args.dataset_dir,
            seq_name=sequence_name,
            resolution=(args.img_width, args.img_height),
            transform="imgnorm",
            data_norm_type=data_norm_type,
            iterate_over_scenes=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return dataset

def log_data_to_rerun(base_name, image0, image1, poses, intrinsics, pts3d0, pts3d1, allo_scene_flow, mask0, mask1):
    """
    Log data to rerun for visualization

    Args:
        image0: first image (numpy array)
        image1: second image (numpy array)
        poses: ground truth camera poses (list of numpy arrays)
        intrinsics: camera intrinsics (numpy array)
        pts3d0: ground truth 3D points for first image (numpy array)
        pts3d1: ground truth 3D points for second image (numpy array)
        allo_scene_flow: ground truth scene flow in allo-centric frame (numpy array)
        mask0: valid mask for points (numpy array)
    """

    # Log camera info and loaded data
    height, width = image0.shape[:2]
    rr.log(
        f"{base_name}/view0",
        rr.Transform3D(
            translation=poses[0][:3, 3],
            mat3x3=poses[0][:3, :3],
            from_parent=False,
        ),
    )
    rr.log(
        f"{base_name}/view1",
        rr.Transform3D(
            translation=poses[1][:3, 3],
            mat3x3=poses[1][:3, :3],
            from_parent=False,
        ),
    )
    rr.log(
        f"{base_name}/view0/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(
        f"{base_name}/view1/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(
        f"{base_name}/view0/pinhole/rgb0",
        rr.Image(image0),
    )
    rr.log(
        f"{base_name}/view1/pinhole/rgb1",
        rr.Image(image1),
    )

    # Log points and scene flow
    filtered_pts3d0 = pts3d0[mask0]
    filtered_pts3d_col = image0[mask0]
    rr.log(
        f"{base_name}/pts3d0",
        rr.Points3D(
            positions=filtered_pts3d0.reshape(-1, 3),
            colors=filtered_pts3d_col.reshape(-1, 3),
        ),
    )

    filtered_pts3d1 = pts3d1[mask1]
    filtered_pts3d_col1 = image1[mask1]
    rr.log(
        f"{base_name}/pts3d1",
        rr.Points3D(
            positions=filtered_pts3d1.reshape(-1, 3),
            colors=filtered_pts3d_col1.reshape(-1, 3),
        ),
    )

    # Log allo-centric scene flow
    filtered_scene_flow = allo_scene_flow[mask0]
    rr.log(
        f"{base_name}/scene_flow",
        rr.Arrows3D(
            origins=filtered_pts3d0.reshape(-1, 3),
            vectors=filtered_scene_flow.reshape(-1, 3),
            # colors=filtered_pts3d_col.reshape(-1, 3),
        ),
    )

    # filtered_pts3d1 = pts3d1[mask0]
    # filtered_pts3d_col1 = image0[mask0]
    # rr.log(
    #     f"{base_name}/pts3d1",
    #     rr.Points3D(
    #         positions=filtered_pts3d1.reshape(-1, 3),
    #         colors=filtered_pts3d_col1.reshape(-1, 3),
    #     ),
    # )

    # # Log allo-centric scene flow
    # filtered_scene_flow = filtered_pts3d1 - filtered_pts3d0
    # rr.log(
    #     f"{base_name}/scene_flow",
    #     rr.Arrows3D(
    #         origins=filtered_pts3d0.reshape(-1, 3),
    #         vectors=filtered_scene_flow.reshape(-1, 3),
    #         # colors=filtered_pts3d_col.reshape(-1, 3),
    #     ),
    # )


def normalize_multiple_pointclouds(pts_list, valid_masks):
    """
    Normalize multiple pointclouds using average distance to origin.
    
    Args:
        pts_list: List of point clouds, each with shape HxWx3
        valid_masks: List of masks indicating valid points
    
    Returns:
        List of normalized point clouds, normalization factor
    """
    # Collect only valid points for normalization calculation
    all_valid_pts = []
    
    for i, pts in enumerate(pts_list):
        mask = valid_masks[i]
        valid_pts = pts[mask]  # Only extract valid points
        if len(valid_pts) > 0:
            all_valid_pts.append(valid_pts)
    
    if not all_valid_pts:
        # Handle edge case where no valid points exist
        return pts_list, 1.0
    
    # Concatenate all valid points
    all_pts = np.concatenate(all_valid_pts, axis=0)
    
    # Compute average distance to origin
    all_dis = np.linalg.norm(all_pts, axis=-1)
    norm_factor = np.mean(all_dis)
    norm_factor = max(norm_factor, 1e-8)  # Prevent division by zero
    
    # Normalize each point cloud
    res = [pts / norm_factor for pts in pts_list]
    
    return res, norm_factor


def find_closest_query_points(track2d_pred, query_pts):
    """
    Find closest query points to track2d_pred[0] predictions.
    
    Args:
        track2d_pred: (T, N, 3) tensor with x,y in first 2 columns
        query_pts: (N, 3) array with x,y in last 2 columns (int)
    
    Returns:
        closest_queries: (N, 3) array of closest query points
    """
    # Extract first frame predictions (N, 2) - x, y coordinates
    pred_xy = track2d_pred[0, :, :2]  # (N, 2)
    
    # Extract query point coordinates (N, 2) - last 2 columns are x, y
    query_xy = query_pts[:, 1:3].astype(float)  # (N, 2) convert to float for distance calc
    
    # Calculate distances between each prediction and all query points
    # pred_xy: (N, 2), query_xy: (M, 2) where M is number of query points
    distances = np.linalg.norm(pred_xy[:, None, :] - query_xy[None, :, :], axis=2)  # (N, M)
    
    # Find closest query point for each prediction
    closest_indices = np.argmin(distances, axis=1)  # (N,)
    
    # Return the closest query points
    closest_queries = query_pts[closest_indices]  # (N, 3)
    
    return closest_queries

def init_hydra_config(config_path, overrides=None):
    """Initialize Hydra config"""
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).split(".")[0]
    relative_path = os.path.relpath(config_dir, os.path.dirname(__file__))
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=relative_path)
    if overrides is not None:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    else:
        cfg = hydra.compose(config_name=config_name)
    return cfg


def init_inference_model(config, ckpt_path, device):
    "Initialize the model for inference"
    # Load the model
    if isinstance(config, dict):
        config_path = config["path"]
        overrrides = config["config_overrides"]
        model_args = init_hydra_config(config_path, overrides=overrrides)
        model = init_model(model_args.model.model_str, model_args.model.model_config)
    else:
        config_path = config
        model_args = init_hydra_config(config_path)
        model = init_model(model_args.model_str, model_args.model_config)
    model.to(device)
    if ckpt_path is not None:
        print("Loading model from: ", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
    # Set the model to eval mode
    model.eval()

    return model


@torch.no_grad()
def run_inference(model, views, device, use_amp):
    """Run inference using the Any4D model"""
    try:
        model_multi_view_inference = True if "multi_view" in model.name else False
    except:
        model_multi_view_inference = False
    
    result = loss_of_one_batch_multi_view(
        views,
        model,
        None,
        device,
        use_amp=use_amp,
        model_multi_view_inference=model_multi_view_inference,
    )
    return result


def spatrackv2_inference(args, views, video_tensor, query_pts,device="cuda"):
    """
    Run inference for the collected images using the SpaTrackV2 model.
    """

    vggt_high_level_config = {
        "path": f"{args.config_dir}/training_config/config.yaml",
        "config_overrides": [
            f"machine={args.machine}",
            "model=vggt",
            # "model=vggt_4d",
            # "model=vggt_4d_agnostic",
        ],
        "checkpoint_path": None, #"/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/any4d_experiments/vggt_finetune_single_dataset_runs/dynamic_all_blr_1e5/checkpoint-30.pth",
        "trained_with_amp": True,
        "data_norm_type": "identity",
    }

    mapa_any4d_high_level_config = {
        "path": f"{args.config_dir}/training_config/config.yaml",
        "config_overrides": [
            f"machine={args.machine}",
            # "model=any4d_mapa",
            # "model=mapanything",
            "model=any4d_mapa_dual_dpt",
            # "model/pred_head/adaptor_config=raydirs_depth_sceneflow_pose_confidence_scale",
            "model.encoder.uses_torch_hub=false",
            # "model/task=images_only",
            "model/task=pass_through_no_pose"
            # "model/task=posed_sfm"
            # "model/task=registration"
        ],
        "checkpoint_path": "/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/any4d_experiments/mapanything_finetune_megatraining/any4d_mapa_dual_dpt_aug_training_dynamic_all_mapa_finetune_run_no_metric_supervision/checkpoint-best.pth",
        "trained_with_amp": True,
        "data_norm_type": "dinov2",
    }

    any4d_model = init_inference_model(mapa_any4d_high_level_config, mapa_any4d_high_level_config["checkpoint_path"], device)

    mv_pred_result = run_inference(any4d_model, views, device=device, use_amp=mapa_any4d_high_level_config["trained_with_amp"])

    # Aggregate and stack predictions
    depths = []
    intrinsics = []
    extrinsics = []
    for idx in range(len(views)):
        pred_depth = mv_pred_result[f"pred{idx+1}"]["pts3d_cam"][:, :, :, 2].squeeze(0)
        pred_ray_directions = mv_pred_result[f"pred{idx+1}"]["ray_directions"][0]
        pred_cam_intrinsics = recover_pinhole_intrinsics_from_ray_directions(pred_ray_directions)
        pred_cam_quats_0 = mv_pred_result[f"pred{idx+1}"]["cam_quats"][0]
        pred_cam_trans_0 = mv_pred_result[f"pred{idx+1}"]["cam_trans"][0]
        pred_cam_rot_0 = quaternion_to_rotation_matrix(pred_cam_quats_0)
        pred_cam_pose_0 = torch.eye(4)
        pred_cam_pose_0[:3, :3] = pred_cam_rot_0
        pred_cam_pose_0[:3, 3] = pred_cam_trans_0
        pred_cam_pose_0 = pred_cam_pose_0

        depths.append(pred_depth)
        intrinsics.append(pred_cam_intrinsics)
        extrinsics.append(pred_cam_pose_0)

    depth_tensor = torch.stack(depths, dim=0).cpu().numpy()  # (N, H, W)
    intrs = torch.stack(intrinsics, dim=0).cpu().numpy()  # (N, 3, 3)
    extrs = torch.stack(extrinsics, dim=0).cpu().numpy()  # (N, 4, 4)
    video_tensor = video_tensor.squeeze()  # (N, C, H, W)

    # depth_tensor = depth_map.squeeze().cpu().numpy()
    # extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    # extrs = extrinsic.squeeze().cpu().numpy()
    # intrs = intrinsic.squeeze().cpu().numpy()
    # video_tensor = video_tensor.squeeze()
    #NOTE: 20% of the depth is not reliable
    # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
    unc_metric = None #depth_conf.squeeze().cpu().numpy() > 0.5

    data_npz_load = {}

    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")

    # config the model; the track_num is the number of points in the grid
    model.spatrack.track_num = 756
    model.eval()
    model.to("cuda")

    # import pdb; pdb.set_trace()

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_pts,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 

        # save as the tapip3d format
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()

        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map[:,2,...].clone()
        # depth_save[conf_depth<0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        data_npz_load["pointmaps"] = point_map.permute(0,2,3,1).cpu().numpy()

        # Save coords in TxHxWx3 format
        T_coords, N = data_npz_load["coords"].shape[:2]
        H, W = data_npz_load["pointmaps"].shape[1:3]

        # Find closest query points
        closest_query_pts = find_closest_query_points(track2d_pred.cpu().numpy(), query_pts)

        # x,y per track (constant over time)
        ys = closest_query_pts[:, 2].astype(int)   # (N,)
        xs = closest_query_pts[:, 1].astype(int)   # (N,)

        # Broadcast to (T, N)
        t_idx = np.arange(T_coords)[:, None].repeat(N, axis=1)       # (T,N)
        ys_b = np.broadcast_to(ys, (T_coords, N))                    # (T,N)
        xs_b = np.broadcast_to(xs, (T_coords, N))                    # (T,N)

        tracks_3d_pointmap = np.zeros((T_coords, H, W, 3), dtype=np.float32)
        tracks_3d_pointmap[t_idx, ys_b, xs_b] = data_npz_load["coords"]  # (T,N,3)

        data_npz_load["tracks_3d_pointmap"] = tracks_3d_pointmap

        return data_npz_load


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", default="psc")
    parser.add_argument("--config_dir", default="/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/configs")
    parser.add_argument('--dataset', help='Dataset type', default="tapvid3d_pstudio", type=str)
    parser.add_argument("--dataset_dir", default="/ocean/projects/cis220039p/mdt2/datasets/dydust3r/tapvid3d_dataset/pstudio")
    # parser.add_argument('--dataset', help='Dataset type', default="tapvid3d_drivetrack", type=str)
    # parser.add_argument("--dataset_dir", default="/ocean/projects/cis220039p/mdt2/datasets/dydust3r/tapvid3d_dataset/drivetrack")
    # parser.add_argument('--dataset', help='Dataset type', default="dynamic_replica_eval", type=str)
    # parser.add_argument("--dataset_dir", default="/ocean/projects/cis220039p/mdt2/datasets/dydust3r/dynamic_replica_data", type=str)
    parser.add_argument("--img_width", default=518, type=int, help="Image width")
    parser.add_argument("--img_height", default=336, type=int, help="Image height")
    parser.add_argument("--num_of_views", default=2, type=int, help="Number of views to use")
    parser.add_argument("--viz", action="store_true", help="Visualize results using rerun")
    parser.add_argument("--max_samples", default=None, type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--sample_ratio", default=1, type=float, help="Ratio of points to visualize")
    parser.add_argument("--only_dynamic", default=True, help="Only evaluate dynamic points", action="store_true")
    
    # Add rerun visualization arguments
    script_add_rerun_args(parser)
    
    args = parser.parse_args()

    return args


def main():
    """Main function for scene flow evaluation"""
    # Set random seed for reproducibility
    seed_everything(0)
    
    # Parse arguments
    args = get_args_parser()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup rerun visualization if requested
    if args.viz:
        rr.script_setup(args, f"SpaTrackv2_Benchmarking_{args.dataset}")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # Initialize metrics
    all_seq_metrics = {
        'epe3d_allo_image_all': 0.0,  # Sum of per-image average EPE3D in allo-centric
        'epe3d_allo_pixel_all': 0.0,  # Sum of all pixel EPE3D values in allo-centric
        'epe3d_sf_allo_image_all': 0.0,  # Sum of per-image average EPE3D in allo-centric for scene flow
        'epe3d_sf_allo_pixel_all': 0.0,  # Sum of all pixel EPE3D values in allo-centric for scene flow
        'num_valid_pixels': 0.0,     # Count of valid pixels
        'delta_0.05': 0.0,            # Count of pixels with abs_rel < 0.05
        'delta_0.1': 0.0,            # Count of pixels with abs_rel < 0.1
        'delta_0.2': 0.0,            # Count of pixels with abs_rel < 0.2
        'delta_0.4': 0.0,            # Count of pixels with abs_rel < 0.4
        'outlier_0.15': 0.0,          # Count of outliers with abs_rel > 0.15
        'delta_epe_0.1': 0.0,        # Count of pixels with EPE3D < 0.1
        'delta_epe_0.3': 0.0,         # Count of pixels with EPE3D < 0.3
        'delta_epe_0.5': 0.0,         # Count of pixels with EPE3D < 0.5
        'delta_epe_1.0': 0.0,         # Count of pixels with EPE3D < 1.0
        'outlier_epe_0.25': 0.0,         # Count of outliers with EPE3D > 0.25
    }

    # Set thresholds for accuracy metrics
    delta_thresholds = [0.05, 0.1, 0.2, 0.4]
    delta_epe_thresholds = [0.1, 0.3, 0.5, 1.0]
    outlier_thresholds = [0.15]
    outlier_epe_thresholds = [0.25]

    # Dataset initialization
    if args.dataset == "tapvid3d_pstudio":
        data_sequences = natsorted(glob.glob(f"{args.dataset_dir}/*.npz"))[:10]
    elif args.dataset == "tapvid3d_adt":
        data_sequences = natsorted(glob.glob(f"{args.dataset_dir}/*multiuser*.npz"))[:10]
    elif args.dataset == "tapvid3d_drivetrack":
        data_sequences = json.load(open('/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/anymap/datasets/multi_view_motion/drive_track_eligible_scenes.json'))[:10]
        # data_sequences = ['/ocean/projects/cis220039p/mdt2/datasets/dydust3r/tapvid3d_dataset/drivetrack/tapvid3d_11940460932056521663_1760_000_1780_000_1_9yu1CheynE6kQgwzmHcInQ.npz']
    elif args.dataset == "kubric_eval":
        data_sequences = [str(i) for i in range(5700, 6000)][:10]
    elif args.dataset == "dynamic_replica_eval":
        data_sequences = json.load(open('/ocean/projects/cis220039p/mdt2/jkarhade/Any4D/anymap/datasets/multi_view_motion/dynamic_replica_eval_val_scenes.json'))[:10]

    print(f"Found {len(data_sequences)} sequences.")

    if len(data_sequences) > 50:
        # Uniformly subsample to 50 sequences
        gap = len(data_sequences) // 50
        data_sequences = data_sequences[::gap]

    print(f"Using {len(data_sequences)} sequences for evaluation.")

    for sequence_path in data_sequences:
        dataset = init_dataset(args, os.path.basename(sequence_path), data_norm_type="identity")

        # Temporarily go down to 50 only
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(50, len(dataset))))

        print(f"Processing sequence: {os.path.basename(sequence_path)} with {len(dataset)} pairs")


        collected_imgs = []

        query_point_indices = np.where(dataset[0][0]["valid_mask"] > 0)
        query_pts = np.column_stack((np.zeros_like(query_point_indices[0]), query_point_indices[1], query_point_indices[0]))  # (0, x, y) format

        # Collect all views before hand for passing all images to model
        for idx in tqdm(range(len(dataset))):
            _, cur_view = dataset[idx]
            cur_img = cur_view["img"]
            collected_imgs.append(cur_img)

        collected_imgs = torch.stack(collected_imgs).to(device) # (N, 3, H, W)
        collected_imgs = collected_imgs[None]

        # Create dataloader
        any4d_dataset = init_dataset(args, os.path.basename(sequence_path), data_norm_type="dinov2")
        any4d_dataset = Subset(any4d_dataset, range(min(50, len(any4d_dataset))))
        any4d_dataloader = build_dataset(any4d_dataset, batch_size=1, num_workers=4, test=True)
        collected_views = []
        for idx, views in enumerate(tqdm(any4d_dataloader)):    
            collected_views.append(views[1])

        # Pass to model inference
        predictions = spatrackv2_inference(args, collected_views, collected_imgs, query_pts)

        # Initialize metrics
        cur_seq_metrics = {
            'epe3d_allo_image_all': 0.0,  # Sum of per-image average EPE3D in allo-centric
            'epe3d_allo_pixel_all': 0.0,  # Sum of all pixel EPE3D values in allo-centric
            'epe3d_sf_allo_image_all': 0.0,  # Sum of per-image average EPE3D in allo-centric for scene flow
            'epe3d_sf_allo_pixel_all': 0.0,  # Sum of all pixel EPE3D values in allo-centric for scene flow
            'num_valid_pixels': 0.0,     # Count of valid pixels
            'delta_0.05': 0.0,           # Count of pixels with abs_rel < 0.05
            'delta_0.1': 0.0,            # Count of pixels with abs_rel < 0.1
            'delta_0.2': 0.0,            # Count of pixels with abs_rel < 0.2
            'delta_0.4': 0.0,            # Count of pixels with abs_rel < 0.4
            'outlier_0.15': 0.0,         # Count of outliers with abs_rel > 0.15
            'delta_epe_0.1': 0.0,        # Count of pixels with EPE3D < 0.1
            'delta_epe_0.3': 0.0,         # Count of pixels with EPE3D < 0.3
            'delta_epe_0.5': 0.0,         # Count of pixels with EPE3D < 0.5
            'delta_epe_1.0': 0.0,         # Count of pixels with EPE3D < 1.0
            'outlier_epe_0.25': 0.0,         # Count of outliers with EPE3D > 0.25
        }

        # Iterate over predictions to calculate metrics
        for idx in tqdm(range(len(dataset))):

            # Get ground truth
            view0, view1 = dataset[idx]

            im0_path = view0["label"]
            im1_path = view1["label"]
            gt_allo_scene_flow = view1["allo_scene_flow"]
            # gt_valid_mask0 = view0["valid_mask"]
            gt_valid_mask1 = view1["valid_mask"]
            gt_pts3d0 = view0["pts3d"]
            gt_pts3d1 = view1["pts3d"]
            gt_cam0 = view0["camera_pose"]
            gt_cam1 = view1["camera_pose"]

            pred_pointmaps = [predictions["pointmaps"][0].copy(), predictions["pointmaps"][1].copy()]

            pred_pts3d = [predictions["tracks_3d_pointmap"][0].copy(), predictions["tracks_3d_pointmap"][idx].copy()]

            gt_valid_mask0 = (pred_pts3d[0][..., 2] > 0)

            pred_allo_scene_flow = pred_pts3d[1] - pred_pts3d[0]

            poses_c2w = [predictions["extrinsics"][0].copy(), predictions["extrinsics"][idx].copy()]
            pred_intrinsics = [predictions["intrinsics"][0].copy(), predictions["intrinsics"][idx].copy()]

            # Combine other masks to gt_valid_mask
            if args.only_dynamic:
                gt_valid_mask0 = gt_valid_mask0 & (np.linalg.norm(gt_allo_scene_flow,axis=-1) !=0)
            
            # Normalize clouds, poses and scene flow
            _, gt_norm_factor = normalize_multiple_pointclouds([gt_pts3d0, gt_pts3d1], [gt_valid_mask0, gt_valid_mask1])
            _, pred_norm_factor = normalize_multiple_pointclouds([pred_pts3d[0], pred_pts3d[1]], [gt_valid_mask0, gt_valid_mask0])

            scaling_factor = gt_norm_factor / pred_norm_factor
            pred_pointmaps = [pmap * scaling_factor for pmap in pred_pointmaps]
            pred_pts3d = [pts * scaling_factor for pts in pred_pts3d]
            pred_allo_scene_flow = pred_allo_scene_flow * scaling_factor
            poses_c2w[0][:3, 3] = poses_c2w[0][:3, 3] * scaling_factor
            poses_c2w[1][:3, 3] = poses_c2w[1][:3, 3] * scaling_factor


            # Calculate EPE3D
            if gt_valid_mask0.sum() == 0:
                print(f"Skipping image pair {idx} due to no valid points in gt_valid_mask0")
                continue

            epe3d = np.linalg.norm((gt_pts3d0 + gt_allo_scene_flow)[gt_valid_mask0] - (pred_pts3d[0] + pred_allo_scene_flow)[gt_valid_mask0], axis=-1)
            epe3d_sf = np.linalg.norm(gt_allo_scene_flow[gt_valid_mask0] - pred_allo_scene_flow[gt_valid_mask0], axis=-1)
            abs_rel = epe3d / (np.linalg.norm(gt_pts3d0[gt_valid_mask0] + gt_allo_scene_flow[gt_valid_mask0], axis=-1) + 1e-8)

            cur_seq_metrics['epe3d_allo_image_all'] += np.mean(epe3d)
            cur_seq_metrics['epe3d_allo_pixel_all'] += np.sum(epe3d)
            cur_seq_metrics['epe3d_sf_allo_image_all'] += np.mean(epe3d_sf)
            cur_seq_metrics['epe3d_sf_allo_pixel_all'] += np.sum(epe3d_sf)
            cur_seq_metrics['num_valid_pixels'] += np.sum(gt_valid_mask0)

            # Calculate delta metrics
            for threshold in delta_thresholds:
                cur_seq_metrics[f'delta_{threshold}'] += np.sum(abs_rel < threshold)

            for threshold in delta_epe_thresholds:
                cur_seq_metrics[f'delta_epe_{threshold}'] += np.sum(epe3d_sf < threshold)

            # Calculate outlier metrics
            for threshold in outlier_thresholds:
                cur_seq_metrics[f'outlier_{threshold}'] += np.sum(abs_rel > threshold)

            for threshold in outlier_epe_thresholds:
                cur_seq_metrics[f'outlier_epe_{threshold}'] += np.sum(epe3d_sf > threshold)

            # Visualization
            if args.viz:
                viz_img0 = rgb(view0["img"], norm_type="dinov2")
                viz_img1 = rgb(view1["img"], norm_type="dinov2")

                # Set time for correct sequence in visualization
                rr.set_time_seconds("stable_time", idx)

                log_data_to_rerun(
                    base_name="world/gt",
                    image0=viz_img0,
                    image1=viz_img1,
                    poses=[gt_cam0, gt_cam1],
                    intrinsics=view0["camera_intrinsics"],
                    pts3d0=gt_pts3d0,
                    pts3d1=gt_pts3d1,
                    allo_scene_flow=gt_allo_scene_flow,
                    mask0=gt_valid_mask0,
                    mask1=gt_valid_mask1,
                )

                # log_data_to_rerun(
                #     base_name="world/pred",
                #     image0=viz_img0,
                #     image1=viz_img1,
                #     poses=[gt_cam0, gt_cam1],
                #     intrinsics=view0["camera_intrinsics"],
                #     pts3d0=gt_pts3d0,
                #     pts3d1=gt_pts3d1,
                #     allo_scene_flow=pred_allo_scene_flow,
                #     mask0=gt_valid_mask0,
                #     mask1=gt_valid_mask1,  # Using the same mask for both images
                # )

                log_data_to_rerun(
                    base_name="world/pred_own_pts",
                    image0=viz_img0,
                    image1=viz_img1,
                    poses=[poses_c2w[0], poses_c2w[1]],
                    intrinsics=pred_intrinsics[0],
                    pts3d0=pred_pts3d[0],
                    pts3d1=pred_pts3d[1],
                    allo_scene_flow=pred_allo_scene_flow,
                    mask0=gt_valid_mask0, 
                    mask1=gt_valid_mask1, # Using the same mask for both images
                )

        if args.viz:
            import pdb; pdb.set_trace()

        # Aggregate and Print sequence metrics
        cur_seq_metrics['epe3d_allo_image_all'] /= len(dataset)
        cur_seq_metrics['epe3d_allo_pixel_all'] /= cur_seq_metrics['num_valid_pixels']
        cur_seq_metrics['epe3d_sf_allo_image_all'] /= len(dataset)
        cur_seq_metrics['epe3d_sf_allo_pixel_all'] /= cur_seq_metrics['num_valid_pixels']
        for threshold in delta_thresholds:
            cur_seq_metrics[f'delta_{threshold}'] /= cur_seq_metrics['num_valid_pixels']
        for threshold in outlier_thresholds:
            cur_seq_metrics[f'outlier_{threshold}'] /= cur_seq_metrics['num_valid_pixels']

        for threshold in delta_epe_thresholds:
            cur_seq_metrics[f'delta_epe_{threshold}'] /= cur_seq_metrics['num_valid_pixels']
        for threshold in outlier_epe_thresholds:
            cur_seq_metrics[f'outlier_epe_{threshold}'] /= cur_seq_metrics['num_valid_pixels']

        print(f"sequence:{sequence_path}")
        print(f"metrics: {cur_seq_metrics}")

        # Add sequence metrics into all_seq_metrics
        all_seq_metrics['epe3d_allo_image_all'] += cur_seq_metrics['epe3d_allo_image_all']
        all_seq_metrics['epe3d_allo_pixel_all'] += cur_seq_metrics['epe3d_allo_pixel_all']
        all_seq_metrics['epe3d_sf_allo_image_all'] += cur_seq_metrics['epe3d_sf_allo_image_all']
        all_seq_metrics['epe3d_sf_allo_pixel_all'] += cur_seq_metrics['epe3d_sf_allo_pixel_all']
        all_seq_metrics['num_valid_pixels'] += cur_seq_metrics['num_valid_pixels']
        for threshold in delta_thresholds:
            all_seq_metrics[f'delta_{threshold}'] += cur_seq_metrics[f'delta_{threshold}']
        for threshold in outlier_thresholds:
            all_seq_metrics[f'outlier_{threshold}'] += cur_seq_metrics[f'outlier_{threshold}']

        for threshold in delta_epe_thresholds:
            all_seq_metrics[f'delta_epe_{threshold}'] += cur_seq_metrics[f'delta_epe_{threshold}']
        for threshold in outlier_epe_thresholds:
            all_seq_metrics[f'outlier_epe_{threshold}'] += cur_seq_metrics[f'outlier_epe_{threshold}']

    # Finalize all_seq_metrics
    all_seq_metrics['epe3d_allo_image_all'] /= len(data_sequences)
    all_seq_metrics['epe3d_allo_pixel_all'] /= len(data_sequences)
    all_seq_metrics['epe3d_sf_allo_image_all'] /= len(data_sequences)
    all_seq_metrics['epe3d_sf_allo_pixel_all'] /= len(data_sequences)
    for threshold in delta_thresholds:
        all_seq_metrics[f'delta_{threshold}'] /= len(data_sequences)
    for threshold in outlier_thresholds:
        all_seq_metrics[f'outlier_{threshold}'] /= len(data_sequences)

    for threshold in delta_epe_thresholds:
        all_seq_metrics[f'delta_epe_{threshold}'] /= len(data_sequences)
    for threshold in outlier_epe_thresholds:
        all_seq_metrics[f'outlier_epe_{threshold}'] /= len(data_sequences)

    print("Final Metrics across all sequences:")
    print(f"epe3d_allo_image_all: {all_seq_metrics['epe3d_allo_image_all']}")
    print(f"epe3d_allo_pixel_all: {all_seq_metrics['epe3d_allo_pixel_all']}")
    print(f"epe3d_sf_allo_image_all: {all_seq_metrics['epe3d_sf_allo_image_all']}")
    print(f"epe3d_sf_allo_pixel_all: {all_seq_metrics['epe3d_sf_allo_pixel_all']}")
    print(f"num_valid_pixels: {all_seq_metrics['num_valid_pixels']}")
    for threshold in delta_thresholds:
        print(f"delta_{threshold}: {all_seq_metrics[f'delta_{threshold}']}")
    for threshold in outlier_thresholds:    
        print(f"outlier_{threshold}: {all_seq_metrics[f'outlier_{threshold}']}")
    for threshold in delta_epe_thresholds:
        print(f"delta_epe_{threshold}: {all_seq_metrics[f'delta_epe_{threshold}']}")
    for threshold in outlier_epe_thresholds:
        print(f"outlier_epe_{threshold}: {all_seq_metrics[f'outlier_epe_{threshold}']}")

    # Save metrics to txt
    metrics_file = os.path.join(f"{args.dataset}_pass_through_spatrackv2_any4d_benchmarking_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Final Metrics across all sequences:\n")
        f.write(f"epe3d_allo_image_all: {all_seq_metrics['epe3d_allo_image_all']}\n")
        f.write(f"epe3d_allo_pixel_all: {all_seq_metrics['epe3d_allo_pixel_all']}\n")
        f.write(f"epe3d_sf_allo_image_all: {all_seq_metrics['epe3d_sf_allo_image_all']}\n")
        f.write(f"epe3d_sf_allo_pixel_all: {all_seq_metrics['epe3d_sf_allo_pixel_all']}\n")
        f.write(f"num_valid_pixels: {all_seq_metrics['num_valid_pixels']}\n")
        for threshold in delta_thresholds:
            f.write(f"delta_{threshold}: {all_seq_metrics[f'delta_{threshold}']}\n")
        for threshold in outlier_thresholds:
            f.write(f"outlier_{threshold}: {all_seq_metrics[f'outlier_{threshold}']}\n")

    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()

