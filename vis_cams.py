import numpy as np

from argparse import ArgumentParser
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom

if __name__ == "__main__":

    parser = ArgumentParser(
        description="A Python script to visualize cam positions with Visdom",
        epilog="python vis_cams.py --pred /path/to/pred_cameras.npz \
                                   --gt /path/to/gt_cameras.npz",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="",
        help="The PerspectiveCameras prediction.",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="",
        help="The PerspectiveCameras groundtruth.",
    )
    args = parser.parse_args()

    assert args.pred.endswith(".npz")
    pred_cameras_dict = np.load(args.pred)
    pred_cameras = PerspectiveCameras(
        focal_length=pred_cameras_dict["focal_length"],
        R=pred_cameras_dict["R"],
        T=pred_cameras_dict["T"],
    )

    if args.gt.endswith(".npz"):
        gt_cameras_dict = np.load(args.gt)
        gt_cameras = PerspectiveCameras(
            focal_length=gt_cameras_dict["gtFL"],
            R=gt_cameras_dict["gtR"],
            T=gt_cameras_dict["gtT"],
        )
        cams_show = {
            "ours_pred": pred_cameras,
            "gt_cameras": gt_cameras,
        }
    else:
        cams_show = {
            "ours_pred": pred_cameras,
        }

    viz = Visdom()
    fig = plot_scene({f"{args.pred}": cams_show})
    viz.plotlyplot(fig, env="visual", win="cams")
