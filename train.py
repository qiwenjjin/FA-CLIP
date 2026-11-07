"""FA-CLIP training entry (partial release).

Documents manuscript defaults (batch size, seeds, cache / ODA hyperparameters,
source protocol). Full data loaders and CLIP encoders are omitted; it only records the settings used in the paper.
"""
import argparse

from method import FaCLIP_Trainer


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def build_trainer(args):
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    return FaCLIP_Trainer(
        learning_rate=args.learning_rate,
        device=device,
        feature_dim=args.feature_dim,
        cache_memory_size=args.cache_memory_size,
        cache_momentum=args.cache_momentum,
        cache_alpha=args.cache_alpha,
        cache_n_thres=args.cache_n_thres,
        cache_obs_history_thres=args.cache_obs_history_thres,
        cache_bandwidth_scale=args.cache_bandwidth_scale,
        oda_temp_init=args.oda_temp_init,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("FA-CLIP", add_help=True)

    parser.add_argument(
        "--training_data", type=str, default=["mvtec", "colondb"], nargs='+',
        help="Source datasets (default: MVTec AD + ColonDB)")
    parser.add_argument(
        "--testing_data", type=str, default="visa",
        help="Monitor / eval dataset (default: VisA)")
    parser.add_argument(
        "--model", type=str, default="ViT-L-14-336",
        help="CLIP backbone (paper default: ViT-L-14-336; encoder omitted here)")
    parser.add_argument("--feature_dim", type=int, default=768,
                        help="Encoder feature dimension (ViT-L/14 embed dim)")

    parser.add_argument("--epoch", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (default: 32)")
    parser.add_argument("--image_size", type=int, default=518, help="Input resolution (default: 518)")

    parser.add_argument(
        "--oda_temp_init", type=float, default=0.1,
        help="Initial value for learnable ODA log-temperature")
    parser.add_argument(
        "--seed", type=int, default=111,
        help="Random seed (paper reports {111, 222, 333})")
    parser.add_argument(
        "--zsad_protocol", type=str, default="default", choices=["default", "swap"],
        help="default: MVTec+ColonDB sources; swap: VisA+ClinicDB sources")

    parser.add_argument("--cache_memory_size", type=int, default=25, help="Cache size M (default: 25)")
    parser.add_argument("--cache_momentum", type=float, default=0.8, help="Cache momentum m (default: 0.8)")
    parser.add_argument("--cache_alpha", type=float, default=0.2, help="Residual coefficient η (default: 0.2)")
    parser.add_argument("--cache_n_thres", type=int, default=100, help="ITE history threshold N_thres (default: 100)")
    parser.add_argument(
        "--cache_obs_history_thres", type=int, default=200,
        help="Phase-II observational history length before ITE gating")
    parser.add_argument(
        "--cache_bandwidth_scale", type=float, default=1.0,
        help="Silverman bandwidth multiplier (default: 1.0)")

    args = parser.parse_args()

    if args.zsad_protocol == "swap" and list(args.training_data) == ["mvtec", "colondb"]:
        args.training_data = ["visa", "clinicdb"]
        if args.testing_data == "visa":
            args.testing_data = "mvtec"

    trainer = build_trainer(args)