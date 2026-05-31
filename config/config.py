import argparse
import os
from datetime import datetime


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="DMSDE: Dual-Memory Spiking Dynamic Experts"
        )

    def parse_args(self):
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument("--stream_batch_size", type=int, default=10)
        self.parser.add_argument("--test_batch_size", type=int, default=1)
        self.parser.add_argument("--gpu", type=str, default="0")
        self.parser.add_argument("--n_epochs", type=int, default=20)
        self.parser.add_argument("--threshold", type=float, default=None)
        self.parser.add_argument("--n_steps", type=int, default=16)
        self.parser.add_argument(
            "--vae_arch",
            "--vae",
            dest="vae_arch",
            type=str.lower,
            default="fsvae",
            choices=["fsvae"],
            help="Spiking VAE backbone used by each expert.",
        )
        self.parser.add_argument(
            "--vae_mmd_type",
            type=str.lower,
            default="rbf",
            choices=["rbf", "linear"],
            help="Kernel used by the latent MMD term.",
        )
        self.parser.add_argument(
            "--vae_distance_lambda",
            type=float,
            default=0.001,
            help="Weight on the latent MMD loss term.",
        )
        self.parser.add_argument("--dataset_fraction", type=float, default=0.15)
        self.parser.add_argument("--num_tasks", type=int, default=5)
        self.parser.add_argument("--num_samples", type=int, default=64)
        self.parser.add_argument("--save_dir", type=str, default="results")
        self.parser.add_argument("--model_dir", type=str, default="modelpth")
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="mnist",
            choices=["mnist", "cifar10"],
        )
        self.parser.add_argument("--input_channels", type=int, default=1)
        self.parser.add_argument("--img_size", type=int, default=32)
        self.parser.add_argument(
            "--classifier_type",
            type=str,
            default=None,
            choices=["ann", "snn", "resnet10", "resnet18"],
            help="Classifier head type.",
        )

        self.parser.add_argument("--short_memory_size", type=int, default=128)
        self.parser.add_argument("--long_memory_size", type=int, default=2000)
        self.parser.add_argument("--memory_size", type=int, default=None)
        self.parser.add_argument("--psp_tau_syn", type=float, default=10.0)
        self.parser.add_argument("--encode_batch_size", type=int, default=64)
        self.parser.add_argument(
            "--no_sample_selection",
            dest="sample_selection",
            action="store_false",
            help=(
                "Disable explanation-guided sample selection during memory "
                "consolidation."
            ),
        )
        self.parser.set_defaults(sample_selection=True)

        self.parser.add_argument(
            "--strategy",
            type=str,
            default="sdm_cl",
            choices=["sdm_cl", "sliding_window", "diversity"],
        )

        args = self.parser.parse_args()
        self._apply_dataset_defaults(args)
        return args

    def _apply_dataset_defaults(self, args):
        dataset = args.dataset.lower()
        if dataset == "cifar10":
            args.input_channels = 3
            args.img_size = 32
            if args.classifier_type is None:
                args.classifier_type = "resnet10"
        elif dataset == "mnist":
            args.input_channels = 1
            args.img_size = 32
            if args.classifier_type is None:
                args.classifier_type = "snn"

        if args.threshold is None:
            args.threshold = {
                "mnist": 0.08,
                "cifar10": 0.08,
            }[dataset]

        args.memory_size = args.short_memory_size + args.long_memory_size

    def generate_log_dir(self, args, dataset_name):
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H-%M-%S")

        param_dir = (
            f"{dataset_name}_"
            f"bs{args.batch_size}_"
            f"sbs{args.stream_batch_size}_"
            f"tbs{args.test_batch_size}_"
            f"frac{args.dataset_fraction:g}_"
            f"ep{args.n_epochs}_"
            f"vae{args.vae_arch}_"
            f"clf{args.classifier_type}_"
            f"sdm_ks{args.short_memory_size}_"
            f"kl{args.long_memory_size}_"
            f"th{args.threshold:.3f}_"
            f"gpu{args.gpu}"
        )

        day_dir = os.path.join(args.save_dir, current_date)
        os.makedirs(day_dir, exist_ok=True)

        time_dir = os.path.join(day_dir, f"{current_time}_{param_dir}")
        os.makedirs(time_dir, exist_ok=True)

        return time_dir
