from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HAVIT_DIR = Path(os.environ.get("HAVIT_DIR", REPO_ROOT.parent / "HAViT")).resolve()

if not (HAVIT_DIR / "main.py").exists():
    sys.exit(
        f"[run.py] Cannot find HAViT/main.py at {HAVIT_DIR}\n"
        "Clone: git clone https://github.com/banik-s/HAViT\n"
        "or set HAVIT_DIR env var."
    )


def _run(model: str, exp_name: str, cli: argparse.Namespace, alpha: float = 0.9) -> None:
    print(f"\n{'='*65}")
    print(f"  model={model}  exp={exp_name}  dataset={cli.dataset}  epochs={cli.epochs}")
    print(f"{'='*65}\n")

    cmd = [
        sys.executable,
        str(HAVIT_DIR / "main.py"),
        "--model",
        model,
        "--dataset",
        cli.dataset,
        "--data_path",
        cli.data_path,
        "--exp_name",
        exp_name,
        "--epochs",
        str(cli.epochs),
        "--batch_size",
        str(cli.batch_size),
        "--lr",
        str(cli.lr),
        "--seed",
        str(cli.seed),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["HAVIT_ALPHA"] = str(alpha)

    proc = subprocess.Popen(
        cmd,
        cwd=str(HAVIT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()


def run_experiment(key: str, cli: argparse.Namespace) -> None:
    if key == "baseline_vit":
        print("\n── Paper repro: Baseline ViT (target 75.74%) ───────────────")
        _run("vitlucidrains", "repro_baseline_vit", cli)

    elif key == "havit_v1":
        print("\n── Paper repro: HAViT-v1 (target 77.07%) ───────────────────")
        _run("vitlucidrains_mod_ver1", "repro_havit_v1", cli)

    elif key == "learnable_alpha":
        print("\n── OUR EXP-A: per-head learnable alpha ─────────────────────")
        _run("havit_learnable_alpha", "our_learnable_alpha", cli)

    elif key == "post_softmax":
        print("\n── OUR EXP-B: history in probability space ─────────────────")
        _run("havit_post_softmax", "our_post_softmax", cli)

    elif key == "zero_init":
        print("\n── OUR EXP-C: H_0=zeros ablation ───────────────────────────")
        _run("havit_zero_init", "our_zero_init", cli)

    elif key == "all":
        for k in ["baseline_vit", "havit_v1", "learnable_alpha", "post_softmax", "zero_init"]:
            run_experiment(k, cli)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--exp",
        required=True,
        choices=["baseline_vit", "havit_v1", "learnable_alpha", "post_softmax", "zero_init", "all"],
    )
    p.add_argument("--dataset", default="CIFAR100", choices=["CIFAR100", "CIFAR10"])
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--lr", default=0.003, type=float)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--data_path", default="./datasets/")
    cli = p.parse_args()
    run_experiment(cli.exp, cli)
    print("\n Done.")


if __name__ == "__main__":
    main()
