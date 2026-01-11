import os
import glob
import argparse
from typing import Optional, Dict, Tuple, Any

import numpy as np
import torch

# DQN value network from navigation.py
from navigation import FCQ


def find_checkpoint_dir(run_dir: str, trial_id: str = "default", seed: int = 22) -> str:
    """Return the checkpoints directory for a given run/trial/seed (navigation.py layout)."""
    ckpt_dir = os.path.join(run_dir, "trials", trial_id, f"seed_{seed}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoints dir not found: {ckpt_dir}")
    return ckpt_dir


def pick_checkpoint(ckpt_dir: str, episode: Optional[int] = None) -> str:
    """
    Pick a DQN checkpoint file saved by navigation.py:
      - If episode is provided, select model.<episode>.tar
      - Otherwise, pick the latest episode (highest number).
    """
    pattern = os.path.join(ckpt_dir, "model.*.tar")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    if episode is not None:
        target = os.path.join(ckpt_dir, f"model.{episode}.tar")
        if not os.path.isfile(target):
            raise FileNotFoundError(f"Checkpoint for episode {episode} not found: {target}")
        return target

    def ep_num(p: str) -> int:
        base = os.path.basename(p)  # model.<num>.tar
        parts = base.split(".")
        if len(parts) < 3:
            return -1
        try:
            return int(parts[1])
        except ValueError:
            return -1

    return max(files, key=ep_num)


def load_model_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    Load the state_dict from a DQN model checkpoint (navigation.py saves raw state_dict).
    Supports either:
      - raw state_dict (mapping param_name -> Tensor)
      - dict containing a nested 'state_dict'
    """
    obj: Any = torch.load(ckpt_path, map_location="cpu")

    if isinstance(obj, dict):
        # If it's already a param dict: values are tensors with .shape
        if obj and all(hasattr(v, "shape") for v in obj.values()):
            return obj  # type: ignore[return-value]

        # Common wrapper
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            if sd and all(hasattr(v, "shape") for v in sd.values()):
                return sd  # type: ignore[return-value]

    raise ValueError("Checkpoint does not appear to contain a valid model state_dict.")


def infer_hidden_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Infer hidden_dims tuple from checkpoint tensor shapes for FCQ.

    Assumes keys like:
      - input_layer.weight: [h1, state_size]
      - hidden_layers.0.weight (optional): [h2, h1]
    """
    if "input_layer.weight" not in state_dict:
        raise KeyError("Missing key 'input_layer.weight' in state_dict.")
    h1 = int(state_dict["input_layer.weight"].shape[0])

    if "hidden_layers.0.weight" in state_dict:
        h2 = int(state_dict["hidden_layers.0.weight"].shape[0])
    else:
        # FCQ might have only one hidden layer (or structure different from expectation)
        h2 = h1

    return (h1, h2)


def infer_io_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Infer (state_size, action_size) from checkpoint tensor shapes for FCQ.
      - input_layer.weight: [hidden, state_size]
      - output_layer.weight: [action_size, last_hidden]
    """
    if "input_layer.weight" not in state_dict:
        raise KeyError("Missing key 'input_layer.weight' in state_dict.")
    if "output_layer.weight" not in state_dict:
        raise KeyError("Missing key 'output_layer.weight' in state_dict.")

    state_size = int(state_dict["input_layer.weight"].shape[1])
    action_size = int(state_dict["output_layer.weight"].shape[0])
    return state_size, action_size


def save_weights_pt(state_dict: Dict[str, torch.Tensor], out_path: str) -> None:
    """Save the state_dict to a .pt file."""
    torch.save(state_dict, out_path)
    print(f"Saved PyTorch weights: {out_path}")


def save_weights_npz(state_dict: Dict[str, torch.Tensor], out_path: str) -> None:
    """Save the state_dict as a NumPy .npz."""
    np_weights = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    np.savez(out_path, **np_weights)
    print(f"Saved NumPy weights: {out_path}")


def rebuild_model_and_load(
    state_dict: Dict[str, torch.Tensor],
    hidden_dims: Optional[Tuple[int, int]] = None,
) -> FCQ:
    """
    Recreate the FCQ network with dims matching the checkpoint and load weights.

    Deduces:
      - nS from input_layer.weight second dimension
      - nA from output_layer.weight first dimension
    """
    if hidden_dims is None:
        hidden_dims = infer_hidden_dims_from_state_dict(state_dict)

    nS, nA = infer_io_dims_from_state_dict(state_dict)

    model = FCQ(nS, nA, hidden_dims=hidden_dims).to("cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def quick_validate_state_dict(state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Quick validation test:
    - Rebuild FCQ using inferred dims
    - Strictly load the checkpoint state_dict
    - Assert loaded tensors match exactly
    - Run a tiny forward pass and check shape + finiteness
    """
    hidden_dims = infer_hidden_dims_from_state_dict(state_dict)
    nS, nA = infer_io_dims_from_state_dict(state_dict)

    model = FCQ(nS, nA, hidden_dims=hidden_dims).to("cpu").eval()

    # First do a non-strict load just to print useful diagnostics on failure.
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing or unexpected:
        raise ValueError(
            "state_dict keys do not match the rebuilt FCQ architecture.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    # Now enforce strict load (should succeed)
    model.load_state_dict(state_dict, strict=True)

    # Verify tensors are exactly equal after load
    loaded_sd = model.state_dict()
    for k, v in state_dict.items():
        if k not in loaded_sd:
            raise KeyError(f"Key '{k}' not found in rebuilt model state_dict.")
        if loaded_sd[k].shape != v.shape:
            raise ValueError(f"Shape mismatch for '{k}': model {tuple(loaded_sd[k].shape)} vs ckpt {tuple(v.shape)}")
        if not torch.equal(loaded_sd[k].cpu(), v.cpu()):
            raise AssertionError(f"Loaded parameter tensor mismatch for key '{k}'.")

    # Forward sanity check
    dummy_state = torch.zeros(2, nS, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        q = model(dummy_state)

    if q.shape != (2, nA):
        raise AssertionError(f"Unexpected Q output shape: got {tuple(q.shape)}, expected {(2, nA)}")
    if not torch.isfinite(q).all():
        raise AssertionError("Non-finite values found in Q output.")

    print(f"[validate] OK: rebuilt FCQ(state={nS}, actions={nA}, hidden_dims={hidden_dims}) and loaded weights.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract DQN model weights from navigation.py checkpoints and validate by rebuilding FCQ."
    )
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help=r"Path to run dir, e.g. ...\results\navigation\run_YYYYMMDD_HHMMSS",
    )
    p.add_argument("--trial-id", type=str, default="default", help="Trial id folder name (e.g. default, trial_003).")
    p.add_argument("--seed", type=int, default=22, help="Seed folder to load (seed_<seed>).")
    p.add_argument("--episode", type=int, default=None, help="Episode index to load. If omitted, loads latest.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. If omitted, saves next to the checkpoint in checkpoints/.",
    )
    p.add_argument("--no-pt", action="store_true", help="Do not save .weights.pt")
    p.add_argument("--no-npz", action="store_true", help="Do not save .weights.npz")
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip quick validation (rebuild FCQ, strict load, and forward sanity check).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_dir = find_checkpoint_dir(args.run_dir, trial_id=args.trial_id, seed=args.seed)
    ckpt_path = pick_checkpoint(ckpt_dir, episode=args.episode)
    print(f"Using checkpoint: {ckpt_path}")

    state_dict = load_model_state_dict(ckpt_path)

    # Choose output directory
    out_dir = args.out_dir or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(ckpt_path)  # model.2566.tar
    stem = os.path.splitext(base)[0]    # model.2566

    if not args.no_pt:
        out_pt = os.path.join(out_dir, f"{stem}.weights.pt")
        save_weights_pt(state_dict, out_pt)

    if not args.no_npz:
        out_npz = os.path.join(out_dir, f"{stem}.weights.npz")
        save_weights_npz(state_dict, out_npz)

    if not args.no_validate:
        quick_validate_state_dict(state_dict)


if __name__ == "__main__":
    main()