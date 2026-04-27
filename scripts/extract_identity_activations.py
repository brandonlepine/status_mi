#!/usr/bin/env python3
"""Extract final-token residual activations for identity prompts.

This script is intended for RunPod/GPU execution against a local Llama-3.1-8B
checkpoint. It saves one row-aligned NumPy array per hidden-state layer:
layer_00.npy is the embedding output, and layer_01.npy onward are transformer
layer outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUT_CSV = Path("/workspace/status_mi/data/mi_identity_prompts.csv")
DEFAULT_MODEL_PATH = Path("/workspace/status_mi/models/llama-3.1-8b")
DEFAULT_OUTPUT_DIR = Path(
    "/workspace/status_mi/results/activations/llama-3.1-8b/"
    "identity_prompts_final_token"
)
CHECKPOINT_FILENAME = "checkpoint.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract final-token hidden states for identity prompts."
    )
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing layer files in output_dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output_dir/checkpoint.json and existing layer files.",
    )
    return parser.parse_args()


def choose_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def prepare_output_dir(output_dir: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("Use only one of --overwrite or --resume.")

    existing_layer_files = sorted(output_dir.glob("layer_*.npy")) if output_dir.exists() else []
    checkpoint_path = output_dir / CHECKPOINT_FILENAME

    if existing_layer_files and not overwrite and not resume:
        examples = ", ".join(path.name for path in existing_layer_files[:5])
        raise FileExistsError(
            f"{output_dir} already contains layer files ({examples}). "
            "Pass --overwrite to replace them or --resume to continue."
        )

    if resume and existing_layer_files and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Cannot resume because {checkpoint_path} does not exist."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for path in output_dir.glob("layer_*.npy"):
            path.unlink()
        for filename in ("metadata.csv", "run_config.json", CHECKPOINT_FILENAME):
            path = output_dir / filename
            if path.exists():
                path.unlink()


def load_checkpoint(output_dir: Path) -> dict[str, int] | None:
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        return None

    with checkpoint_path.open() as f:
        checkpoint = json.load(f)

    return {
        "rows_written": int(checkpoint["rows_written"]),
        "num_prompts": int(checkpoint["num_prompts"]),
        "num_layers_saved": int(checkpoint["num_layers_saved"]),
        "hidden_dim": int(checkpoint["hidden_dim"]),
    }


def write_checkpoint(
    *,
    output_dir: Path,
    rows_written: int,
    num_prompts: int,
    num_layers_saved: int,
    hidden_dim: int,
    batch_size: int,
    max_length: int,
) -> None:
    checkpoint = {
        "rows_written": rows_written,
        "num_prompts": num_prompts,
        "num_layers_saved": num_layers_saved,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "max_length": max_length,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    tmp_path = checkpoint_path.with_suffix(".json.tmp")

    with tmp_path.open("w") as f:
        json.dump(checkpoint, f, indent=2)
        f.write("\n")

    tmp_path.replace(checkpoint_path)


def get_input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def load_prompts(input_csv: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(input_csv, keep_default_na=False)

    assert "prompt" in df.columns, "Input CSV must include a 'prompt' column."
    assert "prompt_id" in df.columns, "Input CSV must include a 'prompt_id' column."

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer when provided.")
        df = df.head(limit).copy()

    if df.empty:
        raise ValueError("No prompts to process after applying --limit.")

    empty_prompts = df["prompt"].astype(str).str.strip().eq("")
    assert not empty_prompts.any(), "Input CSV contains empty prompts."

    return df.reset_index(drop=True)


def extract_final_token_activations(
    *,
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    output_dir: Path,
    batch_size: int,
    max_length: int,
    resume: bool,
) -> tuple[int, int]:
    prompts = df["prompt"].astype(str).tolist()
    num_prompts = len(prompts)
    input_device = get_input_device(model)

    layer_arrays: list[np.memmap] | None = None
    num_layers_saved: int | None = None
    hidden_dim: int | None = None
    checkpoint = load_checkpoint(output_dir) if resume else None
    rows_written = checkpoint["rows_written"] if checkpoint else 0

    if checkpoint:
        if checkpoint["num_prompts"] != num_prompts:
            raise ValueError(
                "Checkpoint num_prompts does not match current input rows. "
                "Use the same input/limit or restart with --overwrite."
            )
        print(f"Resuming from row {rows_written:,} of {num_prompts:,}")

    print(f"Number of prompts: {num_prompts:,}")
    print(f"Input device: {input_device}")

    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"  cuda:{idx}: {torch.cuda.get_device_name(idx)}")
    else:
        print("CUDA devices: 0")

    if rows_written >= num_prompts:
        if not checkpoint:
            raise ValueError("rows_written reached num_prompts without a checkpoint.")
        print("Checkpoint already indicates extraction is complete.")
        return checkpoint["num_layers_saved"], checkpoint["hidden_dim"]

    progress = tqdm(
        range(rows_written, num_prompts, batch_size),
        desc="Extracting batches",
        initial=rows_written,
        total=num_prompts,
        unit="prompt",
    )

    with torch.inference_mode():
        for start in progress:
            end = min(start + batch_size, num_prompts)
            batch_prompts = prompts[start:end]

            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            attention_mask = encoded["attention_mask"]

            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states.")

            if layer_arrays is None:
                num_layers_saved = len(hidden_states)
                hidden_dim = hidden_states[0].shape[-1]
                print(f"Hidden-state tensors returned: {num_layers_saved}")
                print(f"Hidden dimension: {hidden_dim}")
                print("Layer output shapes from first batch:")

                layer_arrays = []
                for layer_idx, hidden_state in enumerate(hidden_states):
                    print(f"  layer_{layer_idx:02d}: {tuple(hidden_state.shape)}")
                    layer_path = output_dir / f"layer_{layer_idx:02d}.npy"
                    mode = "r+" if rows_written > 0 else "w+"
                    if rows_written > 0 and not layer_path.exists():
                        raise FileNotFoundError(
                            f"Cannot resume because {layer_path} is missing."
                        )
                    arr = np.lib.format.open_memmap(
                        layer_path,
                        mode=mode,
                        dtype=np.float32,
                        shape=(num_prompts, hidden_dim),
                    )
                    layer_arrays.append(arr)

            assert layer_arrays is not None
            assert num_layers_saved is not None
            assert hidden_dim is not None
            assert len(hidden_states) == num_layers_saved

            # Padding is set to the right so sum(attention_mask) - 1 is the final
            # non-padding token index for each prompt.
            final_idx = attention_mask.sum(dim=1) - 1
            batch_len = end - start

            for layer_idx, hidden_state in enumerate(hidden_states):
                batch_indices = torch.arange(batch_len, device=hidden_state.device)
                layer_final_idx = final_idx.to(hidden_state.device)
                final_hidden = hidden_state[batch_indices, layer_final_idx, :]

                layer_arrays[layer_idx][start:end, :] = (
                    final_hidden.detach().float().cpu().numpy()
                )

            rows_written = end

            for arr in layer_arrays:
                arr.flush()

            write_checkpoint(
                output_dir=output_dir,
                rows_written=rows_written,
                num_prompts=num_prompts,
                num_layers_saved=num_layers_saved,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                max_length=max_length,
            )
            progress.set_postfix(rows_written=rows_written)

            del outputs, hidden_states, encoded, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    assert rows_written == num_prompts
    assert layer_arrays is not None
    assert num_layers_saved is not None
    assert hidden_dim is not None

    for arr in layer_arrays:
        arr.flush()

    return num_layers_saved, hidden_dim


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive.")

    prepare_output_dir(args.output_dir, args.overwrite, args.resume)

    df = load_prompts(args.input_csv, args.limit)
    metadata_path = args.output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)

    torch_dtype = choose_torch_dtype()
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model from: {args.model_path}")
    print(f"Requested torch dtype: {torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    num_layers_saved, hidden_dim = extract_final_token_activations(
        df=df,
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        resume=args.resume,
    )

    df.to_csv(metadata_path, index=False)

    assert len(df) == len(pd.read_csv(metadata_path, keep_default_na=False))

    run_config = {
        "model_path": str(args.model_path),
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_prompts": len(df),
        "num_layers_saved": num_layers_saved,
        "hidden_dim": hidden_dim,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    run_config_path = args.output_dir / "run_config.json"
    with run_config_path.open("w") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")

    print("Final output paths:")
    print(f"  layers: {args.output_dir / 'layer_XX.npy'}")
    print(f"  metadata: {metadata_path}")
    print(f"  run config: {run_config_path}")
    print(f"  checkpoint: {args.output_dir / CHECKPOINT_FILENAME}")
    print(f"Saved rows: {len(df):,}")
    print(f"Saved layers: {num_layers_saved}")
    print(f"Hidden dim: {hidden_dim}")


if __name__ == "__main__":
    main()
