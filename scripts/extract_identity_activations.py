#!/usr/bin/env python3
"""Extract per-prompt residual activations for identity prompts.

This script is intended for RunPod/GPU execution against a local Llama-3.1-8B
checkpoint. It saves one row-aligned NumPy array per hidden-state layer:
layer_00.npy is the embedding output, and layer_01.npy onward are transformer
layer outputs.

The token at which the activation is read is governed by --token_mode (audit
issue 1.1):

  - final_token         : last non-padding token of the prompt (legacy
                          behavior; for the templated identity prompts this
                          is essentially always the sentence-final period).
  - identity_span_last  : last token of the identity surface form (form_used
                          column from the prompt CSV).
  - identity_span_mean  : mean of the residual stream over all tokens inside
                          the identity span.

The span modes locate form_used in the prompt via case-insensitive regex
match plus tokenizer offset_mapping. Failures (form not found, zero span
tokens after tokenization, span truncated by --max_length) raise loudly so
silent measurement-locus errors cannot accumulate.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUT_CSV = Path("/workspace/status_mi/data/mi_identity_prompts.csv")
DEFAULT_MODEL_PATH = Path("/workspace/status_mi/models/llama-3.1-8b")
DEFAULT_OUTPUT_BASE = Path("/workspace/status_mi/results/activations/llama-3.1-8b")
TOKEN_MODES = ("final_token", "identity_span_last", "identity_span_mean")
DEFAULT_TOKEN_MODE = "final_token"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_BASE / f"identity_prompts_{DEFAULT_TOKEN_MODE}"
CHECKPOINT_FILENAME = "checkpoint.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--token_mode",
        choices=TOKEN_MODES,
        default=DEFAULT_TOKEN_MODE,
        help="Which token position to read residual activations at (audit 1.1).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If omitted, defaults to "
            f"{DEFAULT_OUTPUT_BASE}/identity_prompts_<token_mode>/ so different "
            "token modes do not overwrite each other."
        ),
    )
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


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return DEFAULT_OUTPUT_BASE / f"identity_prompts_{args.token_mode}"


def find_identity_span(prompt: str, form: str) -> tuple[int | None, int | None, str]:
    """Locate the identity surface form `form` in `prompt`. Returns (start_char,
    end_char, status). Status is 'exact' (regex match against the prompt as-is,
    case-insensitive), 'normalized' (matched after collapsing punctuation/whitespace),
    or 'failed' (form is empty or not present).

    Mirrors `find_identity_span` in scripts/extract_token_level_sae_activations.py
    so both scripts locate spans identically (audit 5.10 flags this duplication;
    a shared utility module would be the cleanup).
    """
    if not isinstance(form, str) or not form.strip():
        return None, None, "failed"
    match = re.search(re.escape(form), prompt, flags=re.I)
    if match:
        return match.start(), match.end(), "exact"
    norm_prompt = re.sub(r"[\W_]+", " ", prompt.lower())
    norm_form = re.sub(r"[\W_]+", " ", form.lower()).strip()
    norm_match = re.search(re.escape(norm_form), norm_prompt)
    if not norm_match:
        return None, None, "failed"
    prefix = norm_prompt[: norm_match.start()]
    compact_prefix = re.sub(r"\s+", "", prefix)
    compact_form = re.sub(r"\s+", "", norm_form)
    compact_prompt = re.sub(r"\s+", "", prompt.lower())
    compact_start = compact_prompt.find(compact_prefix + compact_form)
    if compact_start < 0:
        return None, None, "normalized"
    start = len(compact_prefix)
    return start, start + len(compact_form), "normalized"


def span_token_indices(
    offsets: np.ndarray,
    attention_mask_row: np.ndarray,
    span_start: int,
    span_end: int,
) -> list[int]:
    """Indices of tokens in one batch row whose char-offset interval overlaps
    [span_start, span_end). Only content tokens (start_char < end_char) and
    only positions inside the attention mask are returned."""
    indices: list[int] = []
    valid_len = int(attention_mask_row.sum())
    for token_idx in range(valid_len):
        start_char, end_char = int(offsets[token_idx, 0]), int(offsets[token_idx, 1])
        if end_char <= start_char:
            continue  # special tokens (BOS) have zero-width offsets
        if end_char > span_start and start_char < span_end:
            indices.append(token_idx)
    return indices


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


def precompute_span_locations(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    """Tokenize every prompt, locate the identity span, validate that the span
    survives tokenizer truncation, and return the per-row span info DataFrame.
    Raises on any failure so the long GPU run is not started against broken
    inputs.
    """
    if "form_used" not in df.columns:
        raise ValueError(
            "Input CSV must include a 'form_used' column for identity-span token "
            "modes; rerun data/create_dataset.py."
        )

    out: list[dict[str, object]] = []
    for start in tqdm(range(0, len(df), batch_size), desc="span pre-pass"):
        batch = df.iloc[start:start + batch_size]
        prompts = batch["prompt"].astype(str).tolist()
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        offsets = encoded["offset_mapping"]
        attention_mask = encoded["attention_mask"]
        for i, row in enumerate(batch.itertuples(index=False)):
            prompt = str(row.prompt)
            form = str(getattr(row, "form_used", "") or "")
            span_start, span_end, status = find_identity_span(prompt, form)
            if span_start is None:
                raise ValueError(
                    f"Could not locate form_used={form!r} in prompt for "
                    f"prompt_id={row.prompt_id!r}. Status: {status}. "
                    "Identity-span token modes require every prompt to contain "
                    "the form. Fix the input CSV or rerun with --token_mode final_token."
                )
            indices = span_token_indices(offsets[i], attention_mask[i], span_start, span_end)
            if not indices:
                raise ValueError(
                    f"Identity span for prompt_id={row.prompt_id!r} produced zero "
                    f"tokens after tokenization (span_chars=({span_start},{span_end}), "
                    f"max_length={max_length}). The span may be truncated; raise "
                    "--max_length or shorten the templates."
                )
            out.append({
                "prompt_id": row.prompt_id,
                "form_used": form,
                "span_status": status,
                "span_start_char": int(span_start),
                "span_end_char": int(span_end),
                "span_token_first": int(indices[0]),
                "span_token_last": int(indices[-1]),
                "span_n_tokens": len(indices),
                "span_token_indices": ",".join(str(i) for i in indices),
            })

    span_df = pd.DataFrame(out)
    assert len(span_df) == len(df), (
        f"Span pre-pass produced {len(span_df)} rows but expected {len(df)}."
    )
    return span_df


def build_span_aggregation_tensors(
    encoded: dict,
    span_df_batch: pd.DataFrame,
    token_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (gather_idx, span_mask) used in the per-layer aggregation step.

    For identity_span_last: gather_idx[b] is the last-token index of the span
    in row b; span_mask is unused (returned as empty).
    For identity_span_mean: span_mask[b, t] = 1 if token t is inside row b's
    span (and inside the attention mask); gather_idx is unused.
    """
    batch_size = encoded["attention_mask"].shape[0]
    seq_len = encoded["attention_mask"].shape[1]

    if token_mode == "identity_span_last":
        last = torch.tensor(span_df_batch["span_token_last"].to_numpy(), dtype=torch.long, device=device)
        empty = torch.zeros(0, device=device)
        return last, empty

    if token_mode == "identity_span_mean":
        mask = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
        for i, indices_str in enumerate(span_df_batch["span_token_indices"].tolist()):
            indices = [int(x) for x in indices_str.split(",") if x]
            indices = [j for j in indices if 0 <= j < seq_len]
            if not indices:
                raise RuntimeError(
                    "Span pre-pass and main loop disagree on tokenization. "
                    f"Row {i} has no in-range span tokens."
                )
            mask[i, indices] = 1.0
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, mask

    raise ValueError(f"Unsupported token_mode for span aggregation: {token_mode}")


def select_layer_activation(
    hidden_state: torch.Tensor,
    token_mode: str,
    final_idx: torch.Tensor,
    span_gather: torch.Tensor,
    span_mask: torch.Tensor,
) -> torch.Tensor:
    """Reduce a layer's (B, T, D) hidden state to (B, D) according to token_mode."""
    batch_size = hidden_state.shape[0]
    batch_indices = torch.arange(batch_size, device=hidden_state.device)
    if token_mode == "final_token":
        return hidden_state[batch_indices, final_idx.to(hidden_state.device), :]
    if token_mode == "identity_span_last":
        return hidden_state[batch_indices, span_gather.to(hidden_state.device), :]
    if token_mode == "identity_span_mean":
        m = span_mask.to(device=hidden_state.device, dtype=hidden_state.dtype)
        weighted = hidden_state * m.unsqueeze(-1)
        summed = weighted.sum(dim=1)
        counts = m.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return summed / counts
    raise ValueError(f"Unsupported token_mode: {token_mode}")


def extract_final_token_activations(
    *,
    df: pd.DataFrame,
    span_df: pd.DataFrame | None,
    token_mode: str,
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

    batch_starts = range(rows_written, num_prompts, batch_size)
    progress = tqdm(
        desc="Extracting batches",
        initial=0,
        total=num_prompts,
        unit="prompt",
    )
    progress.update(rows_written)

    with torch.inference_mode():
        for start in batch_starts:
            end = min(start + batch_size, num_prompts)
            batch_prompts = prompts[start:end]

            need_offsets = token_mode != "final_token"
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=need_offsets,
                return_tensors="pt",
            )
            if need_offsets:
                # offset_mapping is recomputed per batch but only used to validate
                # against the pre-pass; the gather indices come from span_df.
                encoded.pop("offset_mapping")
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

            if token_mode == "final_token":
                span_gather = torch.zeros(0, dtype=torch.long, device=input_device)
                span_mask = torch.zeros(0, device=input_device)
            else:
                assert span_df is not None
                # Need to translate ndarray -> torch on the input device.
                seq_len = encoded["attention_mask"].shape[1]
                fake_encoded = {"attention_mask": encoded["attention_mask"].new_zeros((batch_len, seq_len))}
                span_gather, span_mask = build_span_aggregation_tensors(
                    fake_encoded,
                    span_df.iloc[start:end],
                    token_mode,
                    input_device,
                )

            for layer_idx, hidden_state in enumerate(hidden_states):
                final_hidden = select_layer_activation(
                    hidden_state, token_mode, final_idx, span_gather, span_mask
                )

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
            progress.update(batch_len)

            del outputs, hidden_states, encoded, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress.close()

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

    output_dir = resolve_output_dir(args)
    prepare_output_dir(output_dir, args.overwrite, args.resume)

    df = load_prompts(args.input_csv, args.limit)
    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)

    torch_dtype = choose_torch_dtype()
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    span_df: pd.DataFrame | None = None
    if args.token_mode != "final_token":
        print(f"Token mode '{args.token_mode}': pre-locating identity spans for {len(df):,} prompts...")
        span_df = precompute_span_locations(df, tokenizer, args.max_length, args.batch_size)
        span_locations_path = output_dir / "span_locations.csv"
        span_df.to_csv(span_locations_path, index=False)
        print(f"  span_locations.csv: {span_locations_path}")
        print(f"  span_n_tokens: min={span_df['span_n_tokens'].min()}, "
              f"max={span_df['span_n_tokens'].max()}, "
              f"mean={span_df['span_n_tokens'].mean():.2f}")
        status_counts = span_df["span_status"].value_counts().to_dict()
        print(f"  span_status counts: {status_counts}")

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
        span_df=span_df,
        token_mode=args.token_mode,
        tokenizer=tokenizer,
        model=model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        resume=args.resume,
    )

    df.to_csv(metadata_path, index=False)

    assert len(df) == len(pd.read_csv(metadata_path, keep_default_na=False))

    run_config = {
        "model_path": str(args.model_path),
        "input_csv": str(args.input_csv),
        "output_dir": str(output_dir),
        "token_mode": args.token_mode,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_prompts": len(df),
        "num_layers_saved": num_layers_saved,
        "hidden_dim": hidden_dim,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    run_config_path = output_dir / "run_config.json"
    with run_config_path.open("w") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")

    print("Final output paths:")
    print(f"  layers: {output_dir / 'layer_XX.npy'}")
    print(f"  metadata: {metadata_path}")
    print(f"  run config: {run_config_path}")
    print(f"  checkpoint: {output_dir / CHECKPOINT_FILENAME}")
    if span_df is not None:
        print(f"  spans:    {output_dir / 'span_locations.csv'}")
    print(f"Token mode: {args.token_mode}")
    print(f"Saved rows: {len(df):,}")
    print(f"Saved layers: {num_layers_saved}")
    print(f"Hidden dim: {hidden_dim}")


if __name__ == "__main__":
    main()
