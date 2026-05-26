# Conceptual Workflow: Identity Representation and Bias Mechanisms in Llama-3.1-8B

This document maps the end-to-end pipeline as it currently exists in the repository. It is descriptive — it records what each script does, what it consumes, what it produces, and how those artifacts feed downstream. A separate doc (`issues_and_opportunities.md`) records what is wrong, weak, or worth changing.

The project has two large pipelines that share a substrate:

1. **Identity-geometry pipeline** — builds a controlled set of templated identity prompts, extracts final-token residual activations from Llama-3.1-8B, then analyzes the geometry of identity in those activations (variance decomposition, contrast directions, shared-subspace SVD, residualized PCA/probes, family-stability cosines, SAE feature selectivity/decoder alignment, feature triage, feature cards).
2. **BBQ steering and causal-feature pipeline** — takes the triaged identity-selective SAE features from pipeline 1, runs them as causal interventions on the BBQ bias benchmark, and analyzes the resulting answer-probability shifts at multiple grouping levels (feature, identity, contrast, axis).

The two pipelines are coupled through three persistent assets:

- `data/mi_identity_prompts.csv` — the templated identity prompts.
- `results/.../layer_XX.npy` final-token activations and `results/.../layer_XX/feature_*.npy` SAE encodings.
- `results/.../triage/intervention_candidate_features_triaged.csv` — the catalog of "kept-for-intervention" features that drives BBQ steering.

Below, each script is described in execution order. Most absolute paths use the RunPod root `/workspace/status_mi` because that is where the GPU runs live; local Mac copies sit under `output_data/`.

---

## Stage 0 — Prompt and identity datasets

### `data/create_dataset.py`

**Inputs**

- `data/templates/mi_identity_templates.csv` — 100+ template rows across 10 template families (A copula, B person-NP, C semantic null, D natural context, E group, F fragment, G plural, H past, I future, J position-shift). Each row has `template_id`, `family`, `template_text` (with `{form}` slot), `required_form` (which morphological column on the identity to use), and `number`.
- `data/bbq_identity_normalized_forms.csv` — 111 identities across 10 axes (`disability_status`, `gender_identity`, `nationality`, `physical_appearance`, `race_ethnicity`, `religion`, `sexual_orientation`, `socioeconomic_status`, `age`, `sex`). Each identity has eight surface forms (`adj_form`, `noun_form`, `person_noun_form`, `plural_noun_form`, `group_form`, `prep_form`, `with_form`, `has_form`), plus alias semicolon-list and `works_*` boolean compatibility flags.

**Action**

- Cross-product templates × identities, picking the `required_form` column per template; skip identity/template pairs where the form is empty (so identities only realize through forms they actually have).
- Sentence-case the realized string. Sort by `axis, identity_id, family, template_id`.
- Spot-check generated prompts against a small set of bad regex patterns (`"has is "`, `"A an "`, `"people are people"`, etc.) and warn but do not drop.

**Outputs**

- `data/mi_identity_prompts.csv` — 12,567 prompt rows with columns `prompt_id, template_id, family, template_text, required_form, form_used, number, identity_id, axis, canonical_label, prompt, template_notes, identity_notes`.
- A separate `data/mi_identity_prompts_audit_2026-04-27.csv` records the bad-pattern warnings from that run.

**Downstream**

- `extract_identity_activations.py` reads `data/mi_identity_prompts.csv` (assumes `prompt` and `prompt_id` columns exist; asserts no empty prompts).

---

## Stage 1 — Identity activations and SAE encodings

### `scripts/download_llama_3_1_8b.py`

Snapshot-downloads `meta-llama/Llama-3.1-8B` from HF into `/workspace/status_mi/models/llama-3.1-8b/`. Requires HF auth. Wrapper around `huggingface_hub.snapshot_download`.

### `scripts/download_openmoss_saes.py`

Downloads OpenMOSS LlamaScope SAEs (`OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x` by default) for the requested layer numbers, scoring repo file names against regex layer patterns and `resid` markers and picking the highest-scoring matches plus their sibling configs. Saves a `download_manifest.json` of what was pulled.

### `scripts/extract_identity_activations.py`

**Inputs**

- `data/mi_identity_prompts.csv` (must contain `prompt` and `prompt_id`).
- Llama model directory.

**Action**

- Loads tokenizer (right-padding, pad=eos), loads model with `output_hidden_states=True`, dtype = bf16 → fp16 → fp32 depending on hardware.
- For each batch: tokenizes prompts (truncate to `--max_length` 128), runs forward, takes `outputs.hidden_states` (which is a tuple of `n_layers + 1` tensors: index 0 is embeddings, index k≥1 is post-block-k residual).
- For each layer k, extracts the **final non-padding token** activation (`attention_mask.sum(dim=1) - 1`), writes to `layer_{k:02d}.npy` as a memmapped 2-D float32 array `(n_prompts, hidden_dim)`.
- Persists `metadata.csv` (a copy of the prompt CSV), `run_config.json`, and a `checkpoint.json` that supports `--resume`.

**Outputs (per run)**

```
results/activations/llama-3.1-8b/identity_prompts_final_token/
  layer_00.npy ... layer_32.npy           # final-token residual stream per layer
  metadata.csv                            # row-aligned with all .npy files
  checkpoint.json
  run_config.json
```

**Downstream**

- Geometry analyses (Stage 2) and SAE encoding (`encode_identity_saes.py`).

### `scripts/encode_identity_saes.py`

**Inputs**

- `results/activations/.../layer_XX.npy` for requested layers.
- `saes/openmoss/Llama3_1-8B-Base-LXR-32x/` (auto-discovers per-layer weight files and configs).

**Action**

- Loads weight files (.safetensors), heuristically identifies `w_enc`, `w_dec`, `b_enc`, `b_dec` by name and shape (encoder is `(hidden_dim, n_features)`, decoder is `(n_features, hidden_dim)`).
- Computes SAE activations: `relu((x − b_dec) @ w_enc + b_enc)`. Comment in the code notes this is a "generic loader" and that OpenMOSS may have a slightly different preprocessing convention.
- For each row, keeps top-k (default 64) features and stores both indices and values.
- Saves per-layer: top-k feature indices and values, decoder weights (and bias), per-feature activation stats (`activation_count`, `activation_frequency`, `mean_activation_nonzero`, `mean_activation_all`, `max_activation`, `p95`/`p99`), the resolved SAE config, and optionally a dense `top-N` feature matrix.

**Outputs**

```
results/sae_identity/llama-3.1-8b/final_token/
  layer_XX/
    feature_indices_top64.npy             # (n_prompts, 64) int32
    feature_values_top64.npy              # (n_prompts, 64) float32
    sae_decoder.npy                       # (n_features, hidden_dim)
    sae_decoder_bias.npy
    sae_config_resolved.json
    feature_stats.csv
    metadata.csv                          # copy of upstream metadata
```

### `scripts/validate_sae_hook_alignment.py`

A safety check that parses LlamaScope-style filenames (`L<layer><position>-<width>x`) and asserts:

- The SAE checkpoint layer matches the requested layer.
- The SAE's "position" tag is `R` (residual stream).
- The SAE input dim matches the activation hidden_dim.

Writes `hook_alignment_validation.json` and `.csv`. Will raise unless `--allow_mismatch`. Includes a literal HF-convention note: `hidden_states[k]` is the post-block-k residual; final-norm/`lm_head` are NOT applied.

---

## Stage 2 — Identity-geometry analyses

These all consume the final-token activations + metadata produced by Stage 1.

### `scripts/analyze_identity_geometry.py` — first-pass geometry

For every layer (or selected layers):

- **PCA**: StandardScaler + sklearn PCA on a stratified sample of prompts (`--max_pca_points`, stratified by `axis||family`). Saves PC scores and explained-variance ratio.
- **Group means**: identity, identity×family, axis centroids (saved as `.npy` + metadata `.csv`).
- **Logistic-regression probes** (group-K-fold CV): predicts (a) `axis` grouped by `template_id` and `family`; (b) `identity_id` within each axis, grouped by `template_id`. Feature space is `--probe_pca_dim`-reduced (default 256) standardized activations. Logs mean ± SD accuracy and macro-F1.
- **Family stability**: per identity, cosine similarity between family-conditional centroids (both raw and globally centered).
- **Contrasts** (`CONTRASTS` list, e.g. `race_black vs race_white`, `sexuality_gay vs sexuality_straight`, etc.): compute the difference-of-means contrast direction in centered activations, score every prompt by projection, report `auc_all`, `cohens_d`, mean projection per identity, and a **family-holdout AUC/d** (train direction on examples outside a held-out family, evaluate on that family).
- For PROJECTION_LAYERS (`{0,8,16,24,32}`), also save per-prompt projection scores.

**Outputs** (under `results/geometry/.../`):

```
pca/pca_layer_XX.csv, pca/pca_explained_variance.csv
means/identity_means_layer_XX.npy + metadata, identity_family_means_..., axis_means_...
probes/axis_probe_scores.csv, identity_within_axis_probe_scores.csv
family_stability/family_cosines_layer_XX.csv, family_cosines_summary.csv
contrasts/contrast_scores.csv, contrast_family_holdout_scores.csv
contrasts/contrast_projection_scores_layer_XX.csv  (for PROJECTION_LAYERS)
run_config.json
```

### `scripts/analyze_identity_geometry_diagnostics.py` — second-pass diagnostics

A larger pass that asks: does identity geometry survive controls for prompt surface form?

For each layer:

- **Variance decomposition (η²)**: between-group SS / total SS for `family`, `template_id`, `required_form`, `axis`, `identity_id`. Writes one row per (layer, factor) to `variance_decomposition.csv`.
- **Residualization variants**: `raw`, `family_residualized` (subtract per-family mean, add back global), `template_residualized`, `required_form_residualized`. For each residualization:
  - PCA per layer + explained-variance log.
  - Identity probes (axis prediction grouped by template/family; identity-within-axis prediction grouped by template_id).
  - Surface-form probes (on raw only): predict `required_form` and `family` from activations, grouped by `identity_id`.
  - Contrasts: full-data AUC/d and family-holdout AUC/d.
- Static plots: variance decomposition by layer, probe scores by layer/residualization, contrast AUC by layer, per-axis PCA scatters at selected layers (with progression panels), and optional UMAP.

Supports `--resume` (reuses incremental CSVs) and a number of `--skip_*`/`--only_*` flags for partial runs. Probe model is `LogisticRegression(class_weight='balanced')`, default solver `saga`.

**Outputs**

```
diagnostics/
  variance_decomposition.csv
  pca_residualized/{raw,family_residualized,template_residualized,required_form_residualized}/pca_layer_XX.csv + pca_explained_variance.csv
  probes/axis_probe_residualized_scores.csv, identity_within_axis_probe_residualized_scores.csv, surface_form_probe_scores.csv
  contrasts/contrast_full_residualized_scores.csv, contrast_family_holdout_residualized_scores.csv
  figures/...
  run_config.json
```

### `scripts/plot_identity_geometry.py` and `plot_identity_directional_visualizations.py`

Pure plotting layers over the previous CSVs:

- `plot_identity_geometry.py` reads the first-pass CSVs (`pca/`, `probes/`, `family_stability/`, `contrasts/`) and produces PCA scatters per layer, axis-centroid scatters, probe-score line plots, family-stability heatmaps, contrast-AUC by layer, plus optional UMAPs.
- `plot_identity_directional_visualizations.py` re-computes contrast directions itself (re-loads activations, residualizes, runs `compute_direction` and family-holdout eval), writes its own `metrics/layerwise_contrast_metrics.csv`, `family_to_family_generalization.csv`, `centroid_ordering.csv`, `direction_cosines/*.csv`, and produces projection-distribution plots, layer curves, direction-cosine summaries, centroid-distance scatters, and 2-D plane scatters using two orthogonalized contrast directions.

### `scripts/plot_identity_directional_followups.py`

A still-larger followup-plotting script that re-derives contrast directions and produces:

- **Centroid-ordering plots**: where every same-axis identity falls on the gay-straight (or analogous) axis, with bootstrap CIs.
- **Family-to-family generalization heatmaps**: train direction on family A, evaluate on family B.
- **Direction-stability curves**: cosine of contrast direction between adjacent layers and vs a reference layer (default 24).
- **2-D plane scatters** (e.g. `sexual_orientation` plane using two contrast directions, second one orthogonalized to the first).
- A 6-panel "paper-ready summary panel": variance decomposition, key contrast AUC, two axis planes, centroid ordering, direction-cosine heatmap.

These scripts are heavy duplications of the analysis code in `analyze_identity_geometry_diagnostics.py` plus their own metric definitions — they are best read as standalone figure-generators.

### `scripts/analyze_shared_social_subspace.py`

Decomposes identity contrast directions into a **shared social subspace** plus contrast-specific residuals.

For each (layer, residualization):

1. For each `(identity_a, identity_b)` contrast in `DEFAULT_CONTRASTS`, compute the centered difference-of-means direction `d_c`, unit-normalize, and sign-flip so identity_a has the larger projection.
2. Stack the contrast directions into a matrix `D ∈ ℝ^(C×hidden_dim)` and run SVD: `D = U Σ Vᵀ`.
3. Treat the top-k right singular vectors as the basis of a "shared subspace." For each contrast direction, decompose into `shared = Vᵀ (V d)` and `residual = d − shared`; record `shared_norm`, `residual_norm`, cosine with full direction. Re-evaluate AUC/d/accuracy of each component on the endpoint identities.
4. PC interpretation: for each PC, project identity centroids onto it, save top/bottom identities. Also save per-contrast loadings on each PC.
5. Cross-axis projection: project all identity centroids (across all axes) onto each contrast direction, summarize per axis.

**Outputs**

```
shared_subspace_decomposition/
  metrics/shared_subspace_spectrum.csv
  metrics/decomposition_metrics.csv
  metrics/axis_sharedness_summary.csv
  metrics/shared_pc_identity_rankings.csv
  metrics/shared_pc_top_bottom.csv
  metrics/contrast_pc_loadings.csv
  metrics/cross_axis_projection_summary.csv
  metrics/cross_axis_identity_projections.csv
  figures/{spectrum,decomposition,axis_summary,pc_interpretation,pc_loadings,cross_axis,paper_panels}/...
```

---

## Stage 3 — Identity-selective SAE feature analysis

### `scripts/analyze_identity_sae_features.py`

The bridge from "geometric directions" to "individual SAE features."

For each layer:

- Loads the SAE top-k indices/values produced by `encode_identity_saes.py`. Builds a sparse long table `(row_idx, feature_id, activation>0)`.
- Loads the residualized activation matrix (default `family_residualized`) and the SAE decoder.
- For each identity (`identity_selectivity`): compares activation distributions for that identity vs all other same-axis identities. Picks top features by `|diff_mean|`, then re-ranks by `|cohen's d|` and AUC. Writes per-feature `mean_identity`, `mean_other`, `freq_identity`, `freq_other`, `cohens_d`, `auc`.
- For each contrast in `DEFAULT_CONTRASTS` (a fixed list of identity pairs):
  - **Feature selectivity** (`feature_selectivity_for_contrast`): same idea but for the contrast pair. Records `mean_a`, `mean_b`, `freq_a`, `freq_b`, `diff_mean`, `cohens_d`, `auc`, plus ordinal ranks.
  - **Decoder alignment** (`decoder_alignment`): cosine between each decoder row and the centered difference-of-means direction `d_c`; also signed dot product. Ranks features by `|cosine|`.
  - **Joined** table merges selectivity + alignment, adds `combined_score = z(|d|) + z(|cosine|) + z(|auc − 0.5|)`.
  - **Direction reconstruction** (`reconstruction_rows`): given different feature selection methods (`decoder_alignment`, `selectivity`, `combined_score`, `random_baseline`), pick top-k decoder rows, project the contrast direction onto the span (treating rows as if orthonormal: `recon = (basis @ d) @ basis`), unit-normalize, and re-score AUC/d. Records `cosine_with_full_direction`, `fraction_norm_captured`, and post-reconstruction AUC/d for `k ∈ {5,10,20,50,100,200}`.
  - **Intervention candidates** (`intervention_candidates`): top-N features by `combined_score` with a `direction_side` flag and "recommended_intervention: ablate".

**Outputs** (per layer, appended):

```
analysis/feature_identity_selectivity.csv
analysis/feature_selectivity.csv
analysis/decoder_direction_alignment.csv
analysis/feature_selectivity_alignment_joined.csv
analysis/direction_reconstruction.csv
analysis/intervention_candidate_features.csv
analysis/run_config.json
```

### `scripts/extract_token_level_sae_activations.py`

Picks features from the analysis CSVs (top per contrast by `combined_score`, plus top per identity by `|d|`, plus user `--features`). Then, for the prompts that had high final-token activation on each feature:

- Re-runs the model on those prompts with `output_hidden_states=True`.
- Encodes the chosen-layer hidden state through the layer's SAE (`relu((x − b_dec) @ w_enc[:, features] + b_enc[features])`).
- For each (prompt, feature), records per-token activation, whether the token is inside the identity span (regex-located in the prompt), localization type (`identity_span_local`, `final_token_integrated`, `template_context`, `diffuse_or_unclear`).

**Outputs**

```
feature_cards/token_level/
  selected_features.json
  layer_XX/
    token_feature_activations.csv         # one row per (prompt, feature, token)
    feature_top_tokens.csv                # top tokens per feature (non-special)
    run_config.json
```

### `scripts/build_sae_feature_cards.py`

Builds standalone HTML "feature cards" for selected SAE features. For each feature:

- Pulls identity-mean activations and selectivity/alignment stats from analysis CSVs.
- Loads token-level activations, classifies prompts by localization type, picks exemplar prompts and identity-span tokens.
- Auto-labels the feature using the highest-activating identity and localization signal.
- Optionally computes a "raw logit-lens" projection: `unembed @ decoder_row` and lists top-positive/top-negative tokens. (Note in the code: "Raw decoder @ lm_head projection; final norm not applied.")
- Saves `feature_<id>.html`, `feature_<id>.json` per feature, plus an identity-profile bar plot and a token-exemplar matplotlib figure.

**Outputs**

```
feature_cards/
  index.html
  feature_card_index.csv
  layer_XX/feature_XXXXX.html + .json + .png
```

### `scripts/plot_identity_sae_features.py`

Bulk plotting over the analysis CSVs: per-axis feature-effect summary plots and supporting figures (this is largely a presentation layer on the same CSVs the triage uses).

### `scripts/triage_sae_identity_features.py`

The single most important glue script in this pipeline: it consumes everything the SAE analysis and token-level extraction produced and assigns each SAE feature a **provisional role** plus a `keep_for_intervention` flag.

For each layer, builds the following per-feature aggregates:

- **Signal metrics** (from `feature_selectivity_alignment_joined.csv` + `intervention_candidate_features.csv` + `decoder_direction_alignment.csv`): `max_abs_cohens_d`, `max_auc_distance_from_0_5`, `max_abs_decoder_cosine`, `max_combined_score`, top contrast(s) by selectivity / decoder alignment, `n_contrasts_seen`, `n_axes_seen`, `n_identities_seen`, `signal_top_axis`.
- **Membership** (from `intervention_candidate_features.csv`): in how many contrasts × axes was this feature in the top-N? Saves as `n_contrasts_where_top_feature`, `n_axes_where_top_feature`.
- **Identity specificity** (from `feature_identity_selectivity.csv`): for which identity does this feature fire most? `top_identity`, `top_axis`, `top_axis_fraction` (fraction of top-10 identities in the same axis), `axis_entropy`, `identity_entropy`, `top_identities_by_activation`.
- **Token-level localization** (from `token_feature_activations.csv` if present): `identity_span_localization_score` (median ratio of max-span activation to max-token activation across exemplar prompts), `final_token_integration_score` (same for final token), `fraction_top_tokens_template_words`, `family_entropy`, `template_entropy`, `cross_axis_activation_score`, and a final `feature_localization_type ∈ {identity_span_local, final_token_integrated, template_context, diffuse_or_unclear}`.
- **Shared-subspace loading** (optional, from `analyze_shared_social_subspace.py` metrics): a 0–1 `shared_pc_loading_score`.

Combines these into derived scores:

- `contrast_specificity_score = 0.6 (1 − min(1, (n_axes_top − 1)/4)) + 0.2 top_axis_fraction + 0.2 min(1, max|d|/2)`
- `sharedness_score = 0.5 min(1, n_axes_top/5) + 0.3 min(1, n_contrasts_top/10) + 0.2 shared_pc_loading_score`
- `template_artifact_score = 0.4 fraction_top_template_words + 0.3 (1 − family_entropy) + 0.2 (1 − template_entropy) + 0.1 (1 − identity_span_localization_score)`
- `polysemanticity_score = 0.35 axis_entropy + 0.35 identity_entropy + 0.20 token_entropy + 0.10 (1 − top_axis_fraction)`

The role-assignment cascade (`assign_roles`):

1. If `max|d| < min_abs_cohens_d` and `max|cos| < min_abs_decoder_cosine` → **low_signal**, drop.
2. Else if `template_artifact_score ≥ max_template_artifact_score_keep` → **template_or_syntax_artifact**, drop.
3. Else if `identity_span_localization_score ≥ 0.7` and `max|d| ≥ threshold` → **identity_token_local**, keep.
4. Else if `final_token_integration_score ≥ 0.7` and `max|d| ≥ threshold` → **sentence_final_integrated**, keep.
5. Else if `sharedness_score ≥ 0.5` and `n_axes_top ≥ 3` → **shared_social_feature**, keep iff `|d|` and artifact pass.
6. Else if `contrast_specificity_score ≥ 0.5` and `|d|, |cos|` pass → **contrast_specific_identity**, keep.
7. Else → **polysemantic_or_unclear**, drop.

Final `keep_for_intervention` requires the role to be a keep role AND `|d|` and artifact thresholds satisfied. Priority is `high` if confidence ≥ 0.7 and `|d| ≥ 1.5 × threshold`, else `medium`/`low`.

**Outputs**

```
triage/
  feature_triage.csv                              # all features × all aggregated metrics
  intervention_candidate_features_triaged.csv    # per-contrast candidates × role/keep
  feature_triage_summary.csv
  role_counts.csv
  triage_index.html                               # filterable HTML table
  figures/{role_counts, score_distributions, role_by_axis_heatmap, keep_for_intervention_by_contrast, scatter_selectivity_vs_artifact, scatter_sharedness_vs_specificity}.{png,pdf}
  intermediate/{signal_metrics, top_feature_membership, identity_specificity_metrics, token_localization_metrics, shared_subspace_scores, feature_metric_table_pre_roles}.csv
  triage_config.json
```

The single most consequential downstream consumer of this file is the BBQ steering pipeline.

---

## Stage 4 — BBQ steering and feature-level causal analysis

### `scripts/prepare_bbq_for_steering.py`

**Inputs**

- `data/bbq/data/*.jsonl` — the BBQ benchmark (per-category JSONL, e.g. `Age.jsonl`, `Race_ethnicity.jsonl`). Each row has `example_id`, `question_index`, `question_polarity` (`neg`/`nonneg`), `context_condition` (`ambig`/`disambig`), `category`, `answer_info` (mapping `ans0/1/2` → `[text, group_label]`), `additional_metadata.stereotyped_groups`, `context`, `question`, `ans0/1/2`, `label`.
- `data/bbq_identity_normalized_forms.csv` — identity alias source.
- `results/.../triage/intervention_candidate_features_triaged.csv` — used only to derive a `contrast → axis` mapping so that BBQ contrasts can be aligned with available SAE contrast directions.

**Action**

For each BBQ row, this script:

1. Maps the raw BBQ `category` to a project axis using `AXIS_MAP` (with `race_x_gender` and `race_x_ses` collapsing to `race_ethnicity`).
2. Tries to align each of `ans0/1/2`'s `group_label` to a project `identity_id` by checking, in order: full-text aliases derived from the identity-forms CSV, a curated `MANUAL_ALIASES` dict (which contains many duplicate "nondisabled" entries — copy-paste artifact), compact (whitespace-removed) matching, and finally a component-decomposition fallback for compound labels like `F-Black`. The `UNKNOWN_ALIASES` set captures the "cannot be determined" answer.
3. Identifies (a) `unknown_answer_idx`, (b) `stereotyped_answer_idx` (the answer whose group label is in `stereotyped_groups`), (c) `nonstereotyped_answer_idx` (the remaining non-unknown answer). Picks `target_identity_id` and `nontarget_identity_id` preferring matches inside the stereotyped-group set and same-axis.
4. Maps the resulting `(target_id, nontarget_id, axis)` to a `mapped_contrast_name` from the SAE triage's contrasts: exact pair, then axis-fallback if no exact match, otherwise `"unmapped"`.
5. Builds the prompt: `"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer choices:\nA. {ans0}\nB. {ans1}\nC. {ans2}\n\nAnswer:"`.
6. Logs notes (`missing_unknown_answer`, `missing_stereotyped_answer`, `unmapped_contrast`) per row.

**Outputs**

```
prepared/
  bbq_prepared_examples.csv + .parquet
  bbq_mapping_diagnostics.csv           # rows with notes
  bbq_contrast_mapping.csv              # available SAE contrasts × identity_a/b/axis
  bbq_prepare_summary.csv               # mapping-coverage metrics
  bbq_prepare_config.json
  logs/prepare_bbq.log
```

### `scripts/extract_bbq_token_level_sae_activations.py`

Re-runs the model on the prepared BBQ prompts and encodes them through the layer-24 SAE (default) for the kept features from triage.

For each batch of BBQ prompts:

- Tokenize with offset mapping, run forward, take `outputs.hidden_states[layer]` (post-block-`layer`).
- Encode through SAE for the chosen feature subset: `relu((x − b_dec) @ w_enc[:, features] + b_enc[features])`.
- For each token in each prompt, record activation, character offsets, and a set of role flags: `is_target_identity_token`, `is_nontarget_identity_token`, `is_stereotype_language_token` (matched against question content words minus a stopword list), `is_question_token`, `is_context_token`, `is_answer_option_token`, `answer_option_idx`, `is_unknown_answer_token`, `is_final_prompt_token`, plus the feature's `feature_role`, `feature_contrast_name`, `feature_top_axis/top_identity`.
- Append to per-batch parquet files with a `manifest.csv` for resume.

**Outputs**

```
token_level_sae/
  token_activations/layer_XX/part_XXXXX.parquet + manifest.csv
  bbq_token_level_sae_summary.csv
  token_level_config.json
```

### `scripts/build_bbq_sae_feature_cards.py`

Same idea as `build_sae_feature_cards.py` but consuming the **BBQ** token-level activations and the prepared BBQ examples. Produces per-feature HTML cards with BBQ exemplars, token-role classifications (target/nontarget/stereotype-language/option), and a behavioral classification (e.g., where the feature fires most). Filters punctuation-only and template-stopword tokens out of the top-token rankings (this is the "filtered" variant of the cards that replaces the earlier `feature_cards/`).

### `scripts/run_bbq_sae_steering.py`

The causal-intervention engine.

**Inputs**

- Llama model, SAE directory.
- Prepared BBQ parquet/CSV.
- Triaged features CSV (selects the feature pool).

**Feature-set construction** (`load_feature_sets`):

- Filters by `keep_for_intervention = True` and the requested layers.
- Builds one of three set modes per row, depending on `--feature_set_modes`:
  - `per_feature`: one set per (layer, feature_id). One vector per feature. This is the only mode that supports clean individual-feature causal claims.
  - `per_contrast_topk`: top-k by priority/role/score per `(layer, contrast_name)`, both per-role and combined.
  - `role_bundle`: all features in a (layer, top_axis, role) bucket.
- Always adds a `template_or_syntax_artifact` "control bundle" (top-20 artifact features per layer).
- Each `FeatureSet` carries `feature_ids`, `signs` (from `direction_side` / sign of `cohens_d` / `cosine`), `axis`, `contrast_name`, etc.

**Axis matching** (`--axis_match_mode`): default `matched_only`. A feature set is only applied to BBQ examples whose `axis_mapped` equals the feature set's axis. `all` runs every set on every example (used historically for wrong-axis controls; produced uninterpretable smoke output).

**Vector construction** (`make_vector`):

```
dec = w_dec[feature_ids]            # (n_features_in_set, hidden_dim)
dec = dec / ||dec||_row             # row-normalize (unless --no_normalize_features)
vec = mean(signed * dec, axis=0)    # weighted mean
vec = vec / ||vec||                 # unit-normalize the resulting direction
```

So the steering "direction" is always unit-norm, and `alpha` is the magnitude of the additive perturbation in hidden-state units.

**Hooks** (`install_hook` / `install_batched_hook`):

- Registered on `model.model.layers[layer - 1]` — i.e., the transformer block whose output is `hidden_states[layer]`.
- Two modes:
  - `add_vector`: `h[:, pos, :] += alpha * vec` at the chosen token positions.
  - `ablate_projection`: `h[:, pos, :] -= alpha * (h[:, pos, :] · unit) * unit`.

**Intervention positions** (`positions_for`):

- `final_prompt_token`: the last non-pad token.
- `target_identity_last_token` / `nontarget_identity_last_token`: tokens overlapping the regex-located identity label (and answer option text containing it). Picks the last such token.
- `stereotype_language_last_token`: tokens overlapping content words in the BBQ question.
- `all_identity_tokens` / `all_stereotype_language_tokens`: every such token.

**Scoring** (`score_first_token` vs `score_answer_logprob`):

- `first_token`: logprob of the first token of `" {answer}"` for each of the 3 options at the final prompt position. Fast, batchable.
- `answer_logprob`: appends `" {answer}"`, runs forward, sums per-token logprobs over the answer span. Slower (one forward per option), but uses the actual continuation likelihood.

**Per (example, feature_set, alpha, position, intervention_mode) "job"**:

1. Compute baseline score for the 3 answers with no hook.
2. Install the hook, recompute scores ⇒ `inter`.
3. Compute deltas: per-answer logprob deltas, `stereotyped_delta`, `nonstereotyped_delta`, `unknown_delta`, `correct_delta`, `bias_margin_delta` (ambig: `Δ[log p(stereotype) − log p(unknown)]`; disambig: `Δ[correct − max(others)]`), `prediction_changed`, `correct_base`/`correct_intervened`.
4. Append to `results_parts/part_XXXXX.parquet`.

**Controls** (skipped by `--disable_controls`, which the current default long run uses): a `sign_flip` of the same feature vector, a `random_direction_norm_matched` random unit vector in hidden space, and a `random_feature_matched` — a random set of decoder rows of the same cardinality. All at `final_prompt_token`.

Job IDs are SHA1-prefix-16 hashes over `(bbq_uid, layer, set_id, alpha, position, mode, scoring_mode)`; `completed_jobs.jsonl` enables resume.

**Outputs**

```
steering/
  steering_config.json
  steering_manifest.csv                 # one row per feature set with eligibility counts
  results_parts/part_XXXXX.parquet
  completed_jobs.jsonl
  logs/steering.log
```

### `scripts/analyze_bbq_steering_results.py` — aggregate analysis (legacy)

Reads `results_parts/*.parquet`, enriches with derived quantities, aggregates at axis/contrast/feature_set×alpha levels, computes bootstrap CIs, writes interpretation docs and overview figures. Marked in `docs/bbq_steering_pipeline.md` as **not** the final feature-level analysis — useful only for high-level sanity plots.

**Outputs**

```
analysis/
  steering_results_merged.csv
  coverage_report.csv
  README_interpretation.md
  SMOKE_TEST_LIMITATIONS.md
  interpretation_summary_by_axis.csv, _by_contrast.csv, _by_feature_set_alpha.csv
  figures/
```

### `scripts/analyze_bbq_feature_level_causal_effects.py` — feature/identity/contrast/axis-level causal analysis

This is the substantive feature-level analyzer.

**Pipeline**:

1. Load `results_parts/*.parquet`, validate, filter by `--layers/--alphas/--positions/--context_conditions`.
2. `enrich_results`: compute per-row metrics including
   - `stereotype_preference_delta = Δ[log p(stereotyped) − log p(unknown)]`
   - `nontarget_preference_delta`, `identity_substitution_delta = Δ[log p(nonstereo) − log p(stereo)]`
   - `correct_margin_delta` (disambig: `Δ[log p(correct) − max log p(others)]`)
   - `stereotype_error_delta`, `accuracy_delta`
   - `steering_direction_label ∈ {feature_amplification, feature_suppression}` from sign of `alpha`.
3. `expand_feature_rows`: for bundle rows, emit one row per `feature_id` with `feature_estimate_type = feature_bundle_membership`. For per-feature rows, `individual_feature`.
4. `build_identity_records`: for each BBQ example, emit identity-role records (`target`, `nontarget`, `stereotyped_identity`, `nonstereotyped_identity`) with the matching `answer_idx_for_identity`.
5. `merge_identity_records`: join expanded rows to identity records, compute `identity_answer_delta` (logprob shift of that identity's answer) and `identity_specific_bias_delta = identity_answer_delta − unknown_delta`.
6. `summarize_effects` at several group-col tuples:
   - feature-level: group by (feature_id, layer, alpha, position, role, contrast, axis_mapped, context_condition, polarity, target/nontarget identity, estimate_type, steering_direction).
   - subgroup-level: also by mapped_contrast_name × target_identity.
   - identity-level: per (feature × identity_id).
   - In each case, per-example aggregation first (mean over rows with same `bbq_uid`), then per-group bootstrap CI (`bootstrap_ci`) and sign-flip permutation p-values (`sign_flip_pvalue`). FDR (Benjamini-Hochberg) is applied **within** `(axis_mapped, context_condition, alpha, intervention_position)` strata.
7. `effect_label`: heuristic taxonomy — `bias_amplifying`, `bias_reducing_uncertainty`, `bias_reducing_substitution`, `general_answer_suppression`, `capability_degrading` (disambig + correct degradation), `identity_only`, `mixed_or_unclear`, `no_reliable_effect`. Effect-size labels (`tiny`/`small`/`moderate`/`large`) are based on absolute `stereotype_preference_delta` against three thresholds.
8. `make_rankings`: strongest bias-reducing/amplifying/unknown-increasing/substitution/capability-degrading features.
9. `make_reports`: per-axis, per-contrast, per-identity HTML/figure reports.

**Outputs**

```
feature_level_causal_analysis/
  merged_results.csv + .parquet
  deltas_long.csv + .parquet
  identity_records.csv
  identity_deltas_long.csv
  feature_level_pre_fdr.csv
  feature_level_effects.csv + .parquet      # post-FDR, with metadata merged in
  subgroup_level_effects.csv
  identity_level_effects.csv
  feature_x_subgroup_matrix.csv
  feature_effect_rankings.csv
  validation_summary.csv
  final_intervention_candidates_table.html
  feature_card_links_table.html
  analysis/axis_reports/<axis>/...
  analysis/contrast_reports/<contrast>/...
  analysis/identity_reports/<identity>/...
  figures/...
  README.md                                  # sign convention + statistics description
  logs/, feature_level_causal_config.json
```

---

## Cross-pipeline dependency diagram (condensed)

```
data/templates/mi_identity_templates.csv ──┐
data/bbq_identity_normalized_forms.csv ────┴──> data/create_dataset.py
                                                   |
                                                   v
                                  data/mi_identity_prompts.csv
                                                   |
                                                   v
                                  extract_identity_activations.py
                                                   |
                                                   v
              results/.../identity_prompts_final_token/{layer_XX.npy, metadata.csv}
                       |                                        |
        +--------------+-----------+                            +----------------+
        |              |           |                                             |
        v              v           v                                             v
 analyze_identity_  analyze_   analyze_shared_                      encode_identity_saes.py
   geometry.py     identity_    social_subspace.py                              |
                   geometry_                                                    v
                   diagnostics.py                          results/sae_identity/.../layer_XX/
        |              |           |                            {feature_indices, feature_values,
        v              v           v                             sae_decoder.npy, feature_stats.csv}
   plot_identity_     diagnostics  shared_subspace                             |
   geometry.py        figures      decomposition CSVs                          v
   plot_identity_                  + figures                       analyze_identity_sae_features.py
   directional_*.py                                                            |
                                                                               v
                                          analysis/{feature_selectivity, decoder_direction_alignment,
                                                    feature_identity_selectivity, intervention_candidate_features,
                                                    direction_reconstruction, feature_selectivity_alignment_joined}.csv
                                                                               |
                                          extract_token_level_sae_activations.py
                                                                               |
                                          feature_cards/token_level/...        |
                                                                               v
                                                              triage_sae_identity_features.py
                                                                               |
                                                              triage/intervention_candidate_features_triaged.csv
                                                                               |
                              +------------------------------------------------+
                              |                                                |
                              v                                                v
                  prepare_bbq_for_steering.py                build_sae_feature_cards.py (identity-prompt cards)
                              |
                              v
                  bbq_prepared_examples.parquet
                              |
                  +-----------+----------------+
                  v                            v
   extract_bbq_token_level_                run_bbq_sae_steering.py
   sae_activations.py                          |
                  |                            v
                  v             steering/results_parts/*.parquet + completed_jobs.jsonl
   token_activations/...                       |
                  |                            v
                  v             analyze_bbq_feature_level_causal_effects.py
   build_bbq_sae_feature_cards.py             |
                  |                            v
                  v          feature_level_causal_analysis/{feature_level_effects, identity_level_effects,
   feature_cards_filtered/...                    subgroup_level_effects, rankings, axis/contrast/identity reports,
                                                 final_intervention_candidates_table.html}
                                                ^
                                                |
                              (consumes triage CSV for feature metadata merge,
                               token_level dir for activation summary)
```

---

## Key invariants the pipeline assumes

- **HF hidden-states convention**: `outputs.hidden_states[k]` is post-block-`k` for `k ≥ 1` (the OpenMOSS `LkR` SAE for the same `k`). Final-norm and `lm_head` are NOT applied. The hook code registers on `model.model.layers[layer - 1]` to inject after that block, matching this convention.
- **Right-padding**: extraction code asserts right padding so `attention_mask.sum(dim=1) - 1` is the last content token.
- **Final-token semantic**: all of the geometry analysis builds its identity representation as the residual stream at the final non-padding token of a templated prompt. The implicit claim is that this token integrates the identity content of the prompt.
- **Difference-of-means as the "identity contrast direction"**: every direction analysis (geometry contrasts, decoder alignment, shared subspace) uses unit-normalized centered mean(A) − mean(B). No supervised probe weight is used as a "direction."
- **Per-row top-k SAE encoding**: only the top-64 feature indices per row are stored after encoding. Anything outside the top-64 is treated as zero downstream. Per-row sparse → per-feature aggregates depend on this.
- **Axis matching at steering time**: BBQ examples are only paired with feature sets whose `axis` matches `axis_mapped`. The default `--axis_match_mode matched_only` enforces this; `all` is for explicit wrong-axis sweeps.
- **Per-feature mode is the only path to single-feature causal estimates**: `per_contrast_topk` and `role_bundle` runs are explicitly labeled `feature_bundle_membership` in the downstream analyzer and should not be used to support individual-feature causal claims.

---

## How a clean end-to-end run looks today (RunPod paths)

```bash
# 0. One-time
python scripts/download_llama_3_1_8b.py
python scripts/download_openmoss_saes.py --layers 24
python data/create_dataset.py

# 1. Identity activations and SAEs
python scripts/extract_identity_activations.py
python scripts/encode_identity_saes.py --layers 24
python scripts/validate_sae_hook_alignment.py --layers 24

# 2. Geometry (any subset)
python scripts/analyze_identity_geometry.py
python scripts/analyze_identity_geometry_diagnostics.py --skip_umap
python scripts/analyze_shared_social_subspace.py

# 3. SAE feature pipeline
python scripts/analyze_identity_sae_features.py --layers 24
python scripts/extract_token_level_sae_activations.py --layers 24
python scripts/build_sae_feature_cards.py
python scripts/triage_sae_identity_features.py --layers 24

# 4. BBQ pipeline
python scripts/prepare_bbq_for_steering.py
python scripts/extract_bbq_token_level_sae_activations.py --layers 24
python scripts/build_bbq_sae_feature_cards.py
python scripts/run_bbq_sae_steering.py \
  --layers 24 --feature_set_modes per_feature --require_per_feature \
  --axis_match_mode matched_only --scoring_mode first_token \
  --intervention_positions final_prompt_token,target_identity_last_token,stereotype_language_last_token
python scripts/analyze_bbq_feature_level_causal_effects.py --layers 24
```

The `docs/bbq_steering_pipeline.md` document already records the production RunPod commands with the `nohup`/`resume` invocations used for long runs.
