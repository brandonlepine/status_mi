# Triage Pre-Registration — 2026-05-27

**Purpose.** Audit 5.2 of `docs/issues_and_opportunities.md` flagged that the SAE feature triage in `scripts/triage_sae_identity_features.py` was a chain of hand-picked weights and thresholds that had never been validated against any external criterion. This document pre-registers the cascade, score weights, and keep rule **before** the corrected (audit 1.4) SAE encodings and BBQ steering results are available, so the rule cannot be tuned to the causal outcome. Any change to the constants in this document after a BBQ run is published must be justified explicitly, with the date and rationale.

The corresponding code is fixed as of commits `7f2c302` (firing-count entropy), `235b5f5` (soft scoring head), and `f306869` (sensitivity sweep). Run from a clean checkout pointing at these SHAs reproduces the pre-registered behavior.

---

## 1. What the triage is for, and what it is NOT

**It is** a *feature-selection* step: given a corpus of SAE features that survive upstream selectivity / decoder-alignment / activation-stats filtering, produce a `keep_for_intervention` flag and ranking that downstream BBQ steering uses to decide which features to intervene on. The kept-feature count and ranking are an honest pre-BBQ choice.

**It is NOT** a finding. The four soft role labels — `identity_token_local`, `sentence_final_integrated`, `shared_social_feature`, `contrast_specific_identity` — are **descriptive**. They surface in the triage HTML and in `provisional_role` for human inspection and grouping, but they should not appear in the paper's causal claims unless one of the validation paths in §5 returns a positive result.

The audit's framing was right: a 7-branch first-match cascade with hand-picked weights and thresholds is a series of stylistic decisions, not a measurement. The reframing is that the *kept-feature count* and the *single-threshold keep rule* are the load-bearing parts; the taxonomy is decoration unless validated.

---

## 2. Frozen score weights

Four weighted-sum scores are computed in `scripts/triage_sae_identity_features.py:compute_scores` from upstream signals. The weights below are the `DEFAULT_SCORE_WEIGHTS` dict in that file (module-level constant). All inputs are pre-clipped to `[0, 1]`; outputs are post-clipped to `[0, 1]`.

### 2.1 `contrast_specificity_score`

Intent: a feature is contrast-specific if it concentrates on one or a small number of axes, has its activation skewed toward its top axis, and has a strong selectivity effect size.

```
contrast_specificity_score =
    0.6 · (1 − min(1, (n_axes_where_top_feature − 1) / 4))    # axes_inverse
  + 0.2 · top_axis_fraction
  + 0.2 · min(1, max_abs_cohens_d / 2)                         # d_capped
```

### 2.2 `sharedness_score`

Intent: a feature is shared if it appears as a top feature on multiple axes and multiple contrasts, and (optionally) has substantial shared-PC loading.

```
sharedness_score =
    0.5 · min(1, n_axes_where_top_feature / 5)
  + 0.3 · min(1, n_contrasts_where_top_feature / 10)
  + 0.2 · clip(shared_pc_loading_score, 0, 1)
```

### 2.3 `template_artifact_score`

Intent: a feature is template-driven (rather than identity-driven) if its top tokens are template words, if it concentrates on a single family or template, or if it does NOT localize to identity-span tokens.

```
template_artifact_score =
    0.4 · fraction_top_tokens_template_words
  + 0.3 · (1 − family_entropy)
  + 0.2 · (1 − template_entropy)
  + 0.1 · (1 − identity_span_localization_score)
```

`family_entropy` and `template_entropy` are categorical entropy of the family / template distribution among the feature's top-activating prompts, normalized to `[0, 1]`.

### 2.4 `polysemanticity_score`

Intent: a feature is polysemantic if its firing is diffuse across axes, identities, and tokens, and not concentrated on one top axis.

```
polysemanticity_score =
    0.35 · axis_entropy
  + 0.35 · identity_entropy
  + 0.20 · token_entropy
  + 0.10 · (1 − top_axis_fraction)
```

`axis_entropy`, `identity_entropy`, `token_entropy` are computed as Shannon entropy of **firing-count** categorical distributions (audit 5.2 part 1, commit `7f2c302`):

- `axis_entropy`: categorical entropy of the axis-distribution of prompts on which the feature fired.
- `identity_entropy`: categorical entropy of per-identity firing counts (`freq_identity × n_identity`). The implicit probability model is "given the feature fired somewhere, what is the probability it fired in identity i."
- `token_entropy`: categorical entropy of per-token-string firing counts among the feature's top tokens.

The prior implementation treated activation magnitudes as a probability mass, which is not motivated by any probability model.

---

## 3. Frozen keep rule (single threshold, post-soft-head)

```
keep_for_intervention =
        (not is_low_signal)                                 # both d and cos below their min thresholds
    AND (not is_template_artifact)                          # template_artifact_score >= 0.5
    AND (max(role_fit_*) >= 0.5)                            # --min_role_fit_keep
    AND (max_abs_cohens_d >= 0.5)                           # --min_abs_cohens_d
```

The four role-fit scores (each clipped to `[0, 1]`):

```
role_fit_identity_token_local       = mean(span_score, norm_d, 1 − template_artifact_score)
role_fit_sentence_final_integrated  = mean(final_score, norm_d, 1 − template_artifact_score)
role_fit_shared_social_feature      = mean(sharedness_score, min(n_axes_where_top_feature / 3, 1),
                                            1 − template_artifact_score)
role_fit_contrast_specific_identity = mean(contrast_specificity_score, norm_d, norm_cos,
                                            1 − template_artifact_score)
```

with normalized versions:

```
norm_d   = clip01(max_abs_cohens_d         / (2 · --min_abs_cohens_d))
norm_cos = clip01(max_abs_decoder_cosine   / (4 · --min_abs_decoder_cosine))
```

The descriptive `provisional_role` is `argmax(role_fit_*)`, with overrides to `low_signal`, `template_or_syntax_artifact`, or `polysemantic_or_unclear` when the corresponding hard flags fire or when no role-fit reaches `--min_role_fit_keep`. The audit's first-match-cascade objection — a feature scoring 0.71 on span_local and 0.85 on sharedness was permanently labeled `identity_token_local` — is closed by the soft head.

### 3.1 Frozen CLI defaults

| Flag | Default | Used for |
| --- | --- | --- |
| `--min_abs_cohens_d` | `0.5` | low-signal gate + keep rule |
| `--min_abs_decoder_cosine` | `0.03` | low-signal gate + role_fit_contrast_specific_identity normalizer |
| `--identity_span_local_threshold` | `0.7` | informational only (no longer a cascade gate; retained for the sensitivity sweep) |
| `--final_token_integrated_threshold` | `0.7` | informational only |
| `--max_template_artifact_score_keep` | `0.5` | hard ceiling: features with `template_artifact_score >= 0.5` are dropped from `keep_for_intervention` |
| `--min_contrast_specificity_keep` | `0.5` | informational only |
| `--min_sharedness_score_shared` | `0.5` | informational only |
| `--min_role_fit_keep` | `0.5` | keep rule threshold on `max(role_fit_*)` |

The first two and the artifact ceiling are the *load-bearing* knobs. The four informational thresholds are kept for the sensitivity sweep but no longer participate in the keep decision.

---

## 4. What gets reported, and how

- **Kept-feature count by axis.** Single sentence in the methods section: "Of `N_total` features surviving upstream filters, `N_kept` were retained for BBQ intervention under the rule in §3."
- **Sensitivity table.** `triage_sensitivity_summary.csv` (one row per perturbation): role-change fraction and keep-change fraction across the perturbations defined by `--sensitivity_perturb_fractions` (default `±10%, ±20%`). Reported in the methods appendix; the worst-case perturbation is reported in the main text as a single number.
- **Role taxonomy.** Tabulated under "Descriptive feature profile" in the appendix, with a disclaimer that the labels are argmax of soft role-fit scores and have not been validated against an external criterion (unless §5 returns positive).

---

## 5. Validation paths for the taxonomy

The role taxonomy enters the paper's findings **only** if at least one of the following returns positive. Both are deferred until BBQ steering results are available.

### 5.1 Behavioral criterion — position-conditional causal effect

The audit's falsifiable criterion: under the audit-3.1 feature-level intervention (encode → modify-latent-f → decode → patch), `identity_token_local` features and `sentence_final_integrated` features must produce a **double dissociation** across the two BBQ intervention positions:

- `identity_token_local` features: larger causal effect at `target_identity_last_token` than at `final_prompt_token`.
- `sentence_final_integrated` features: larger causal effect at `final_prompt_token` than at `target_identity_last_token`.

The test is a paired contrast of the absolute `bias_margin_delta` (or `stereotyped_delta`) across the two positions, evaluated on `keep_for_intervention = True` features, stratified by `provisional_role`. The test passes if, for both relevant role groups, the median per-feature delta-of-effects has the predicted sign at `p < 0.05` (signed-rank test).

The other two role labels (`shared_social_feature`, `contrast_specific_identity`) do not have a clean position-conditional prediction; their criterion is dropped to inter-rater agreement only.

### 5.2 Inter-rater criterion — human labelers

Two labelers independently label a stratified sample of `N = 80` features (20 per role) from the feature cards (Step 15 output) according to a fixed rubric. The cascade label and the human label must agree at Cohen's κ ≥ 0.6 (substantial agreement) for the role to be reported as a finding. The rubric and the sample-feature list are themselves pre-registered as a sub-document of this file when the cards are generated post-1.4 re-encode.

---

## 6. Frozen-as-of

- **Date:** 2026-05-27
- **Code commits:** `7f2c302` (entropy), `235b5f5` (soft head), `f306869` (sensitivity sweep)
- **Upstream data state:** the (audit-1.4-corrected) SAE re-encode is **not yet run**. BBQ steering is **not yet run** post-3.1 fix. This document precedes both.

Any modification to §2, §3, or §3.1 after either of those data sources is materialized must be recorded as an appended changelog entry below, with the date and the specific BBQ result that motivated the change. Changes made without such a record render the kept-feature set un-publishable as a pre-registered selection.

---

## Changelog

(none yet)
