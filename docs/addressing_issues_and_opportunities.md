# Addressing Issues and Opportunities Identified

This is in reference to the document currently located here:

```Python
"/Users/brandonlepine/Repositories/Research_Repositories/status_mi/docs/issues_and_opportunities.md"
```

## Foundational Measurement Issues

### 1.1 Identity Representation is Measured at the Final Token Which is Almost Always a Period

#### Summary of the Issue

`extract_identity_activations.py` stores the per-layer residual stream at `attention_mask.sum(dim=1) - 1` which corresponds to the last non-padding token. However, every template ends with `.`. After tokenization the final token is the sentence-final period in essentially every prompt. 

The implicit assumption based on this is that the final token integrates the identity content of the prompt, but for a *base* model this is untested. A base LM's final-token residual is optimized to predict the next token, not to summarize the sentence. 

#### Potential Resolution

1. Run the geometry analysis on:
    - the last token of the identity span
    - the mean over identity-span tokens
    - and compare to the final token result
2. Report which location carries the signal and justify it. If final token does carry it, that is also a valid and interesting finding to report. 

### 1.2 Base Model vs. Multiple-Choice QA Benchmark

#### Issue Summary

BBQ is a QA benchmark, and `prepare_bbq_for_steering.py` builds a zero-shot prompt ending in "Answer:". But base models are weak at and often off-distribution for this format. 

If the base model puts 1-2% total mass on the three potential answer options and 98% on the continuation text, the logprob deltas we steer are in a degenerate regime and "bias" is barely defined. 

#### Potential Resolution

1. Report, at baseline:
    a. total probability mass on the three answer options
    b. the standard BBQ accuracy and bias score for Llama-3.1-8B-Base in this prompt format
    c. how often the argmax-over-options matches the model's actual greedy continuation. 
    These are preconditions
2. Few-shot prompt (3-5 BBQ exemplars) to pull the base model onto the task distribution before steering. 
    - it might even be interesting to have the few shot examples condition for stereotypical behavior and compare with non-stereotypical few-shot examples. 
3. If the behavior is too degenerate, we should either accept that and frame the results around logprob margins (which are still defined), or reconsider scope. 

### 1.3 First-Token Answer Scoring is Degenerate for BBQ Answers

#### Issue Summary

`run_bbq_sae_steering.py:score_first_token` scores the log probability of the first token of the asnwer text (`first_token_ids` tokenizes `" " + answer`). 

BBQ answers are noun phrases and many begin with the same word (`"The grandmother", "the boy", "Cannot be determined"`). When two of the three options share a first token, their first-token logprobs are identitical and the metric cannot distinguish them. 

**Separately** the prompt presentes labelled choices `A. / B. / C.` and ends with `Answer:`. The natural model continuation is the letter (`A / B / C`), but scoring targets the answer **text**. So there is a mismatch between prompt design and scoring target. 

#### Potential Resolutions

1. Score the answer **letters** `A / B / C` (single tokens, mutually distinct, matched to the prompt format). This is as fast as current first-token scoring and removes degeneracy. 

2. Keep `answer_logprob` as a confirmatory mode, but length-normalize it 

3. Rerun, treating first-token-text results as preliminary. 


### 1.4 SAE Preprocessing Conventions Are Unverified

#### Issue Summary

`encode_identity_saes.py:load_sae` is explicitly a *generic* loader. It heuristically picks: 
    - `W_enc`
    - `W_dec`
    - `b_enc`
    - `b_dec`
by name/shape and applies `relu(x - b_dec) @ W_enc + b_enc`. 

LlamaScope SAEs apply an input normalization step (scaling the residual activation to a fixed norm) before encoding. If the generic loader skips that normalization, every SAE feature activation in the project is computed on mis-scaled input. 

#### Potential Resolution: 

Inspect the OpenMOSS SAE Hyperparamter and configuration files. 

```Python

# for Llama3_1-8B-Base-LXR-32X/LLama3_1-8B-Base-L24R-32X/hyperparameters.json:

{
    "device": "cuda:0",
    "seed": 42,
    "dtype": "torch.bfloat16",
    "hook_point_in": "blocks.24.hook_resid_post",
    "hook_point_out": "blocks.24.hook_resid_post",
    "use_decoder_bias": true,
    "apply_decoder_bias_to_pre_encoder": false,
    "expansion_factor": 32,
    "d_model": 4096,
    "d_sae": 131072,
    "bias_init_method": "all_zero",
    "act_fn": "jumprelu",                       # So they use jumprelu which is a different activaiton function
    "jump_relu_threshold": 0.75390625,
    "norm_activation": "dataset-wise",
    "dataset_average_activation_norm": {
        "in": 29.125,
        "out": 29.125
    },
    "decoder_exactly_fixed_norm": false,
    "sparsity_include_decoder_norm": true,
    "use_glu_encoder": false,
    "init_decoder_norm": 0.5,
    "init_encoder_norm": null,
    "init_encoder_with_decoder_transpose": true,
    "lp": 1,
    "l1_coefficient": 8e-05,
    "l1_coefficient_warmup_steps": 78125,
    "top_k": 50,
    "k_warmup_steps": 78125,
    "use_batch_norm_mse": true,
    "use_ghost_grads": false,
    "tp_size": 1,
    "ddp_size": 1
}
```

- also add a numerical reconstruction check to `validate_sae_hook_alignment.py`: encode then decode a sample of real activations and confirm reconstruction error (FVU/cosine) matches the SAE's reported quality. 

### 1.5 Activations are bf16-precision stored as float32

----
## Statistical Rigor Issues 

### 2.1 Headline Contrast AUC / Cohen's d are in-sample and circular

#### Issue Description

`analyze_identity_geometry.py:run_constrasts` and `analyze_identity_geometry_diagnostics.py:run_contrasts` compute the contrast direction from `mean(A) - mean(B)`, then evaluate AUC/Cohen's d of the projection on the same A and B prompts. 

In-sample separate is optimistically biased - a difference-of-means direction is *defined* to separate the two means. 

The family-holdout variants (`contrast_family_holdout_scores.csv`, `contrast_family_holdout_residualized_scores.csv`) are the honest tests. But the in-sample `auc_all` and `cohens_d_all` columns are what gets plotted as the headline "contrast AUC by layer"

#### Potential Resolutions

1. Demote in-sample AUC to a clearly-labeled diagnostic or remove it. Held-out AUC (cross-template, cross-family) should be the headline number everywhere. 

2. For the shared-subspace decomposition, we should evaluate shared/residual components with the direction estimated on held-out prompts too. 

### 2.2 Non Null Model for the Central Claims 

The geometry probes (`crossval_probe`) report accuracy/macro-F1 but never a label-permutation null. 

The shared-subspace SVD reports a singlar-value spectrum but never compares it to the spectrum of **random directions** from shuffled identity labels. Without a null: 

- "Identity is linearly decodable" - high cross-validation accuracy could partly reflect group structure / template leakage rather than identity content. 
- "There is a shared social subspace" - any set of about 19 unit vectors in 4096-d has some SVD spectrum; concentration only means something relative to a null. As currently written, the "shared subspace" claim is not yet supported. 

#### Potential Resolution

- Probes: add a permutation null (shuffle `identity_id` / `axis`labels within the grouping structure, re-run CV, repeat more than 100x). Report observed accuracy as a z-score / empirical p against that null. 

- Shared subspace: build directions from shuffled identity assignments (or from random splits of each axis), re-SVD, and compare the real spectrum's concentration (e.g., participation ratio, or variance in top-k) to the null distribution. Only then is "shared subspace" a finding. 

### 2.3 Steering Controls are Disabled in the Production Run

`run_bbq_sae_steering.py` implements three controls - `sign_flip`, `random_direction_norm_matched`, and `random_feature_matched` - but they are gated behind `--disable_controls` and the documented production command passes this CLI argument. 

Without the controls we cannot make claims that a feature's effect is **specific**. A norm-matched random direction at the same position may shift the bias margin just as much (steering vectors of any kind perturb logits). The whole "feature X is causally implicated in bias" claim needs: effect(feature X) >> effect(random_direction) >> effect(random feature set), at matched norm. 

#### Potential Resolutions

- Re-enable controls for the final run. If cost is the issue, we can run controls on a stratefied subsample of examples x features rather than dropping them. 

- add one more control: the **raw difference-of-means contrast direction** from the geometry pipeline, steered identically. If SAE features do not beat the difference-of-means direction, the SAE is not adding causal value over a linear probe and that comparison must be in the paper. 

### 2.4 `answer_logprob` Is Summed Over Different-Length Answers

`score_answer_logprob` sums per-token logprobs over the answer span. BBQs three options have different token lengths `"Cannot be determined"` is typically the longest, so summed logprob systematically penalizes the unknown option. 

`within-example deltas (intervened - base) cancel the length bias because length is consistent per example - so `stereotype_preference_delta` etc. are OK. But `predicted_base`, `correct_base`, `prediction_changed`, and `accuracy_delta` use argmax over **raw** summed logprobs, which is length based. Baseline accuracy and any accuracy change metric are contaiminated. 

#### Potential Resolutions

Length normalize (mean per-token logprob) for any argmax/accuracy metric, or score the answer letter (1.3) which has consistent length and dissolves the problem. 
