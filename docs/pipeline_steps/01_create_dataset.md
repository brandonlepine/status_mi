# Step 1 — `data/create_dataset.py`

**Stage:** 0 — Prompt and identity datasets
**Runs after:** — (entry point; consumes hand-curated CSVs)
**Feeds into:** [Step 4 — `extract_identity_activations.py`](04_extract_identity_activations.md) and, transitively, the entire identity-geometry pipeline.

## Purpose
Build the 12,567-row identity prompt corpus by cross-producting 100+ templates (10 families: A copula, B person-NP, C semantic null, D natural context, E group, F fragment, G plural, H past, I future, J position-shift) against 111 identities (10 axes). Each template specifies which morphological surface form (`adj_form`, `noun_form`, etc.) on the identity row to splice in. This is the substrate every Stage 1+ artifact is row-aligned to.

## Inputs
- `data/templates/mi_identity_templates.csv` — template rows: `template_id`, `family`, `template_text` (with `{form}` slot), `required_form` (the identity column to read), `number`.
- `data/bbq_identity_normalized_forms.csv` — 111 identities across 10 axes; each row has eight surface forms (`adj_form`, `noun_form`, `person_noun_form`, `plural_noun_form`, `group_form`, `prep_form`, `with_form`, `has_form`), an alias list, and `works_*` boolean compatibility flags.

## Outputs
- `data/mi_identity_prompts.csv` — 12,567 prompt rows with columns `prompt_id, template_id, family, template_text, required_form, form_used, number, identity_id, axis, canonical_label, prompt, template_notes, identity_notes`.
- `data/mi_identity_prompts_audit_<date>.csv` — (run-stamped) bad-pattern warnings.

## Key implementation details
- Template × identity cross-product; pair is dropped silently if `identity[required_form]` is empty (so e.g. an identity with no `adj_form` cannot realize copula templates).
- Final prompt is sentence-cased after `.format(form=form)` splicing.
- Output is sorted stably by `axis, identity_id, family, template_id` — this is the canonical row order every downstream `.npy` is aligned to.
- A bad-pattern regex (`"has is "`, `"A an "`, `"people are people"`, etc.) warns on suspicious outputs but does **not** drop them.
- The `works_*` flags on identity rows are **not** consulted; pair realization is purely "is the form column non-empty."

## Issues & Opportunities

### 4.5 [MINOR] — `works_*` template-compatibility flags are dead metadata

**What's wrong:** The identity CSV carries `works_is_adj`, `works_group`, `works_with`, etc. flags that are intended to signal which templates each identity is compatible with, but `create_dataset.py` ignores them. The "is the required form column non-empty" check is used as a proxy. In practice the two agree most of the time (forms tend to be blank exactly when the flag is 0), but the flags carry no enforcement weight.

**Why it matters:** The CSV gives the false impression of a compatibility constraint that is not actually enforced. If a future edit fills in a form column for an identity that should not realize that grammatical role, the bad row will silently appear in the prompt corpus.

**Targeted fix:** Either (a) use the `works_*` flags as the single source of truth in the realization check and treat empty form cells as an error, or (b) delete the `works_*` columns from the identity CSV and document that form-column presence is the rule.

## Rebuild checklist
- [ ] Decide whether `works_*` flags are authoritative or vestigial; pick one and remove the other.
- [ ] If keeping `works_*`: add an assertion that `form_present == works_flag` for every (identity, template family) pair, and fix any mismatch.
- [ ] Promote the bad-pattern warning into an explicit audit artifact written next to the prompt CSV on every run (not just on the `2026-04-27` run that is checked in).
- [ ] Add a tiny sanity check that asserts the output prompt CSV row count equals the expected `sum_over_templates(n_identities_with_required_form)` so silent drops cannot grow unnoticed.

## Notes from the doc audit
- The script raises `ValueError` if a template's `required_form` is not a column on the identity CSV — good guard. The complementary case (a column exists but is mostly empty so most identities silently drop) is the one that needs the explicit count assertion above.
