# Scoring method: purpose-specific profiles (first profile: office wall prints)

## Context

The pipeline so far (ingest → GPU scores → `rank.py`) picks "generally best" photos, using hard filters plus an
additive z-score composite. Kept photos are already deduplicated into bursts and copies and grouped into events. The
user wants different selection criteria at different times. The first purpose is **framed prints for an office**:
- No prominent faces, individuals or small groups. Crowds, or a few distant figures in a landscape, are fine.
- It must read well **from a distance**, with some large-scale structure.
- It must be in focus, at least in part.
- Sunsets and similar clichés are too boring.
- Animals, landscapes and natural abstracts (rocks, moss, clouds) are good.

Rather than a chain of filters, the user wants a score in which one badly deficient metric drags a photo down even
when everything else is excellent. The open question is how to combine metrics whose scales and relative weights
are uncertain.

## Recommended aggregation method (answers the "best practice" question)

1. **Put every metric on a common scale using percentile rank** (the empirical CDF over the candidate pool), as the
   user suggested. This is robust to incomparable units and to monotone transforms.
   - Caveat: ranks exaggerate tiny differences in metrics where almost every photo is fine (e.g. sharpness).
   - So step 2 maps each percentile, or for absolute metrics the raw value, through a desirability curve.
2. **Desirability functions** (Derringer–Suich, the standard tool in multi-response optimization) map each metric to
   d ∈ [ε, 1]. A "must not be bad" metric, such as focus or prominent people, gets a ramp: d≈ε when unacceptable and
   1 once adequate, with no further credit above that. A "more is better" metric, such as aesthetic or distance
   readability, rises across its whole range. The floor ε (e.g. 0.05) keeps it a score, not a filter.
3. **Combine with a weighted power mean** M_p = (Σ wᵢ dᵢ^p)^(1/p).
   - The exponent p is the "how compensatory" knob: p=1 is the additive mean, p→0 the geometric mean (the
     Derringer–Suich default), p=−1 the harmonic mean, and p→−∞ the minimum.
   - The user's "n-th worst metric" idea is the OWA (ordered weighted average) family. The power mean with p≈−1 gives
     the same "one bad metric hurts" behaviour while staying smooth and weightable.
4. **Handle uncertainty in the weights and in p by Monte Carlo** (SMAA, stochastic multicriteria acceptability
   analysis).
   - Draw ~2,000 weight vectors from a Dirichlet distribution centred on the nominal weights, and p from [−2, 0.5].
   - Re-rank each time and record each photo's **probability of landing in the top K**.
   - Select by that probability. Photos that win under most plausible weightings rank above photos that need one
     particular weighting. The per-photo spread also shows where the method is unsure.
5. **Calibrate against the user's own picks once there are labels.** The user marks ~200 photos "yes/no for the
   office" in the review page; Claude's grades of the shortlist add more labels. Then:
   - Fit the weights by logistic regression on log dᵢ. log of the geometric mean is linear in log dᵢ, so the fitted
     coefficients are the weights.
   - Train a CLIP linear probe ("looks like what I pick for the office") as one more metric. This works well with a
     few hundred labels.
6. **Diversity is a selection step, not a metric.** Pick the final set greedily by maximal marginal relevance: next
   photo = argmax(score − λ·max CLIP similarity to photos already picked). This replaces the per-event quotas for
   this profile, so 20 near-identical Dolomites views can't crowd out everything else.

## Metrics for the office profile

| Metric | Source | Type | Notes |
| --- | --- | --- | --- |
| Aesthetic | LAION + TOPIQ-IAA (have) | more-is-better | percentile |
| Technical quality | TOPIQ-NR (have) | must-not-be-bad | ramp on percentile |
| **Focus somewhere** | new: tile-max Laplacian on the **original** at full resolution | must-not-be-bad | stage 2 only; shallow DOF is fine if some tile is sharp |
| **Prominent people** | new: torchvision Faster R-CNN (MobileNet-v3) person boxes on thumbnails | must-not-be-bad | d=1 if the largest person box < ~2% of frame, or a crowd (≥8 people, none large); ramp down to ε at ~15% |
| **Reads from a distance** | new: LAION aesthetic of the image shrunk to ~48 px and re-enlarged, plus coarse-scale contrast/edge energy from the thumbnail | more-is-better | simulates viewing across a room |
| Subject fit | CLIP prompt ensembles, each scored as a binary contrast vs a neutral "a photo" | more-is-better | landscape/vista, animal/wildlife, natural abstract texture (rock, moss, lichen, clouds, ice, water, bark) |
| Cliché | CLIP: sunset/sunrise, generic tourist snapshot | must-not-be-bad (soft) | mild ramp, not a veto |
| Someone else's art / junk | CLIP artwork + junk labels (have) | must-not-be-bad | replaces the current hard junk exclusion for this profile |
| Print size | megapixels | must-not-be-bad | ramp: ε below ~4 MP, 1 above ~10 MP (16×20 in at ~150–200 dpi) |

Only structural steps stay as hard steps: removing duplicates and bursts (one shot per moment), and dropping
non-photos (GIFs, camera-less PNG screenshots).

## Two-stage funnel (keeps the expensive metrics cheap)

- **Stage 1, all photos.** Existing scores, plus the person detector and coarse-structure metrics. The detector takes
  ~20 min per 18k on the GPU; the coarse metrics take ~1 min on the CPU. Compute the profile score and keep the top
  ~2,000.
- **Stage 2, shortlist only.** Full-resolution focus map from the originals (~3,000 reads from F:, ~5–10 min) and the
  shrunk-image aesthetic (~5 min). Rescore → Monte Carlo top-K probability → diversity selection → ~300 candidates.
- **Stage 3.** Claude reviews the ~300 on contact sheets for "office suitability", giving a grade plus a reason, and
  that grade becomes one more desirability. The user then reviews and labels, and the labels feed calibration (step 5
  above).

## Code changes

- `profiles/office.yaml` (new). Lists the metrics, each with a source column, percentile or raw, desirability curve
  (type, lo, hi, ε), weight and stage. Also holds `p`, the Monte Carlo settings (Dirichlet concentration, p range,
  draws, K), the diversity λ and the target count. A future profile is a new YAML, not new code.
- `photosel/profile.py` (new). Loads a profile, builds the metric table, computes percentiles, desirabilities, power
  mean, Monte Carlo top-K probability and diversity selection. Writes `profile_scores(profile, sha1, score,
  p_topk, d_<metric>…, selected)`.
- `photosel/detect.py` (new). Person detection on thumbnails using `torchvision.models.detection`, which is already
  installed, in fp32. Stores n_persons, max_person_area and total_person_area in a `persons` table.
- `photosel/structure.py` (new). Coarse-scale metrics from thumbnails, plus the stage-2 focus map from originals,
  run niced with ≤12 workers.
- `photosel/embed.py`: add a `small_view_aesthetic` pass over a sha1 list by reusing `run_clip`'s model and
  `_aesthetic_model`.
- `photosel/zeroshot.py`: add prompt ensembles and a binary-contrast scoring function next to `label_probs`.
- Reuse what exists:
  - from `rank.py`: `bursts`, `merge_copies` and `burst_winners`, to pick one winner per moment before scoring;
  - from `embed.py`: `load_embeddings`;
  - from `sheets.py`: `make_sheets`, with captions showing score, p_topk and the weakest metric.
- CLI: `python -m photosel detect`, `structure`, `profile office [--stage 1|2]`, and
  `sheet --profile office` (ordered by p_topk). The general-purpose `rank` stays as it is.

## Verification

- Unit tests:
  - desirability curves and the power mean, including that p→0 matches the geometric mean and that p→−∞ approaches
    the minimum;
  - Monte Carlo reproducibility with a fixed seed;
  - that a photo excellent on everything but one floor-level metric ranks below a photo that is uniformly good;
  - the diversity pick on synthetic embeddings.
- On the 17.8k photos:
  - contact sheets of the top 40 for the office profile, a random 20 from ranks 300–600, and the 20 photos with the
    highest aesthetic score but low office score, each captioned with its weakest metric;
  - expect no portraits or small groups, sunsets only rarely, and readable large shapes.
- Check that tilting the nominal weights ±50% leaves most of the top 100 in place; the Monte Carlo spread should show
  this directly.

## Open choices (defaults in brackets)

- Do architecture and cityscapes count as good subjects for the office, or are they neutral? [neutral]
- Largest print size planned [16×20 in, so about 10 MP is "fully adequate"].
- Final number wanted for this round [~30–50 prints chosen from ~300 candidates].
