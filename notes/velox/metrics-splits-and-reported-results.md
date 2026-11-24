# VELOX — metrics, splits and reported results

Companion to [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md).
This note carries the chapter's metric contribution: **what ADP measures**, **how the reported
epoch is chosen**, **what the E3 table actually says**, and **whether the real-time claim is in
the artifact**.

Analysed against the artifact, `upstream/velox` @ `54f687c` (2026-09-16) — see the companion
note for why `main` is the wrong target. Paper text: `resources/VELOX.txt`. Measurements: `code/scratchpad/velox_adp_probe.py`
(probes V1–V6, deterministic, no database required — `python3 code/scratchpad/velox_adp_probe.py`).
Provenance markers as in the companion note.

---

## 0. Answer in one paragraph

ADP is a reasonable idea — score a detector by how many *distinct attacks* it surfaces at a
given precision, rather than by node-level recall — and this thesis should take it seriously as
a triage metric. But as implemented it has three properties that need stating before any ADP
number is quoted. It is **not monotone in detection quality**: a detector that finds all 300
malicious nodes can score 0.25 lower than one that finds three, because the curve is anchored at
an artificial origin and the resulting interpolation penalises a high precision floor
`[measured]`. It **rewards losing ground truth**: an attack whose uuids fail to resolve leaves
the denominator rather than counting as missed `[measured]`. And it sits under **three nested
maxima**, all resolved on the test split — an envelope over thresholds, an argmax over epochs,
and the "Best" column's max over five seeds `[read]`. The last of these is what the headline
claim rests on, and the paper's own SC5 recommends reporting the mean instead.

---

## 1. ADP, as implemented

`[read]` `src/detection/evaluation_methods/evaluation_utils.py:280-335`. The reported
`adp_score` is the return value of `plot_detected_attacks_vs_precision`. Sort all scored nodes
by descending anomaly score; walk the ranking; at each prefix record `precision = tp/(tp+fp)`
and `detected% = |attacks with ≥1 TP so far| / total_attacks × 100`; collapse duplicate
precision values keeping the maximum detected%; integrate with `np.trapz` and divide by 100.
The logic is identical on `main`, so every probe below applies to both; the only difference is
that the artifact updates `detected_attacks` outside the true-positive branch (`:305-314`)
rather than inside it, which is equivalent whenever `node2attacks` holds only ground-truth
nodes `[read]`.

Two implementation details that do the work:

```python
detected_attacks_percentages = [0]   # :298   the artificial origin
precisions = [0]                     # :299
...
precision_to_attacks[precision] = max(precision_to_attacks[precision], detected_percentage)  # :326
...
area_under_curve = np.trapz(max_detected_attacks_percentages, unique_precisions) / 100        # :335
```

`total_attacks` (`:296`) is derived from `node2attacks`, i.e. from ground-truth attacks that
resolved to at least one node — see §4 of
[`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md).

## 2. What the probes show

All figures `[measured]`, `code/scratchpad/velox_adp_probe.py`.

**V1 — ADP depends on how many benign nodes were scored.** Hold the detection fixed (one node
from each of 3 attacks ranked at the very top, perfect precision) and vary only the benign
population:

| benign nodes | p_min | ADP |
| ---: | ---: | ---: |
| 3 | 0.5000 | 0.750 |
| 9 | 0.2500 | 0.875 |
| 47 | 0.0600 | 0.970 |
| 297 | 0.0100 | 0.995 |
| 2997 | 0.0010 | 1.000 |

Identical detections, ADP from 0.750 to 1.000. The cause is the seed point at
`(precision=0, detected=0)`: `np.trapz` interpolates linearly from there up to the smallest
precision the ranking actually reaches, carving out a triangular hole whose area grows with
that floor. `p_min` is the malicious fraction of the scored node set — a property of windowing,
filtering and dataset size, not of the detector.

**V2 — ADP is not monotone in detection quality.** Two systems, same 597-node test set:

| System | detections | p_min | ADP |
| --- | --- | ---: | ---: |
| A | 3 malicious nodes at top, 594 benign below | 0.0050 | **0.997** |
| B | all 300 malicious at top, perfectly ranked, 297 benign | 0.5025 | **0.749** |

B dominates A at every cut-off and scores 0.249 lower. This is the V1 artefact at full strength:
B never lets precision fall below 0.50, so a quarter of its area is removed. A system is
penalised for keeping precision high across the whole ranking.

**V3 — an attack that leaves the denominator raises the score.** 0.660 → 0.990 when the third of
three attacks contributes no nodes and is therefore not counted at all rather than counted as
missed. **Scope**: on the artifact a *failed uuid lookup* crashes instead
(`src/labelling.py:56`), so the only route in is an empty ground-truth file; on `main`, whose
`.get()` guard converts the crash into a silent skip, one unmappable uuid per attack suffices.
See §4 of [`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md).

**V4 — ADP scores attack coverage, not node recall.** Two rankings with bit-for-bit identical
label vectors and therefore identical precision curves, differing only in which attack each
malicious node belongs to: covering all three attacks by rank 5 gives 0.797, covering them only
by rank 9 gives 0.691. One node per attack saturates the numerator; the other correct
detections of an already-covered attack add nothing. **This part is by design** — the paper
argues for it explicitly (`resources/VELOX.txt:612-617`) and it is a defensible triage
objective. It is also why Table 4 rows can pair ADP ≈ 0.94 with TP = 8, and why ADP is not
comparable with any recall-based number.

**V5 — quantisation.** With `k = 3` on CADETS_E3, `D(p)` takes four values. Moving attack 2's
only node down the ranking while holding everything else fixed:

| rank of that node | 3 | 4 | 6 | 11 | 26 | 52 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADP | 0.971 | 0.904 | 0.823 | 0.728 | 0.675 | 0.654 |

The metric is dominated by the top of the ranking. Combined with the instability the paper
itself measures — relative ADP standard deviation up to 143% (`:1348-1353`) — a single node's
position accounts for much of the spread between a run's Min and Best.

## 3. The reported epoch is chosen on the test set

`[read]` This is the finding with the widest blast radius, because it applies to **every system
in the comparison**, not just VELOX.

`src/detection/evaluation.py:18-77` iterates over every saved epoch directory, calls the
evaluation function, and keeps the best — and on the artifact the rule is **hardcoded**, with no
config key to change it:

```python
for model_epoch_dir in sorted_files:
    stats = evaluation_fn(val_tw_path, test_tw_path, model_epoch_dir, cfg, ...)
    ...
    if stats["adp_score"] > best_adp:        # :69-71
        best_adp = stats["adp_score"]
        best_stats = stats
```

`stats["adp_score"]` is computed on the **test** split. In `node_evaluation.py`, `val_tw_path`
is consulted at `:21` for the threshold and nowhere else; every score in `results` comes from
`test_tw_path` (`:28-30`), and `adp_score` is computed from those scores at `:320` `[read]`.
With `num_epochs: 8` (`config/velox.yml:70`) ~~that is a maximum over eight candidates~~.
**CORRECTION (2026-11-24):** over five. Validation and test are scored only when
`(epoch+1) % 2 == 0 or epoch == 0` (`src/detection/training_methods/orthrus_gnn_training.py:130`),
so edge losses exist for epochs 0, 1, 3, 5 and 7, and those five directories are what
`evaluation.py:31` iterates.

**Framework drift, and it runs the wrong way:** `main` at least exposes the rule as
`best_model_selection`, defaulting to `best_adp` across all system configs
(`pidsmaker/tasks/evaluation.py:93-99`). On the artifact there is no knob at all.

So the checkpoint whose number is reported is the one that scored best on the test set. This is
the same defect the ORTHRUS pass found in the standalone repo (there, argmax over test MCC) —
but here it is framework-wide, and it affects the baselines VELOX is compared against as well as
VELOX itself. The chapter should present it as a property of the *comparison*, not as a charge
against one system.

The contrast with their hyperparameter tuning is what makes this worth reporting: every sweep
file sets `metric: {name: val_score, goal: maximize}`, so **tuning is selected on validation**
(see [`the-ablation-behind-the-claim.md`](the-ablation-behind-the-claim.md) §3). They were
careful at the outer loop and not at the inner one.

## 4. Three nested maxima

`[read]` Putting §1–§3 together, one reported "Best ADP" number is:

1. **max over thresholds** — ADP is the upper envelope over prefix cut-offs
   (`evaluation_utils.py:326`). Defensible as an achievability curve, but it is a maximum.
2. **max over epochs** — `argmax` of test ADP across all eight saved checkpoints
   (`src/detection/evaluation.py:69-71`, hardcoded).
3. **max over seeds** — the "Best" column of Tables 4–6 is the maximum across the five
   instability iterations (`resources/VELOX.txt:1397-1399`).

Every stage is resolved on test data. And the paper's own SC5 recommends the opposite: "we
recommend averaging metrics over multiple iterations, a practice we adopt throughout this paper"
(`:800-802`) `[paper]`. The headline claim does not follow that recommendation — see §5.

## 5. Table 4 (E3), recovered

`[paper]` `resources/VELOX.txt:1071-1373`. The plain-text conversion separates the numeric
columns from the row labels, so this reconstruction is by column order and is
`[unverified]` in its alignment for the THEIA block; the CADETS and CLEARSCOPE blocks are
unambiguous. Re-check against the PDF before any of it reaches the thesis.

**E3-CADETS** (ADP mean/min/best · σ̃ · precision mean/min/best · TP/FP at best precision):

| System | ADP mean | min | best | σ̃ | Prec mean | TP | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SIGL | 0.00 | 0.00 | 0.01 | 8% | 0.00 | 0 | 45 |
| ThreaTrace | 0.01 | 0.01 | 0.01 | 36% | 0.00 | 26 | 17k |
| NodLink | 0.33 | 0.15 | 0.96 | 56% | 0.00 | 50 | 55k |
| MAGIC | 0.03 | 0.02 | 0.06 | 37% | 0.00 | 68 | 144k |
| KAIROS | 0.01 | 0.01 | 0.01 | 24% | 0.00 | 0 | 8 |
| FLASH | 0.10 | 0.00 | 0.34 | 127% | 0.05 | 1 | 3 |
| R-CAID | 0.21 | 0.04 | 0.44 | 64% | 0.21 | 2 | 0 |
| **ORTHRUS** | **0.94** | 0.85 | 1.00 | 12% | 0.66 | 10 | 0 |
| **VELOX** | **0.94** | 0.77 | 1.00 | 9% | 0.85 | 8 | 0 |

**E3-CLEARSCOPE**: ORTHRUS 0.75 / 0.25 / 1.00 (σ̃ 42%), VELOX **0.30** / 0.02 / 1.00 (σ̃ 116%).
Post-threshold precision is 0.00 for both; VELOX's best-precision run is TP 1, FP 913.

**E3-THEIA** `[unverified]`: ORTHRUS 0.44 / 0.10 / 1.00 (σ̃ 73%), VELOX **0.72** / 0.42 / 0.97
(σ̃ 30%), TP 10, FP 1.

Three readings the chapter should take from this:

- **The headline is a max, and the mean tells a different story.** "Matching or exceeding the
  best ADP scores of significantly more complex systems on eight out of nine datasets"
  (`:1666-1672`) is a comparison of *Best* columns. On E3-CLEARSCOPE the means are ORTHRUS 0.75
  vs VELOX 0.30 — a wide gap — and only the max-over-five-seeds ties at 1.00, with VELOX's own
  σ̃ at 116%. Quote the means alongside.
- **VELOX genuinely wins on THEIA.** 0.72 vs 0.44 mean, with *lower* variance. This is the
  strongest single result in the paper and the chapter should not bury it — a graph- and
  time-free model beating a graph model on the most information-rich E3 dataset is exactly the
  observation that motivates this thesis's THEIA experiments.
- **Post-threshold performance is poor for everyone on CLEARSCOPE.** Precision 0.00 across the
  board. The gap between ADP and precision is the paper's own SC2 point, honestly reported.

## 6. Splits

`[read]` `src/config.py:403-406`. CADETS_E3 is specified as graph files, not dates:
`train_files` graphs 2–5, 7–9; `val_files` graph 10; `test_files` graphs **6**, 11, 12, 13;
`unused_files: []`.

Day 06 sits *inside* the training range, so **the CADETS_E3 split is not chronological** — the
same defect the ORTHRUS pass found, inherited here and therefore affecting every system's
CADETS_E3 number in Tables 4–6, VELOX's included.

`unused_dates` drops whole days per dataset with no stated criterion `[read]`: THEIA_E3 drops
one day (`:47`), CLEARSCOPE_E3 two (`:182`), CADETS_E5 four (`:78`), THEIA_E5 four (`:16`). The ORTHRUS pass measured
THEIA_E3's single unused day at 7,339,776 filtered events — 18.39% of the dataset.

`preprocessing.build_graphs.time_window_size` is **tuned** over {5.0, 15.0} minutes
(`config/experiments/tuning/systems/default/tuning_default_baselines.yml`) `[read]`, so "15
minutes" is a default, not a protocol constant.

## 7. CORRECTION — the real-time path IS in the artifact

`[paper]` SC9 is the shortcoming VELOX claims to uniquely resolve; Table 3 (`:1010-1034`) marks
it ✔ on all nine, and §5.3 (`:2042-2084`) reports a real-time deployment (peak RAM 5.7 MB, CPU
peaking at 107% of one core).

An earlier draft of this note, reading `main`, claimed no real-time or streaming inference path
existed. **That was wrong for the artifact.** `[read]` The artifact's README documents the
experiment and the mechanism it uses:

> ### Real-Time Scalability metrics (Fig. 16)
> To simulate real-time detection, we set the batch size to 1 edge instead of 15min, and we
> monitor metrics.
> ```shell
> python plot_real_time_cpu_watt_memory.py velox CADETS_E3 \
>   --detection.gnn_training.edge_batch_size_inference=1 \
>   --detection.gnn_training.batch_mode=edges --tuned
> ```

and both keys are real: `src/config.py:161-162` declares `edge_batch_size_inference` and
`batch_mode: ["edges" | "minutes"]` `[read]`. So VELOX genuinely can be run one edge at a time,
and SC9 is addressed by a mechanism in the code, not only by design argument. Credit it.

`[read]` What is missing is narrower: **the measurement harness**.
`plot_real_time_cpu_watt_memory.py` is not in the repository — the only `plot_*` script is
`src/plot_GP_figs.py`, and `scripts/` contains no equivalent. So the per-edge inference mode is
reproducible; the RAM/CPU/watt figure built on top of it is not.

Two further notes:

- The detection numbers in Tables 4–6 still come from the ordinary batched pipeline
  (15-minute graphs), not from the per-edge mode. The two setups are different runs; that much
  of the earlier draft stands.
- `[paper]` `:2076-2077` remains internally inconsistent: "approximately 41 ms per edge, or
  roughly 2,400 edges/s" disagree by a factor of ~100 (2,400/s ⇔ 0.41 ms). The claim that this
  beats E5-CADETS's 1,832 edges/s peak only holds on the 2,400 figure. Note also that the
  artifact README calls the real-time figure **Fig. 16** where our `resources/VELOX.txt`
  conversion has it as Fig. 15, so the branch tracks a later revision of the paper than the text
  we hold — check both against the published PDF before citing either.

## 8. For the thesis's own metric decisions

Recorded for [`notes/metric-decisions.md`](../../../../notes/metric-decisions.md), not decided here:

- ADP is worth adopting **as a triage metric alongside**, never instead of, a recall-bearing
  metric — V4 shows it deliberately ignores per-attack recall.
- If adopted, **re-implement it**; do not import this one. Anchor the integral over the full
  `[0, 1]` precision axis with an explicit convention for `p < p_min` rather than interpolating
  from a synthetic origin (V1/V2), and make an attack with no resolvable node count as
  undetected rather than vanishing (V3).
- Report the **mean and standard deviation over seeds**, following VELOX's SC5 rather than its
  headline.
- Select checkpoints on **validation**, and say so — §3 shows the whole comparison does not.
- One correction to `PLAN.md`'s note that ADP "doesn't punish false positives": it does — the
  x-axis is precision, so false positives lower the curve. What it nearly ignores is
  node-level *recall within an attack* (V4). Worth restating that way.

**Related:** [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md) ·
[`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md) ·
[`the-ablation-behind-the-claim.md`](the-ablation-behind-the-claim.md)
