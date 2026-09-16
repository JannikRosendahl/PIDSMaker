# VELOX — detection, thresholding and ground truth

Companion to [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md).
Answers **how a per-edge loss becomes a node alarm**, **where the threshold comes from**, and
**what the ground truth is** — the three places where VELOX's protocol differs from ORTHRUS's,
two of them for the better.

Analysed against the artifact, `upstream/velox` @ `54f687c` (2026-09-16) — see the companion
note for why `main` is the wrong target. Paper text: `resources/VELOX.txt`.
Provenance markers as in the companion note.

---

## 0. Answer in one paragraph

VELOX scores edges, aggregates the losses onto both endpoints, and thresholds nodes at the
maximum loss seen on the validation split `[read]`. That is a genuine fix for SC3: unlike
ORTHRUS, which ships `use_kmeans: True` and thereby discards its threshold and caps detections
at the top-K scoring nodes, VELOX's `orthrus_non_snooped` parent sets `use_kmeans: False`, so
the validation threshold actually decides `[read]`. The ground truth is ORTHRUS's node-uuid
CSVs, and the attack set is smaller than the DARPA campaign: CADETS_E3 enables **three of five**
ground-truth files, the other two commented out. That three is also the denominator of the ADP
metric, with the consequences measured in
[`metrics-splits-and-reported-results.md`](metrics-splits-and-reported-results.md). Two claims
from an earlier draft are corrected here: on the artifact an unresolvable ground-truth uuid
**crashes** rather than being dropped silently (§4), and the artifact carries an inverted
`percent_detected_attacks` statistic fixed later on `main` (§4a).

---

## 1. From edge loss to node score

`[read]` `src/detection/evaluation_methods/node_evaluation.py:13-40`. Every CSV in the
test loss directory is read, and for each scored edge the loss is appended to **both** endpoints:

```python
node_to_losses[srcnode].append(loss)
if cfg.evaluation.node_evaluation.use_dst_node_loss:
    node_to_losses[dstnode].append(loss)
```

`use_dst_node_loss: True` (`config/velox.yml:124`) `[read]`, so a file or netflow node
inherits the surprise of every event touching it. The per-node score is then a reduction over
that list (`reduce_losses_to_score`) selected by the threshold method.

Two consequences worth carrying into the chapter:

- **A node's score is degree-coupled.** A busy process accumulates many losses; the reduction
  chosen by `max_val_loss` takes the maximum, so a node participating in more events has more
  chances to exceed the threshold. This is the same implicit degree weighting this thesis has
  flagged for its own model, arriving here through the evaluation rather than the training loop.
- **Detection granularity is node, modelling granularity is edge.** VELOX never scores a node
  directly; nodes exist only as aggregation targets. Attribution therefore inherits whatever
  the edge scores do.

## 2. Thresholding: SC3 is genuinely fixed

`[read]` `node_evaluation.py:17-21`:

```python
threshold_method = cfg.evaluation.node_evaluation.threshold_method
if threshold_method == "magic":      # data leaking by using test data
    thr = get_threshold(test_tw_path, threshold_method)
else:
    thr = get_threshold(val_tw_path, threshold_method)
```

VELOX uses `threshold_method: max_val_loss` (`config/velox.yml:123`), so `thr` is the maximum
node score on the **validation** split `[read]`. The framework's own comment marks the MAGIC
path as leaking, which is a fair and unusually candid piece of self-documentation.

The decisive difference from ORTHRUS is the `use_kmeans` branch:

```python
use_kmeans = cfg.evaluation.node_evaluation.use_kmeans
...
if use_kmeans:                       # in this mode, we add the label after
    results[node_id]["y_hat"] = 0
else:
    results[node_id]["y_hat"] = int(pred_score > thr)
if use_kmeans:
    results = compute_kmeans_labels(results, topk_K=...kmeans_top_K)
```

`config/orthrus.yml` ships `use_kmeans: True` with a top-K cap, while `config/velox.yml:125`
(copied from `orthrus_non_snooped.yml`) sets it `False` `[read]`. So:

| | ORTHRUS as shipped | VELOX |
| --- | --- | --- |
| `use_kmeans` | `True` | `False` |
| Threshold consulted? | no, `thr` is discarded | yes |
| Max nodes flaggable | `kmeans_top_K` | unbounded |

This matters for reading Table 4. ORTHRUS's tiny FP counts in its own README come partly from
the top-K cap; in *this* study both systems run without it, so the CADETS_E3 row (ORTHRUS TP 10
FP 0, VELOX TP 8 FP 0) is a like-for-like comparison. Credit the study for that — it is the
fairest head-to-head either system has had.

## 3. Ground truth: three attacks, not five

`[read]` `ground_truth_version: orthrus` (`config/velox.yml:120`), and the attack set is listed
per dataset in `src/config.py`. For CADETS_E3 (`:407-418`):

```python
"ground_truth_relative_path": [
    # "E3-CADETS/node_E_mail_Server.csv",
    "E3-CADETS/node_Nginx_Backdoor_06.csv",
    # "E3-CADETS/node_Nginx_Backdoor_11.csv",
    "E3-CADETS/node_Nginx_Backdoor_12.csv",
    "E3-CADETS/node_Nginx_Backdoor_13.csv",
],
"attack_to_time_window": [ ... same three, same two commented out ... ],
```

Two of five ground-truth files are commented out with no stated reason `[read]`. The ORTHRUS
pass found the same exclusions in the standalone repo, and established that two of the excluded
attacks have no CSV in the pinned ground-truth submodule at all — they could not have been
scored even if uncommented. PIDSMaker inherits both the exclusions and that explanation.

**So `k = 3` on CADETS_E3.** The paper's own SC2 walkthrough confirms it from the other side:
"when detecting 66.6% of attacks (2 out of 3 on E3-CADETS)" (`resources/VELOX.txt:632-634`)
`[paper]`. Everything the ADP metric reports on this dataset is quantised into thirds.

## 4. CORRECTION — on the artifact an unresolvable uuid CRASHES

An earlier draft of this note, reading `main`, reported that an unresolvable ground-truth uuid
is silently skipped and that an attack losing all its uuids therefore vanishes from the ADP
denominator. **That describes `main`, not the artifact**, and the direction is the opposite of
what the draft implied.

`[read]` On the artifact, `src/labelling.py:56` indexes the map directly:

```python
for row in reader:
    node_uuid, node_labels, _ = row[0], row[1], row[2]
    node_id = uuid2nids[node_uuid]          # unguarded
    attack_to_nids[i]["nids"].add(int(node_id))
```

An unresolvable uuid raises `KeyError` and kills the run — exactly like ORTHRUS's standalone
implementation (`related-work/orthrus/notes/detection-thresholding-and-ground-truth.md`). It is
a loud failure, not a silent one. **Framework drift:** `main` later replaced this with
`uuid2nids.get(node_uuid)` and an `is not None` guard
(`pidsmaker/utils/labelling.py:133-135`) — which converts the crash into precisely the silent
denominator drop the probe demonstrates. The "fix" introduced the quiet failure mode.

`[read]` The denominator hole itself is latent in both versions. `node2attacks` is built by a
local helper over a list of per-attack node sets (`node_evaluation.py:305-315`), so an attack
whose set came out empty contributes no entries; `total_attacks`
(`evaluation_utils.py:296`) counts distinct attacks *in that mapping*. An attack with no nodes
is therefore not counted as missed — it is not counted at all.

`[measured]` Probe V3 (`code/scratchpad/velox_adp_probe.py`) quantifies the consequence: losing
the third of three attacks this way raises ADP from 0.660 to 0.990. On the artifact the only
route into that state is an empty or unreadable ground-truth file, since a failed lookup
crashes first. On `main` a single unmappable uuid per attack is enough.

Neither is observed on the shipped datasets — that would need a database run. Record it as a
property of the plumbing and, more usefully, as a guard this thesis's own evaluation code should
have: an attack with zero resolvable nodes must count as undetected, not disappear.

## 4a. A reporting bug on the artifact

`[read]` `src/detection/evaluation_methods/node_evaluation.py:336`:

```python
stats["percent_detected_attacks"] = round(len(attack_to_GPs) / len(attack_to_TPs), 2) if len(attack_to_TPs) > 0 else 0
```

The ratio is inverted — it divides total attacks by detected attacks. With three ground-truth
attacks and one detected it reports `3.0`; it reads `1.0` only when every attack is detected,
and `0` when none is. `main` has it the right way round
(`pidsmaker/detection/evaluation_methods/node_evaluation.py:402-404`,
`len(attack_to_TPs) / len(attack_to_GPs)`), so this is a genuine artifact-era bug fixed later.

`[unverified]` This statistic is logged to wandb rather than being the ADP y-axis (which
`plot_detected_attacks_vs_precision` computes independently, `evaluation_utils.py:318`), so it
most likely does not feed Tables 4–6. Do not claim it corrupts the published numbers. Do flag it
if any figure in the paper is labelled "percentage of detected attacks" outside the ADP curve.

## 5. Positioning

| Axis | ORTHRUS (standalone) | VELOX |
| --- | --- | --- |
| Threshold source | val max loss, then discarded | val max loss, used |
| Detection cap | top-20/30 k-means | none |
| Unresolvable GT uuid | crashes the run | **crashes the run** (`main` later made it silent) |
| Attack with no resolvable node | n/a (crash) | latent: leaves the ADP denominator |
| GT nodes never scored | absent from the confusion matrix | absent from the confusion matrix |
| Attacks scored (CADETS_E3) | 3 of 5 | 3 of 5 |

The last row of agreement is the one the chapter should carry into the synthesis: every system
in this comparison is scored against three CADETS attacks, so "detects all attacks" is a claim
about three events.

**Related:** [`metrics-splits-and-reported-results.md`](metrics-splits-and-reported-results.md) ·
[`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md)
