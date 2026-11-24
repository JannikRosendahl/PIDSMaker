# VELOX — where time enters, and where it cannot

Companion to the four notes beside this one. It asks the question the KAIROS and ORTHRUS
passes asked of their systems: **how does time reach the model, if at all?** It is answered
from the artifact's code, and each claim is then checked with a replay over the data
VELOX runs on (the lab note `notes/velox-time-handling.md` in the thesis repository holds the
commands and every number).

Artifact: the `velox` branch @ `54f687c54aa03e5519cf44953d5ee44f5f6a4a28` (see
[`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md),
*Which code this analyses*). Paths are relative to the artifact; `src/` is implied for `.py`
files. Analysed 2026-11-24.

## 0. Answer in one paragraph

At inference VELOX is **time-invariant by construction**: an edge's score is a fixed function
of (source type and label, destination type and label, edge type), a node's score is the
maximum over the edges it touches anywhere in the test split, and the threshold is the maximum
over the validation split. No timestamp, order, window, batch or fusion decision can change a
node's score, and the replay confirms it to the last bit on both datasets. The paper says as
much (`resources/VELOX.txt:2044-2046`). Time still acts in three places, none of which
the paper discusses: **(i)** in training, each 15-minute window is one batch and one optimizer
step, so the window cut sets how much each edge weighs in the loss; **(ii)** fusion runs per
window, so the window size sets how often a repeated interaction counts, which reshapes the
class distribution of the edge-type objective; **(iii)** in ADP, nodes with exactly equal
scores are ranked by their position in the score list, which is their first appearance in the
time-ordered test stream. Most test nodes tie with another node, including 20 of the 68
attack nodes on CADETS E3.

## 1. Graph construction: ORTHRUS's, with three changes

`[read]` `preprocessing/build_graph_methods/build_orthrus_graphs.py` is ORTHRUS's
`graph_construction/build_orthrus_graphs.py` (fork `related-work/orthrus` @ `e7cab78`) with
these differences (a `diff` of the two files):

1. **The day's trailing window is written.** `last_batch` (`:222-230`) flushes the last partial
   window of each day; ORTHRUS's repo only closes a window when a batch crosses the 15-minute
   mark, so its tail is never written. Measured: 12 tail windows / 493,714 events on CADETS
   (7,017 to 159,267 per day; day 11 alone 159,267), 8 / 206,794 on THEIA, all of which VELOX
   keeps.
2. **Nodes are keyed by `index_id`** (`:47`, `:68`, `:88`, `:240`, `:247`) instead of `hash_id`:
   still one node per uuid.
3. **The label string carries the node type, and netflows carry the remote port**
   (`config/velox.yml:9-12`, read positionally at `:39-51`): `subject <path> <cmd_line>`,
   `file <path>`, `netflow <remote_ip> <remote_port>`; a missing field is the string `None`.
   On CADETS E3 the port turns 53,390 training netflows into 12,309 distinct labels.
4. Optional mimicry events (`:153-157`, `:211-215`), off: `mimicry_edge_num` is unset.

Unchanged: the day query `timestamp_rec > lo AND timestamp_rec < hi ORDER BY timestamp_rec,
event_uuid` (`:192-199`), the 10-type filter (`:204-209`, `config.py:1202-1223`), the
1024-event batching of the window cut (`:217-230`), fusion (`:252-286`) and the `networkx`
graph (`:289-305`). The strict bounds drop no admitted event: no event of the 10 types is
stamped exactly at an ET midnight on either dump.

## 2. Inference: nothing temporal survives

`[read]`, in pipeline order:

- `featurization/embed_edges.py:20-26` carries each edge's `time` into `TemporalData.t`.
- `data_utils.py:142-176` builds `x_src`, `x_dst` (`node_emb,node_type`, `config/velox.yml:79`)
  and removes the edge type from the message for edge-type prediction.
- `factory.py:399-436`: with `neighbor_sampling: []` and no TGN, the loader returns the window
  as one batch (`:433-436`). It reindexes it, but the per-node `x` that produces is not read on
  this path (next bullet).
- `model.py:55-113`: `predict_edge_type` is not node-level (`config.py:54`, `:962-964`), so
  `_reshape_x` passes `(x_src, x_dst)` (`:115-120`); `LinearEncoder` (`encoders.py:550-558`)
  returns `(lin1(x_src), lin1(x_dst))`; `EdgeMLPDecoder` (`decoders.py:9-21`) and
  `EdgeTypePrediction` (`decoders.py:136-152`) turn that into a per-edge cross-entropy
  (`losses.py:39-41`, `reduction="none"` at inference). `t` is passed to the encoder and
  ignored (`*args, **kwargs`).
- `detection/evaluation_methods/node_evaluation.py:28-47`: a node's score is the max of the
  losses of every test edge it is source or destination of (`use_dst_node_loss: True`,
  `config/velox.yml:124`; `evaluation_utils.py:39-48`).
- `evaluation_utils.py:23-26`, `:50-69`: the threshold is the max validation edge loss;
  `use_kmeans: False` (`config/velox.yml:125`), so `y_hat = score > thr`
  (`node_evaluation.py:66-71`).

**So** `score(v) = max { s(type_u, label_u, type_w, label_w, etype) : (u, w, etype) ∈ test edges, v ∈ {u, w} }`.
Each of the following leaves every node's score unchanged:

| change | why |
| --- | --- |
| reorder the test edges | `max` over a set |
| re-batch (any size, incl. the README's per-edge mode, `README.md:179-185`) | the forward pass is row-wise |
| re-window (any `time_window_size`) | windows only partition the set |
| fuse or not fuse | fusion drops repeats of the same `(u, w, etype)`, which score the same |
| move an event in time within the test split | nothing reads `t` |

`[measured]` The artifact's own `LinearEncoder`, `EdgeMLPDecoder`, `EdgeTypePrediction`,
`build_mlp_from_string` and `cross_entropy`, exec'd from its source with random weights at the
tuned sizes (`emb_dim 64`, `node_out_dim 256`), over the real fused edges of one attack day
(CADETS day 6, 576,264 edges; THEIA day 12, 2,526,221): node scores are **identical**
(max |Δ| = 0.0) under a shuffled order in 100k batches, under 5-minute windows re-fused from
the raw events, and with no fusion at all (1,919,383 / 7,055,614 raw events); per-edge batches
differ by at most 4.8e-7 in a 20,000-edge sample (float summation order).

Consequences:

- **The real-time claim is about the same detector as the reported numbers.** Tables 4–6 come
  from 15-minute batches and the real-time figure from 1-edge batches (`README.md:179-185`,
  `data_utils.py:81-82`), but for VELOX the two produce the same scores. That is not true of
  KAIROS or ORTHRUS, whose state depends on the batch.
- **The node decision can be streamed**: `max > thr` holds iff some incident edge exceeds `thr`,
  which is known when that edge arrives. ADP, which ranks the whole test split, cannot.
- **Detection is a function of label co-occurrence.** VELOX can only flag a node through a
  (label, label, type) triple whose learned score is high. An attack that recombines known
  triples in a new order, rhythm or timing is invisible by construction.

## 3. Training: the window is the batch

`[read]` `config/velox.yml` sets no `edge_batch_size` and no `batch_mode`, so both are `None`
(`config.py:652-700` creates every key as `None`). `data_utils.py:73-84` then leaves each window
graph as its own `TemporalData`, `factory.py:433-436` hands it over as one batch, and
`detection/training_methods/orthrus_gnn_training.py:30-46` takes **one optimizer step per
window**, on the **mean** cross-entropy of its edges (`losses.py:39-41`). Windows are visited in
file-name order, which is chronological within a split (`data_utils.py:92`), 8 epochs
(`config/velox.yml:70`), no shuffling.

An edge's weight in an epoch's objective is therefore `1 / |its window|`. `[measured]`

| | CADETS E3 | THEIA E3 |
| --- | --- | --- |
| training windows = steps per epoch | 582 | 197 |
| edges per window, min / median / max | 574 / 4,317 / 54,930 | 806 / 2,333 / 63,301 |
| per-edge weight, largest / smallest window | 96× | 79× |
| largest decile of windows: share of edges / of steps | 33.3% / 10% | 26.7% / 10% |
| smallest decile: share of edges / of steps | 4.9% / 10% | 0.9% / 10% |

This is a time-defined loss weighting, not a temporal model: a burst puts many edges into
one window and thereby gets fewer steps per edge than a quiet period.

**Fusion is the second route**, because it runs per window (`build_orthrus_graphs.py:252-286`).
How often a repeated `(u, w, op)` counts is set by the window size, and that reshapes the class
distribution of the training objective. `[measured]` On the training split:

| type | CADETS before → after | THEIA before → after |
| --- | --- | --- |
| `EVENT_READ` | 66.4% → 32.3% | 23.2% → 36.7% |
| `EVENT_OPEN` | 21.1% → 54.8% | 14.2% → 43.5% |
| `EVENT_WRITE` | 8.5% → 4.1% | 9.2% → 3.7% |
| `EVENT_RECVFROM` | 1.1% → 1.2% | 27.4% → 3.9% |
| `EVENT_RECVMSG` / `SENDMSG` | 0.02% / 0.01% → 0.04% / 0.02% | 12.2% / 9.6% → 0.4% / 0.4% |

With `balanced_loss: False` (`config/velox.yml:103`), these frequencies are the priors the
classifier fits, and the per-edge cross-entropy is the anomaly score. So the window size
influences which edge types score high.

There is **no state across windows, days or splits**: no memory (`use_memory: False`, `:96`), no
sampler, no carried-over buffer. The ORTHRUS leaks (validation and training neighbourhoods
reaching test, backwards in time on CADETS) cannot occur.

## 4. Evaluation: ties are broken by stream position

`[read]` ADP sorts nodes with `np.argsort(scores)[::-1]`
(`detection/evaluation_methods/evaluation_utils.py:290-292`); `nodes` is in the insertion order
of `node_to_losses` (`node_evaluation.py:24-47`, `:279-288`), i.e. the order the test CSVs,
read chronologically, first mention each node. For exactly tied scores the rank therefore comes
from that order, not from the model.

`[measured]` Nodes incident to the same *set* of triples tie under **any** weights.

| | CADETS E3 | THEIA E3 |
| --- | --- | --- |
| test nodes | 281,585 | 700,902 |
| distinct triple sets | 16,295 | 365,889 |
| nodes sharing their set with another node | 266,339 (94.6%) | 342,667 (48.9%) |
| largest group | 71,692 | 18,605 |
| attack nodes tied with ≥ 1 benign node | 20 of 68 (largest tie: 5,122 benign) | 6 of 118 (largest: 11) |

A deterministic probe of the same arithmetic (`code/scratchpad/velox_adp_probe.py`, V7)
gives 0.517 against 0.833 for the same scores in two input orders. How much this moves a
reported ADP depends on where the trained weights put those tied groups, which only a trained
run can show.

## 5. Score inputs: how many different things VELOX can say

`[measured]` The number of distinct (src label, dst label, edge type) triples is the number of
distinct scores VELOX can produce:

| | CADETS E3 | THEIA E3 |
| --- | --- | --- |
| test edges → distinct triples | 1,989,005 → 24,361 | 5,002,043 → 675,087 |
| test triples absent from training | 16,944 (69.6%) | 652,205 (96.6%) |
| test edges carrying such a triple | 30,608 (1.5%) | 2,596,902 (51.9%) |
| attack nodes with such a triple | 50 of 68 | 117 of 118 |
| training subjects → distinct subject labels | 119,625 → 112 | 93,546 → 853 |

On CADETS E3 a subject's representation is one of 112 strings; uuid keying (one node per
process) does not change what the model can tell apart.

## 6. Positioning (time axis only)

| System | How time enters |
| --- | --- |
| KAIROS | per-node GRU memory, Fourier Δt encoder, temporal neighbour sampling; reset per day |
| ORTHRUS | last-N neighbour buffer (insertion order, not time); no time representation |
| **VELOX** | **none at inference (provably and measured); in training only as the batch (window = step) and the fusion scope** |
