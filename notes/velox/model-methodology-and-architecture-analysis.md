# VELOX — model, methodology and architecture

VELOX and PIDSMaker are two different papers sharing one repository. This note and its
four companions analyse the **VELOX paper** ("Sometimes Simpler is Better: A Comprehensive
Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems", Bilot et al.).
The framework paper itself is analysed in
[`../model-methodology-and-architecture-analysis.md`](../model-methodology-and-architecture-analysis.md)
and the shared preprocessing in [`../pidsmaker-data-import-analysis.md`](../pidsmaker-data-import-analysis.md).
VELOX has no import pipeline of its own.

## Which code this analyses

`[read]` The paper's artifact is the **`velox` branch**, not `main`
(`resources/VELOX.txt:2164-2167`). These are different codebases, not different revisions
of one: the artifact has a `src/` package and a `preprocessing / featurization / detection`
config schema; `main` has a `pidsmaker/` package and a flat `construction / batching /
training / evaluation` schema. **Everything below cites the artifact.**

| | ref | contents |
| --- | --- | --- |
| **Artifact (cited here)** | `upstream/velox` @ `54f687c54aa03e5519cf44953d5ee44f5f6a4a28` | `src/`, `config/velox.yml` as a full 125-line config |
| Publication snapshot | tag `usenixsec` @ `ef9a9e2` | ancestor of the above; the only diff in the files cited here is `num_workers: 15 → 1` in `config/velox.yml`, so the two are interchangeable for every claim in these notes |
| Current framework | `origin/main` @ `c8c656d` | `pidsmaker/`, `config/velox.yml` as a 10-line include; **not what the paper ran** |

Fetched with:

```sh
git -C related-work/PIDSMaker fetch \
  https://github.com/ubc-provenance/PIDSMaker.git velox:refs/remotes/upstream/velox
```

**Three claims in an earlier draft of these notes were artefacts of reading `main`, and are
corrected below**: the "dead dropout" (§2a), the silently-dropped ground-truth uuid
(companion note §4), and the size of the baseline tuning effort
([`the-ablation-behind-the-claim.md`](the-ablation-behind-the-claim.md) §3). Where `main`
differs in a way that matters to *this thesis reusing the framework*, it is flagged as
**framework drift**.

Analysed 2026-09-16. Line references are to the artifact unless stated.
Paper text: `resources/VELOX.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code during this analysis.
`[paper]` = claim from the paper. `[measured]` = produced by a probe in
`code/scratchpad/velox_adp_probe.py`. `[unverified]` = inference or argument, not confirmed by
execution; never cite.

---

## 0. Answer in one paragraph

VELOX is ORTHRUS with the encoder replaced by a bare `nn.Linear` and the neighbour sampler
emptied. `[read]` What remains is: look up a word2vec embedding for each of an edge's two
endpoints, project both through one shared linear layer, and classify the edge's type from the
pair. No memory, no message passing, no neighbourhood, and no representation of time — not even
the last-N ordering that is ORTHRUS's only temporal signal. That makes VELOX the **null floor**
of the chapter's ordering by temporal and structural modelling, and the number every
architectural claim in this thesis has to clear or explain. The chapter should say so plainly,
and then say what [`the-ablation-behind-the-claim.md`](the-ablation-behind-the-claim.md)
establishes: the grid search that licenses "a simple linear layer suffices" never contained a
memory-bearing encoder, and does not ship.

---

## 1. What VELOX is, as configuration

`[read]` On the artifact, `config/velox.yml` is a standalone 125-line config — there is no
include mechanism. Against `config/orthrus_non_snooped.yml` (the paper's de-snooped ORTHRUS)
it differs in exactly two substantive keys:

| Key | ORTHRUS non-snooped | VELOX | Effect |
| --- | --- | --- | --- |
| `detection.gnn_training.encoder.used_methods` | `tgn,graph_attention` | `none` (`velox.yml:83`) | `LinearEncoder` instead of a TGN-wrapped GAT |
| `detection.gnn_training.encoder.neighbor_sampling` | `[tgn_last_neighbor]` | `[]` (`velox.yml:80`) | no neighbour sampler is built at all |

The empty `neighbor_sampling` list is the easy-to-miss half and it matters: VELOX has no
neighbourhood structure available even in principle.

Everything else VELOX inherits verbatim from the de-snooped ORTHRUS, and the inheritance is by
copy, not by include: `time_window_size: 15.0` (`:7`), word2vec `training_split: train` (`:20`),
`num_epochs: 8` (`:70`), `lr: 0.0001` (`:72`), `node_out_dim: 128` (`:75`), `dropout: 0.3`
(`:82`), `use_memory: False` (`:96`), `use_time_order_encoding: False` (`:97`),
`threshold_method: max_val_loss` (`:123`), `use_kmeans: False` (`:125`).

## 2. The encoder is one shared linear layer

`[read]` `src/encoders.py:550-558` is the whole encoder:

```python
class LinearEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple):
            return self.lin1(x[0]), self.lin1(x[1])
        return self.lin1(x)
```

Selected at `src/factory.py:156-157` (`elif method == "none": encoder = LinearEncoder(in_dim,
node_out_dim)`).

**(a) CORRECTION — there is no dropout here, and that is not a bug.** An earlier draft of this
note reported that `LinearEncoder`'s dropout was constructed at `p=0.0` and therefore dead.
That is true of `main` only. On the artifact the class takes no `dropout` argument at all: the
encoder is *literally and exactly* `nn.Linear`, which is what the paper says it is
(`resources/VELOX.txt:886-887`). The tuned `encoder.dropout: 0.3` regularises the decoder, as
intended. **Framework drift:** `main`'s `LinearEncoder`
(`pidsmaker/encoders/linear_encoder.py:4-16`) gained a `dropout=0.0` parameter that
`pidsmaker/factory.py:244-245` never passes — so the knob is unreachable *there*. Worth knowing
if this thesis reuses the framework; it is **not** a criticism of the paper.

**(b) One shared projection, not two.** The tuple branch is taken only when the batch supplies
`(x_src, x_dst)`; VELOX's node features arrive as a single tensor, so one `lin1` projects every
node. The paper's Fig. 10 describes "source and destination entity embeddings ... processed
through separate linear layers to capture orientation-specific semantics"
(`resources/VELOX.txt:928-931`) `[paper]`. That separation is real but lives in the **decoder**,
not the encoder. The encoder treats a node identically whichever end of an edge it sits on.

**(c) There is no message passing.** `src/model.py:69` indexes the encoder output by the edge
endpoints:

```python
h_src, h_dst = (h[edge_index[0]], h[edge_index[1]]) ...
```

`[read]` `edge_index` is used only to *select* rows. No neighbour is ever aggregated. A node's
representation is a function of its own features alone, so two nodes with identical
word2vec-plus-type features are indistinguishable to VELOX however differently they behave.

## 3. What reaches the model

`[read]` Node features are `node_emb,node_type` (`velox.yml:79`): a 128-dimensional
`feature_word2vec` vector (`:18`, `:21`) concatenated with a 3-way node-type one-hot
(`num_node_types: 3`, `src/config.py:399`). Edge features are `edge_type` only (`:81`).

**No temporal signal of any kind.** `use_time_order_encoding: False` (`:97`), no time encoder,
no Δt. Timestamps reach the batcher, which orders edges, and stop there. This places VELOX at
the bottom of the chapter's "how time enters each system" ordering:

| System | Temporal mechanism |
| --- | --- |
| KAIROS | Fourier Δt encoder feeding edge attention, plus per-node memory |
| ORTHRUS | ordering implicit in a last-N neighbour buffer; no time representation |
| GRASP | none in-window; time is only the window/step size |
| **VELOX** | **none, and no neighbourhood either** |

## 4. Memory is one boolean, and VELOX is not the experiment that tests it

`[read]` On the artifact, node memory is a single flag, set per system:

- `config/kairos.yml:53` → `use_memory: True`
- `config/orthrus.yml:88`, `orthrus_non_snooped.yml:97`, `nodlink.yml:62`, `velox.yml:96` →
  `False`

So of the systems in the study, **only KAIROS carries a memory** — which is exactly the encoder
the Fig. 9 grid excluded. See the companion note.

Because everything else is held fixed, toggling that flag would produce the controlled memory
ablation no paper has published — the experiment ORTHRUS's Table 6 is read as having run but
did not (`related-work/orthrus/notes/the-published-ablation.md`). Two caveats, both important.
**Only the toggle is a valid ablation** — the `kairos` and `orthrus` configs differ in far more
than `use_memory`, so the cross-config comparison is not one. And it would measure memory
*inside this reimplementation*, on its lossy preprocessing, its ground truth and its discretised
windows — a floor result, not a substitute for this thesis's continuous-time model. Recorded
here; scheduled in the Evaluation chapter, not this one.

## 5. Positioning on the thesis axes

| Axis | VELOX |
| --- | --- |
| Memory | none |
| Time | none — no encoding, no Δt, not even neighbour ordering |
| Graph structure | none — `edge_index` selects rows, never aggregates |
| Node keying | UUID |
| Featurisation | `feature_word2vec`(128) + node-type one-hot, trained on the train split only |
| Proxy task | 10-way edge-type cross-entropy; per-edge loss *is* the anomaly score |
| Granularity | node (scored), edge (modelled) |
| Window | 15 min, tuned over {5, 15} |
| Thresholding | max validation loss; no k-means cap |
| Split | day-based; CADETS_E3 is not chronological |
| Model selection | best **test** ADP over 8 epochs |

## 6. Verdict

Adopt nothing; clear the bar. VELOX's value to this thesis is as a floor and as a warning: if a
per-node linear map over a text embedding reaches the same ADP as a TGN on eight of nine
benchmarks, then any claim that memory or structure contributes must be demonstrated against
*this* baseline, not against KAIROS. The counter-consideration — and the reason the floor is
less damning than it reads — is that the comparison selects epochs on test ADP, quotes a max
over five seeds, and uses a metric with the properties measured in
[`metrics-splits-and-reported-results.md`](metrics-splits-and-reported-results.md). Take the
floor seriously; do not take it at face value.

**Related:** [`the-ablation-behind-the-claim.md`](the-ablation-behind-the-claim.md) ·
[`objective-training-and-featurisation.md`](objective-training-and-featurisation.md) ·
[`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md) ·
[`metrics-splits-and-reported-results.md`](metrics-splits-and-reported-results.md)
