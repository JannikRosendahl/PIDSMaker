# PIDSMaker / VELOX — model, methodology & architecture analysis

Companion to [`pidsmaker-data-import-analysis.md`](pidsmaker-data-import-analysis.md),
which covers the shared ingestion pipeline. This note covers the **model layer**: the
encoder/decoder zoo, how each reimplemented system is configured, and what the
`use_memory` flag makes possible.

Analysed at commit `9bc46d7d74cf6b5f8ee8f81497d9bdd09fb1c9f4` (2026-09-08).

**Two papers, one artifact.** This repository is the artifact behind *both*:
- **VELOX** — "Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art
  Provenance-Based Intrusion Detection Systems" (Bilot et al.), artifact on branch
  `velox` (`resources/VELOX.txt:2164-2167`);
- **PIDSMaker** — the framework paper (Bilot, Jiang & Pasquier, KDD'26).

Keep the citations distinct. The thesis draft previously cited PIDSMaker where it meant
VELOX; both `\cite{VELOX}` and `\cite{PIDSMaker}` now exist in `assets/thesis.bib`.

**Provenance convention.** `[read]` = verified by reading the cited code during this
analysis. `[import-note]` = from the companion import analysis. `[unverified]` = not yet
checked.

---

## 1. Shape of the framework

~220k LOC, Python. An 8-stage pipeline driven from `pidsmaker/main.py`
(`get_task_to_module`, `main.py:55-97`): construction → transformation → featurization →
feat_inference → batching → training → evaluation → triage.

Entry point:

```shell
python pidsmaker/main.py <SYSTEM> <DATASET> [--dotted.override=value ...]
```

Config is `config/default.yml` plus one YAML per system (`config/velox.yml`,
`kairos.yml`, `orthrus.yml`, …), merged and validated in `pidsmaker/config/`. Dotted CLI
overrides are documented at `README.md:141-156` `[read]`, e.g.:

```shell
python pidsmaker/main.py kairos CADETS_E3 \
  --training.node_hid_dim=128 \
  --batching.intra_graph_batching.tgn_last_neighbor.tgn_neighbor_size=10
```

Reimplemented systems: Velox, Orthrus, R-Caid, Flash, Kairos, Magic, NodLink,
ThreaTrace, OCR-APT.

---

## 2. Encoder zoo, and what VELOX actually is

`[read]` `pidsmaker/encoders/linear_encoder.py` — the entire VELOX encoder:

```python
class LinearEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, *args, **kwargs):
        ...
        return {"h": h}
```

Selected by `pidsmaker/factory.py:244-245` when `used_methods: none`. And
`config/velox.yml` in full `[read]`:

```yaml
# Simply the same config as orthrus_non_snooped + encoder set to none
_include_yml: orthrus_non_snooped
batching:
  intra_graph_batching:
    used_methods: edges
training:
  encoder:
    used_methods: none # uses only a linear layer
```

So VELOX = ORTHRUS's pipeline with the graph encoder deleted: **no graph attention, no
memory, no temporal neighbour sampler.** One `nn.Linear` over node features. That is the
null floor, and it is worth quoting verbatim in the thesis — it is more striking as code
than as prose.

---

## 3. Memory and time are real here

Unlike the standalone ORTHRUS repo (which has no memory class at all), PIDSMaker
implements both.

`[read]` `pidsmaker/tgn.py:55-62` — a genuine GRU-based memory:

```python
self.time_enc = TimeEncoder(time_dim)
self.gru = GRUCell(message_module.out_channels, memory_dim)
self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
```

`[read]` `pidsmaker/tgn.py:317-326` — a Time2Vec-style time encoder:

```python
class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        self.lin = Linear(1, out_channels)
    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()
```

A lighter `TimeEncodingMemory` variant also exists (`tgn.py:195+`, wired at
`factory.py:277-283`) for when time encoding is wanted without full memory. Temporal
neighbour sampling is `LastNeighborLoader` (`tgn.py:330+`).

---

## 4. ⭐ `use_memory` is a single flag — the isolated ablation nobody published

`[read]` the flag appears in exactly three places:

```
config/kairos.yml:67    use_memory: True
config/orthrus.yml:80   use_memory: False
config/default.yml:70   use_memory: False
```

**Read the config carefully — it is easy to get wrong.** Both `kairos.yml:56` and
`orthrus.yml:69` set `used_methods: tgn,graph_attention`. That selects the **TGN encoder
wrapper**, *not* the memory; the memory itself is governed by the nested
`tgn.use_memory` flag, and ORTHRUS sets it `False`. A first pass over this repo can
easily conclude PIDSMaker's ORTHRUS is memory-bearing. It is not, and that is consistent
with the standalone ORTHRUS repo.

Because everything else is held fixed by the shared pipeline, four runs would yield the
memory ablation that no paper has published — the experiment ORTHRUS's Table 7 claims to
have performed but did not (it swapped whole encoders instead; see the ORTHRUS note §3):

```shell
python pidsmaker/main.py orthrus CADETS_E3 --training.encoder.tgn.use_memory=True
python pidsmaker/main.py orthrus CADETS_E3 --training.encoder.tgn.use_memory=False
python pidsmaker/main.py kairos  CADETS_E3 --training.encoder.tgn.use_memory=False
python pidsmaker/main.py kairos  CADETS_E3 --training.encoder.tgn.use_memory=True
```

**Caveats, which must be stated if this is ever run.**

1. It measures memory *within PIDSMaker's reimplementation*, on its lossy preprocessing
   and its ground truth, with windows still discretised at 15 min. A **floor result**,
   not a substitute for a continuous-time model.
2. Only the **toggle** is a valid ablation. The `kairos.yml` and `orthrus.yml` configs
   differ in much more than `use_memory` — heads 4 vs 8, dropout 0.0 vs 0.5,
   `use_node_feats_in_gnn` False vs True, featurisation, and `node_label_features`
   (`kairos.yml:8-11` uses `subject: path`; `orthrus.yml:9-13` uses
   `subject: type, path, cmd_line`). The cross-config comparison is **not** an ablation.
3. Seeds must be pinned and the exact command lines recorded.

Out of scope for the Related Work chapter; recorded here because it is cheap and would
de-risk the thesis's lead hypothesis before the main implementation lands.

---

## 5. Windowing and splits

| Property | Value | Source |
| --- | --- | --- |
| Window | `construction.time_window_size`, default **15.0 min** in every system config | `config/*.yml` |
| Window cut | `window_size_in_ns = time_window_size * 60_000_000_000`; boundary check at the batch level | `build_default_graphs.py:316-322` |
| Splits | hardcoded calendar dates per dataset, in `train/val/test/unused` buckets | `pidsmaker/config/config.py:13-15+` |
| GT windows | attack ground truth pinned to absolute timestamps | `config/config.py:19-25` |

Note the `unused_dates` bucket: days are manually dropped from evaluation per dataset.
Worth auditing — it is a curation step with no stated criterion `[unverified]`.

Hardcoded filters `[import-note]`: `edge_reversed` (10 event types,
`dataset_utils.py:2-14`); `exclude_edge_type = {EVENT_FCNTL, EVENT_OTHER,
EVENT_ADD_OBJECT_ATTRIBUTE, EVENT_FLOWS_TO}` (`:17-24`); per-dataset GT file allowlists
with manual exclusions carrying inline rationale, e.g. `config.py:47-51`
`# "E3-THEIA/node_Phishing_E_mail_Executable_Attachment.csv", # attack failed so we don't use it`.

---

## 6. The fidelity question — the open item that matters most

VELOX's central claim is comparative: a linear encoder matches the elaborate systems.
That claim rests entirely on PIDSMaker's **reimplementations** of those systems being
faithful.

They are demonstrably *not* line-for-line identical to the originals — the standalone
ORTHRUS repo has no memory module at all, while PIDSMaker's ORTHRUS runs a TGN wrapper
with memory disabled; the standalone ORTHRUS has no time encoder, while PIDSMaker has a
working `TimeEncoder` that ORTHRUS's config simply does not enable. Those two happen to
agree behaviourally, but that is a coincidence of configuration, not evidence of
fidelity in general.

`[unverified]` **Open:** systematically diff PIDSMaker's `kairos` and `orthrus`
configurations against the standalone repos — encoder, decoder, objective,
featurisation, neighbour size, split. If they diverge materially, every VELOX comparison
number is affected, and with it the "simpler is better" conclusion. This is the single
highest-value verification remaining in the related-work analysis.

---

## 7. Positioning on the thesis axes (VELOX row)

| Axis | VELOX |
| --- | --- |
| Node memory | none (`used_methods: none`) |
| Time representation | none (`intra_graph_batching: edges`, no temporal sampler) |
| Node keying | UUID (inherited pipeline) |
| Encoder | `nn.Linear` + dropout |
| Proxy task | edge-type prediction |
| Granularity | edge |
| Window | 15 min |

---

## 8. Open items

- Diff the reimplementations against the standalone repos (§6).
- Audit the `unused_dates` curation (§5).
- Confirm whether `TimeEncodingMemory` is used by any shipped system config, or is
  dead code in practice.
