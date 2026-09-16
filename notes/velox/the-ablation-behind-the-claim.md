# VELOX — the ablation behind "simpler is better"

Extends §1 of [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md),
which records which code this analyses and why. That note established *what VELOX is*. This one
asks **what experiment licenses it**, and answers two questions: which encoders the grid search
contained, and whether the grid ships.

Analysed against the artifact, `upstream/velox` @ `54f687c` (2026-09-16).
Paper text: `resources/VELOX.txt`. Provenance markers as in the companion note.

---

## 0. Answer in one paragraph

VELOX's design is justified by the Fig. 9 grid search over Objective × Encoder × Featurisation
(`resources/VELOX.txt:857-895`). Two facts about that grid change how much it can support.
First, **KAIROS's encoder — the only memory-bearing encoder in the study — was explicitly
excluded from it** `[paper]`. The encoders compared are ORTHRUS's, NodLink's and "Linear", and
on the artifact `use_memory: True` appears in exactly one config, `kairos.yml:53` `[read]`. So
the conclusion "a simple linear layer suffices as an encoder" was reached without ever putting a
linear layer against node memory. Second, **the grid does not ship** `[read]` — and this is now
checked against the paper's own artifact branch, not a proxy: `config/experiments/` contains the
Fig. 8 featurisation sweep and per-system hyperparameter sweeps, and no sweep varies the encoder
or the objective across more than one value.

This is the same shape of gap found in ORTHRUS
(`related-work/orthrus/notes/the-published-ablation.md`), and the symmetry belongs in the
synthesis: the paper claiming node memory is unnecessary and the paper claiming the graph is
unnecessary both rest on ablations their artifacts do not reproduce. With one twist that cuts
the other way — ORTHRUS's Table 6 harness turns out to live *here*, in VELOX's artifact, and it
names `use_memory` explicitly. See §2a; it sharpens the ORTHRUS section rather than overturning it.

---

## 1. KAIROS's encoder is excluded from the grid

`[paper]` `resources/VELOX.txt:884-887`, in the SC7 "Overly Complex Architectures" section that
sets up Fig. 9:

> **Encoders.** For the encoders, we focus on those utilized in ■ O RTHRUS and ■ N OD L INK,
> chosen for their demonstrated superior performance in prior experiments (Fig. 6). ■ K AIROS'
> encoder is excluded due to its architectural similarity to ■ O RTHRUS and its consistently
> lower performance. Additionally, we introduce a new encoder, referred to as "Linear," which
> consists of a single linear layer.

`[read]` Cross-referencing the artifact's configs settles what that excludes. `use_memory` is
`True` in `config/kairos.yml:53` and `False` in `orthrus.yml:88`,
`orthrus_non_snooped.yml:97`, `nodlink.yml:62` and `velox.yml:96`. **No cell of Fig. 9 contains
a node memory.**

The exclusion is defended on two grounds, and both are worth separating:

- *"architectural similarity to ORTHRUS"* — this is precisely the claim at issue. KAIROS's
  encoder differs from ORTHRUS's in the two respects this thesis cares about: it carries a
  per-node memory and it has a genuine Fourier Δt encoder feeding edge attention (see
  `related-work/kairos/notes/`). Calling them similar presumes the answer.
- *"consistently lower performance"* — true in their Fig. 6, but that compares whole systems,
  not encoders. KAIROS's weak showing is confounded by its featurisation (HFH), its batch-level
  detection granularity and its thresholding, all of which SC1–SC3 criticise independently.

`[unverified]` Nothing here says memory *would* have helped. The point is narrower and safe: the
experiment behind "a simple linear layer suffices" did not test it, so it cannot speak to it.

## 2. The grid does not ship

`[read]` The artifact's sweep configurations, in full:

```
config/experiments/tuning/components/tuning_featurization_methods.yml
config/experiments/tuning/systems/default/tuning_default_baselines.yml
config/experiments/tuning/systems/default/tuning_orthrus_non_snooped.yml
config/experiments/tuning/systems/default/tuning_tokenization_hetero.yml
config/experiments/tuning/systems/cadets_e3/tuning_{flash,kairos,magic,nodlink,orthrus,provd,rcaid,sigl,threatrace}.yml
config/experiments/tuning/systems/clearscope_e5/tuning_{flash,kairos,magic,orthrus,rcaid,sigl,threatrace}.yml
config/experiments/uncertainty/{no_experiment,run_n_times,run_n_times_hyperparameters,uncertainty}.yml
```

Grepping all of them for `used_methods` returns exactly two hits:
`cadets_e3/tuning_nodlink.yml:17` sweeps `preprocessing.transformation.used_methods`, and
`default/tuning_tokenization_hetero.yml:18` sets
`detection.gnn_training.encoder.used_methods` to a **single** value
(`"entity_type_encoding,token_encoding,tgn,graph_attention"`) — a hetero-tokenisation
experiment, not an encoder comparison. No file anywhere varies
`detection.gnn_training.decoder.used_methods` at all `[read]`.

So the two axes of Fig. 9 that matter — Objective ∈ {edge type, node type, node feature
reconstruction} and Encoder ∈ {ORTHRUS, NodLink, Linear} — are not sweepable from any shipped
configuration. Both *can* be set by hand one run at a time: `config/velox.yml:100-116` carries
fully-specified blocks for all three objectives with `used_methods: predict_edge_type` selecting
one `[read]`. What is missing is the grid that produced the figure, and with it the record of
which cells ran and how they were averaged.

This closes the open item the earlier draft carried: fetching the `velox` branch was the check,
and the grid is not there either.

## 2a. But ORTHRUS's Table 6 harness *is* here — and it toggles `use_memory`

`[read]` A finding for the ORTHRUS section, not this one, turned up in the same sweep:
`scripts/run_orthrus_ablation.sh` reconstructs ORTHRUS's four Table 6 ablations, and it is
present in **VELOX's** artifact although
`related-work/orthrus/notes/the-published-ablation.md` established it is absent from ORTHRUS's
own repository. Its four arms are Featurization (`hierarchical_hashing`), a word2vec control,
Clustering (`use_kmeans=False`), and Encoding:

```sh
./run_serial.sh orthrus THEIA_E3 --exp=E3_encoding \
  --detection.gnn_training.encoder.tgn.use_memory=True \
  --detection.gnn_training.decoder.predict_edge_type.used_method=kairos
```

Three consequences, in order of importance to this thesis:

1. **The "Kairos' TGN encoder" swap turns node memory ON.** The ORTHRUS note listed what that
   ablation changed as unknown. It changes `use_memory` — so memory *was* one of the factors changed,
   which is more than the ORTHRUS artifact could tell us.
2. **It is still confounded, so F2 survives.** The arm sets two flags: memory *and* a `kairos`
   decoder variant. A result attributed to memory is therefore attributable to either. No
   isolated memory experiment exists in any of the three repositories.
3. **It is stale against this branch's schema.** `src/config.py:232-236` defines
   `predict_edge_type` with keys `decoder`, `**DECODERS` and `balanced_loss` — there is no
   `used_method`. So the second flag names a key this schema does not define, and the script is
   a leftover from an earlier revision rather than a runnable reconstruction.
   `[unverified]` Whether an unknown override key raises or is ignored at runtime was not traced
   (`get_runtime_required_args` uses `parse_known_args`, `src/config.py:703-739`).

`[unverified]` Do **not** cite this as "ORTHRUS's ablation is reproducible after all". Cite it
as: the harness exists, it names memory explicitly, and it changes memory together with a
decoder — which is exactly the confound the ORTHRUS section already argues for, now with the
script as evidence rather than inference. The ORTHRUS notes should be updated to point here.

## 3. CORRECTION — the baseline tuning is far more thorough than `main` suggests

An earlier draft of this note, reading `main`, described SC4's tuning as "an 8-point grid shared
by all baselines". **That was wrong for the artifact**, and the correction runs in the paper's
favour.

`[read]` The artifact ships `config/tuned_baselines/<dataset>/tuned_<system>.yml` — roughly
seventy files covering nine datasets (`cadets_e3`, `cadets_e5`, `clearscope_e3`, `clearscope_e5`,
`theia_e3`, `theia_e5`, `optc_h051`, `optc_h201`, `optc_h501`) × up to ten systems, VELOX
included (`tuned_velox.yml` in every dataset directory, plus `tuned_velox_alacarte.yml` for
`cadets_e3` and `theia_e3`). It also ships `config/tuned_components/` — seven featurisation
configs (`tuned_word2vec`, `tuned_doc2vec`, `tuned_fasttext`, `tuned_hfh`,
`tuned_word2vec_alacarte`, `tuned_word2vec_flash`, `no_featurization`) matching Fig. 8's method
axis exactly.

So the *outcome* of tuning is recorded per system per dataset, even though the Fig. 9 grid is
not. `tuning_default_baselines.yml` is only the fallback starting grid (time window {5, 15} min,
lr {1e-3, 1e-4}, hid dim {64, 128}), and systems with a dedicated file under
`tuning/systems/<dataset>/` use that instead.

`[read]` One more point in the paper's favour: every sweep file declares

```yaml
metric:
  name: val_score
  goal: maximize
```

so hyperparameter selection maximises a **validation** score. SC4 complains that "in the PIDS
literature, hyperparameter tuning often relies on test data" (`resources/VELOX.txt:752-754`)
`[paper]`, and the artifact does not. Credit it. That makes the epoch-selection finding in
[`metrics-splits-and-reported-results.md`](metrics-splits-and-reported-results.md) §3 sharper
rather than weaker: they were careful about tuning on validation and still pick the *reported
epoch* by test ADP.

`[read]` Residual asymmetry, much smaller than the earlier draft claimed: `cadets_e3` has
dedicated tuning files for nine systems but `clearscope_e5` for seven, and only ORTHRUS has a
`tuning_orthrus_non_snooped.yml`. Worth one clause, not a paragraph.

## 4. What this does and does not license

| Claim | Supported by the artifact? |
| --- | --- |
| "Edge-type prediction beats node-type prediction and feature reconstruction" | **No** — Fig. 9's objective axis is not sweepable from any shipped config |
| "A simple linear layer suffices as an encoder" | **No** — same grid; and its encoder axis never included a memory-bearing encoder |
| "Text embeddings matter more than the encoder" | **Yes** — the Fig. 8 featurisation sweep and its seven tuned component configs ship |
| "Training embeddings on train-only costs little" | **Yes** — `featurization.embed_nodes.training_split` plus that sweep's train/all arm |
| "We tuned every baseline" | **Yes** — ~70 per-dataset per-system tuned configs, selected on `val_score` |
| "VELOX is cheap" | **Yes** for the architecture; see the real-time caveat in [`metrics-splits-and-reported-results.md`](metrics-splits-and-reported-results.md) §7 |

## 5. For the chapter

Lead the VELOX section with the floor, not with the gap — the floor is the honest and important
part, and this thesis has to clear it. Credit the tuning and the de-snooping; both are real and
better than what the literature does. Then make one narrow, defensible point: the grid that
licenses "simpler is better" contains no memory, by explicit choice, so VELOX is evidence that
*graph structure* contributed little on these benchmarks and is silent on node memory. That
keeps the chapter's thesis statement intact and sharpens it — the received story reads VELOX as
the strongest form of "none of this machinery is needed", and the artifact shows it never tested
the piece this thesis is about.

**Related:** [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md) ·
`related-work/orthrus/notes/the-published-ablation.md`
