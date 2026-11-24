# VELOX — objective, training lifecycle and featurisation

Companion to [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md).
That note covered the encoder; this one covers **what the model is trained to predict**, **what
state survives an edge**, and **what the word2vec pipeline actually feeds it** — including the
one shortcoming (SC6) VELOX genuinely fixes in code.

Analysed against the artifact, `upstream/velox` @ `54f687c` (2026-09-16) — see the companion
note for why `main` is the wrong target. Paper text: `resources/VELOX.txt`.
Provenance markers as in the companion note.

---

## 0. Answer in one paragraph

VELOX's objective is ORTHRUS's, unchanged: a ten-way cross-entropy over edge types whose
per-edge loss *is* the anomaly score at inference `[read]`. Nothing persists between edges — no
memory, and now not even a neighbour buffer — so the model is a pure function of the two
endpoint feature vectors and is trained edge-by-edge with no sequential structure whatsoever.
The featurisation is where VELOX makes its one substantive code-level improvement over ORTHRUS:
word2vec is trained on the training split only and out-of-vocabulary tokens fall back to a zero
vector, which is exactly what SC6 prescribes and is verifiable in the artifact `[read]`.

---

## 1. Objective: ten-way edge-type cross-entropy

`[read]` `config/velox.yml:100-116`, copied unchanged from the de-snooped ORTHRUS:

```yaml
decoder:
  used_methods: predict_edge_type
  predict_edge_type:
    balanced_loss: False
    decoder: edge_mlp
    edge_mlp:
      architecture_str: linear(0.5) | relu
```

Blocks for `reconstruct_node_features` (`:107-111`) and `predict_node_type` (`:112-116`) are
fully specified alongside it but not selected — the Fig. 9 objective axis, present as options,
absent as a sweep. The objective is built at `src/factory.py:288-299`; its class body is:

```python
def forward(self, h_src, h_dst, edge_type, inference, **kwargs):
    h = self.decoder(h_src=h_src, h_dst=h_dst)
    class_weights = compute_class_weights(...) if self.balanced_loss else None
    edge_type_classes = edge_type.argmax(dim=1)
    loss = self.loss_fn(h, edge_type_classes, inference=inference, class_weights=class_weights)
    return {"loss": loss}
```

`num_edge_types: 10` for every DARPA dataset in `src/config.py` (e.g. `:400` for CADETS_E3) `[read]`, matching the paper's "evaluated across ten different system calls"
(`resources/VELOX.txt:944`) `[paper]`. The `inference` flag switches the loss between a
reduced scalar (training) and a per-edge vector (scoring), so **the training loss and the
anomaly score are the same quantity** — the surprise-based detection scheme ORTHRUS uses.

`[read]` The structural asymmetry the paper attributes to the encoder lives here instead:
`src/decoders.py:13-19` projects source and destination through separate `lin_src` / `lin_dst`
layers — each widening to `in_dim*2` — then **concatenates** them:

```python
self.lin_src = Linear(in_dim, in_dim*2) # TODO: not sure it's good to project in *2, try *1 instead
self.lin_dst = Linear(in_dim, in_dim*2)
...
h = torch.cat([self.lin_src(h_src), self.lin_dst(h_dst)], dim=-1)
```

So "emitting" and "receiving" roles are modelled — one layer deep, after a shared encoder. The
authors' own `TODO` on the widening factor is worth a footnote: it is an unexamined choice in
the system the paper proposes as minimal. (`main` replaced this with a sum-based
`EdgeLinearDecoder`; the concatenating version is commented out there at
`src/decoders.py:360-370` on the artifact — the two repositories disagree about which is live.)

## 2. Training lifecycle: nothing persists

`[read]` With `neighbor_sampling: []` (`config/velox.yml:80`) the pipeline batches edges and
nothing else. (Added 2026-11-24: and the batch is the window. `velox.yml` sets no
`edge_batch_size`, so each 15-minute graph is one batch and one optimizer step, with the loss
averaged over its edges; see [`time-handling.md`](time-handling.md) §3.) There is:

- no memory module (`use_memory: False`, `:96`),
- no neighbour sampler — the list is empty, so none is constructed,
- no state to reset between epochs, splits or windows.

This is worth stating explicitly in the chapter because it removes a whole class of the protocol
defects found in ORTHRUS. ORTHRUS's neighbour sampler is never reset at inference, so the
validation split's adjacency leaks into test scoring
(`related-work/orthrus/notes/objective-and-training-lifecycle.md`). VELOX has no sampler, so
**that particular leak cannot occur** `[read]`. The comparison is not "VELOX is sloppier"; on
this axis it is cleaner, precisely because it models less.

What remains shared with ORTHRUS is the upstream preprocessing: `time_window_size: 15.0`
(`config/velox.yml:7`) and ORTHRUS-style graph building (`used_method: orthrus`, `:5`) `[read]`. Runs of identical operations on
the same node pair still collapse to one edge before VELOX sees them. ~~The measurement from the
ORTHRUS pass — CADETS 18,699,410 filtered events → 6,454,730 edges, ~16% of the unfiltered
corpus — applies to VELOX unchanged, since it reads the same graphs.~~ CORRECTION (2026-11-24):
VELOX's graph builder is not byte-identical to ORTHRUS's: it also writes each day's trailing
window, and its splits differ. Its own numbers are measured in [`time-handling.md`](time-handling.md) §1. A model with no temporal
mechanism is, admittedly, the one least harmed by losing burst structure.

## 3. Featurisation: SC6 is genuinely fixed

`[read]` Two changes relative to stock ORTHRUS. First, `config/velox.yml:20` sets
`featurization.embed_nodes.training_split: train` where ORTHRUS uses the whole corpus. Second,
the OOV fallback in
`src/featurization/embed_edges_methods/embed_edges_feature_word2vec.py:26-39`:

```python
zeros = np.zeros((cfg.featurization.emb_dim,))
...
word_vectors = [model.wv[word] if word in model.wv else zeros for word in tokens]
weighted_vectors = [w * v for w, v in zip(weight_list, word_vectors)]
sentence_vector = np.mean(weighted_vectors, axis=0)
normalized_vector = sentence_vector / (np.linalg.norm(sentence_vector) + 1e-12)
```

So a token absent from the training vocabulary contributes a zero vector, and a node whose
tokens are *all* OOV gets an exact zero embedding (the `+1e-12` guard keeps the division finite)
`[read]`. This matches the paper's description — "OOV tokens replaced by a zero vector",
`resources/VELOX.txt:923-925` `[paper]` — and it is one of two shortcomings VELOX addresses with
code rather than with an argument. The chapter should credit it plainly: ORTHRUS trains
word2vec over the whole database including test, and VELOX does not.

Two details worth carrying:

- **The token weights are positional, not identity-based.** `cal_word_weight` (`:7`, with
  `decline_rate: 30`, `config/velox.yml:55`) builds a declining arithmetic sequence over token
  *positions* `[read]`. An OOV token therefore zeroes its slot but keeps its positional weight in
  the mean, diluting the remaining tokens rather than renormalising over them. Minor, and it
  applies equally to ORTHRUS.
- **The embedding is L2-normalised** (`:38`), so every node feature is a unit vector before the
  node-type one-hot is concatenated `[read]`.

`[read]` Node label sources are ORTHRUS's, unchanged (`config/velox.yml:9-12`):

```yaml
node_label_features:
  subject: type, path, cmd_line
  file:    type, path
  netflow: type, remote_ip, remote_port
```

The ORTHRUS import analysis established that CADETS carries no subject path at all, so subjects
are labelled from the executable name scraped out of Event records — about 130 distinct strings
across the dataset. VELOX inherits that ceiling. Since VELOX has *only* the endpoint features to
work with, this matters more for it than for any other system in the study: on CADETS, a subject
node's entire representation is one of ~130 executable-name embeddings plus a 3-way type
one-hot. `[unverified]` That is a plausible partial explanation for VELOX's inconsistency across
datasets (strong on THEIA, weak on CLEARSCOPE), but the study does not test it and neither have we.

## 4. What the paper says the ablation showed about objectives

`[paper]` `resources/VELOX.txt:881-883` and `:893-902`: edge-type prediction beats node-type
prediction and node-feature reconstruction, attributed to there being more edges than nodes
("enabling more backpropagation steps") and to edge-level decoding modelling pairwise
interactions.

Two notes for the chapter. First, the supporting grid is not in the artifact (see
[`the-ablation-behind-the-claim.md`](the-ablation-behind-the-claim.md) §2). Second, and more
directly relevant here: **executable classification — the proxy task this thesis adopts from
GRASP — is not among the three objectives compared** `[paper]`. The objective axis is {edge
type, node type, feature reconstruction}. So VELOX's objective finding does not rank this
thesis's task, favourably or otherwise, and the chapter should say so rather than letting
"edge-type prediction performs better" stand as a verdict on node-classification proxies in
general.

**Related:** [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md) ·
[`the-ablation-behind-the-claim.md`](the-ablation-behind-the-claim.md) ·
[`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md)
