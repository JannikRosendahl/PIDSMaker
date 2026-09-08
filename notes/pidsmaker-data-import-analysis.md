# PIDSMaker — data import & preprocessing analysis

Code pass over `related-work/PIDSMaker` (fork of `ubc-provenance/PIDSMaker`, the framework
that hosts ORTHRUS, VELOX and re-implementations of KAIROS, FLASH, MAGIC, NODLINK, RCAID,
THREATRACE and OCRAPT). Repo HEAD at analysis time: `ae1e9fd`.

**Scope.** Every point in the pipeline where a decision is made about *which data enters
the model and in what form* — record selection, node and edge construction, attribute
retention, filtering, and the transformations applied before training. Each decision is
given with the code that makes it.

**Provenance format.** `path:line`.

**Delta markers.** PIDSMaker descends from two earlier codebases by the same group. Where
a decision differs from one of them, a marker follows:

- **ΔK** — differs from KAIROS (`related-work/kairos`), the original implementation.
- **ΔO** — differs from ORTHRUS (`related-work/orthrus`), the direct predecessor.

The markers are supplementary. The document is complete without them.

## Sources

| Stage | File |
| --- | --- |
| Schema | `dataset_preprocessing/create_database.sh` |
| DARPA TC E3 import | `dataset_preprocessing/darpa_tc/create_database_e3.py` |
| DARPA TC E5 import | `dataset_preprocessing/darpa_tc/create_database_e5.py` |
| OpTC import | `dataset_preprocessing/optc/create_database_optc.py` |
| Input file lists | `dataset_preprocessing/darpa_tc/filelist.py` |
| Edge/node vocabularies | `pidsmaker/utils/dataset_utils.py` |
| Graph construction | `pidsmaker/preprocessing/build_graph_methods/build_default_graphs.py` |
| Graph transformations | `pidsmaker/preprocessing/transformation_methods/` |
| Tokenizers, time, hashing | `pidsmaker/utils/utils.py` |
| Featurization | `pidsmaker/featurization/` |
| `msg` assembly and slicing | `pidsmaker/tasks/feat_inference.py`, `pidsmaker/utils/data_utils.py` |
| Dataset splits, attack windows | `pidsmaker/config/config.py` (`DATASET_DEFAULT_CONFIG`) |
| Per-system settings | `config/*.yml` |

---

## 1. Pipeline overview

Six stages. Stages 1–2 are the database import; 3–6 decide what the model actually sees.
Decisions in each stage are enumerated in §2 (`D1`…`D22`).

1. **Schema** — `create_database.sh <dataset>` creates one database with four tables.
2. **Import** — one script per corpus family (E3 / E5 / OpTC) parses raw logs → Postgres.
3. **Construction** — `build_default_graphs.py` → one `networkx.MultiDiGraph` per time
   window, saved under `graph_<date>/<tw_interval>`.
4. **Transformation** *(optional)* — `undirected`, `dag`, `rcaid_pseudo_graph`, synthetic
   attack injection.
5. **Featurization** — train a text-embedding model on the corpus of node labels.
6. **Feat-inference + batching** — build a per-edge `msg` vector, then select which slices
   become node and edge features.

Datasets configured: the six DARPA TC E3/E5 combinations (CADETS, THEIA, CLEARSCOPE),
plus `TRACE_E3/E5`, `FIVEDIRECTIONS_E3/E5`, three OpTC hosts, `ATLASV2_EDR` and
`CARBANAKV2_EDR` — 15 in total (`config/config.py:3-420`).

> **ΔO** ORTHRUS covers only the six DARPA TC datasets and has no OpTC path at all
> (`orthrus/src/create_database.py:5-18`).
> **ΔK** KAIROS covers the six DARPA TC datasets, OpTC, and StreamSpot.

---

## 2. The decisions

### D1 — Which input files are read

E3/E5 read a **hardcoded file list** per dataset:

```python
# dataset_preprocessing/darpa_tc/filelist.py:542-551
def get_filelist(dataset_name):
    if dataset_name == "THEIA_E5":   return theia_e5_list
    elif dataset_name == "CADETS_E5": return cadets_e5_list
    elif dataset_name == "CADETS_E3": return cadets_e3_list
    elif dataset_name == "THEIA_E3":  return theia_e3_list
```

- Only **four** datasets have a list. CLEARSCOPE, TRACE and FIVEDIRECTIONS fall through
  and return `None`, so they cannot be built from scratch with these scripts despite being
  fully configured; they are consumed as pre-built downloadable databases.
- `raw_dir` is hardcoded to `/data/` in both `__main__` blocks
  (`create_database_e3.py:328`), with the cfg value commented out.

> **ΔO** ORTHRUS discovers inputs by glob — `glob.glob(f"{filepath}/*json*")`
> (`orthrus/src/provnet_utils.py:484-486`) — so its databases have no record of which
> files produced them. The hardcoded lists are reproducible but cover fewer datasets.

### D2 — Which raw records are selected

Records are selected by a **substring test against the raw line**, before any parsing.

| Script | Target | Exact condition | Where |
| --- | --- | --- | --- |
| E3 | netflow | `if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' not in line: continue` | `create_database_e3.py:38` |
| E3 | subject | `if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' not in line: continue` | `:101` |
| E3 | file | `if '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' not in line: continue` | `:159` |
| E3 | event | `if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:` | `:261` |
| E5 | netflow | `if "NetFlowObject" in line:` | `create_database_e5.py:27` |
| E5 | subject | `if "schema.avro.cdm20.Subject" in line:` | `:74` |
| E5 | file | `if "avro.cdm20.FileObject" in line:` | `:117` |
| E5 | event | `if '{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line:` | `:199` |
| OpTC | all | `json.loads` on every line, then three field tests (D3) | `create_database_optc.py:37-46` |

- **E3 gates are anchored on the `{"datum":{…}` wrapper** and followed by a real
  `json.loads`, so the type name must appear in the datum position.
- **E5 gates are unanchored**: `"NetFlowObject" in line` matches the type name anywhere in
  the record, including inside a nested reference.

> **ΔK** KAIROS used unanchored gates throughout, including `if "Event" in line:` to
> harvest process identity (`kairos/DARPA/CADETS_E3/create_database.py:77`), and never
> parsed JSON.
> **ΔO** ORTHRUS is all-regex with unanchored gates for every dataset; the anchored
> `json.loads` path in E3 is new here. E5 is unchanged from ORTHRUS.

### D3 — OpTC: which hosts and object types

```python
# dataset_preprocessing/optc/create_database_optc.py:40-46
if temp_dic["hostname"].split(".")[0] != hostname:
    continue
if temp_dic["action"] not in rel2id:
    continue
if temp_dic["object"] not in ["FILE", "FLOW", "PROCESS"]:
    continue
```

- **One host per dataset**, exact match: `optc_h201` → `SysClient0201`, `optc_h501` →
  `SysClient0501`, `optc_h051` → `SysClient0051` (`dataset_utils.py:313-317`).
- **Three object types** kept; `MODULE`, `REGISTRY`, `SHELL`, `TASK`, `THREAD`,
  `USER_SESSION` never enter.
- The action filter runs at **import**, so the OpTC `event_table` is already restricted to
  the 10 `rel2id_optc` types.
- Raw text is modified before parsing: `line = line.replace("\\\\", "/")` (`:37`, `:177`).

> **ΔK** KAIROS mixed six hosts into one corpus and used *different* host lists for the
> benign and evaluation imports (`0209` vs `0207`), matched by substring scan of the raw
> line.

### D4 — Which CDM record types become nodes

Three node tables, one per type (`create_database.sh:32-58`):

| Table | CDM record type | Columns |
| --- | --- | --- |
| `subject_node_table` | `Subject` | `node_uuid, hash_id, path, cmd, index_id` |
| `file_node_table` | `FileObject` | `node_uuid, hash_id, path, index_id` |
| `netflow_node_table` | `NetFlowObject` | `node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port, index_id` |

- `Subject.type` and `FileObject.type` are **never read**, so `SUBJECT_PROCESS` /
  `_THREAD` / `_UNIT` / `_BASIC_BLOCK` collapse to one node type, and
  `FILE_OBJECT_FILE` / `_DIR` / `_NAMED_PIPE` / `_UNIX_SOCKET` / `_PEFILE` / `_BLOCK` /
  `_CHAR` / `_LINK` collapse to another. The model sees a 3-valued typing
  (`dataset_utils.py:301-308`).
- **CDM record types never imported**: `Principal`, `MemoryObject`, `SrcSinkObject`,
  `UnnamedPipeObject`, `IpcObject`, `RegistryKeyObject`, `PacketSocketObject`,
  `ProvenanceTagNode`, `TagRunLengthTuple`, `Value`, `CryptographicHash`,
  `UnitDependency`, `Host`, `TimeMarker`, `StartMarker`/`EndMarker`. Any event whose
  `predicateObject` is one of these fails D9 and is dropped with it.
- OpTC writes into the same three tables; `PROCESS` objects become subject rows.

> **ΔO** ORTHRUS additionally routes `FILE_OBJECT_UNIX_SOCKET` into the **netflow** table
> with all four address fields `None`, and excludes it from the file table, for CADETS
> E3/E5 only (`orthrus/src/create_database/cadets_e3.py:55-67`). PIDSMaker has no such
> special case — unix sockets stay `FileObject`s in every dataset.
> **ΔK** KAIROS had no `index_id` column; it built a separate `node2id` table.

### D5 — What identifies a node

```python
# create_database_e3.py:136-138  (identically at :190-191, and in create_database_e5.py)
datalist.append([i] + [stringtomd5(i)] + subject_obj2hash[i] + [index_id])
#                       ^^^^^^^^^^^^^^ i is the UUID
subject_uuid2hash[i] = stringtomd5(i)
```

- All three node types are keyed on **`sha256(uuid)`** — one node per entity. Two `sshd`
  processes are two nodes with two memory slots.
- `stringtomd5` is named for MD5 and computes SHA-256, with the comment
  `# TODO: check why we don't use hierarchical hashing here` (`create_database_e3.py:21-25`).
- OpTC assigns `index_id` per uuid directly (`create_database_optc.py:103-105`); its
  `hash_id` is `stringtomd5(str(attr_tuple))`, a *label* hash that is written to the table
  but never used for identity or indexing.

> **ΔK** KAIROS keyed nodes on `sha256(label)` — the process's `exec` name, the file's
> path, the netflow 4-tuple. All entities sharing a label collapsed into one node: on
> CADETS E3, 613,713 uuid rows became 268,242 nodes
> (`kairos/DARPA/CADETS_E3/create_database.py:235-251`), and every process that ever ran
> `sshd` shared one memory slot. **This is the single largest behavioural difference
> between the two codebases.**
> **ΔO** Same as ORTHRUS.

### D6 — In what order nodes are numbered

`index_id` is assigned incrementally during import, across the three tables in the order
**netflow → subject → file** (`create_database_e3.py:333-347`). This integer is the node
id used by the model and the TGN memory slot.

> **ΔK** KAIROS assigned indices in a separate pass, file → subject → netflow.

### D7 — Which node attributes are extracted

**E3 — structured access after `json.loads`:**

```python
# create_database_e3.py:110-125
uuid = subject["uuid"]
cmd = "null"
cmd_raw = subject.get("cmdLine")
if isinstance(cmd_raw, str):        # cadets_e3
    cmd = cmd_raw
elif isinstance(cmd_raw, dict):     # theia_e3 / clearscope_e3
    cmd = cmd_raw.get("string", "null")
path = "null"
props = subject.get("properties", {}).get("map", {})
if "path" in props:
    path = props["path"]
```

| Node | Fields read | Missing value |
| --- | --- | --- |
| netflow | `localAddress`, `localPort`, `remoteAddress`, `remotePort`, each unwrapped from `{"string":…}` / `{"int":…}` if a dict (`:47-61`) | `"null"` per field |
| subject | `cmdLine` (str **or** dict), `properties.map.path` | `"null"` |
| file | `baseObject.filename` → `baseObject.path` → `properties.map.filename` → `properties.map.path`, later assignments overwriting earlier so `properties.map` wins (`:167-179`) | `"null"` |

**E5 — positional regex:**

- netflow `'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}'`
- subject `'avro.cdm20.Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"path":"(.*?)"'` → `[path, cmd]`
- file `'avro.cdm20.FileObject":{"uuid":"(.*?)",(.*?)"filename":"(.*?)"'`

**OpTC** (`create_database_optc.py:48-91`): actor → `properties.image_path`; `FILE` →
`properties.file_path`; `FLOW` → `src_ip / src_port / dest_ip / dest_port`; `PROCESS`
object → `properties.parent_image_path` and `properties.command_line`. Every field
defaults to `None` when absent rather than dropping the record.

> **ΔO** This is where PIDSMaker diverges most from ORTHRUS, which hand-wrote a separate
> module per dataset with divergent and partly broken extraction:
> - ORTHRUS CADETS E3/E5 store subject `path` as **always `None`** and put the `exec` name
>   in `cmd` (`orthrus/src/create_database/cadets_e3.py:97-107`); PIDSMaker reads the real
>   `properties.map.path` for every E3 dataset.
> - ORTHRUS CLEARSCOPE E3 stores `subject_cmd[0][0]` — the **first character** of the
>   command line (`orthrus/src/create_database/clearscope_e3.py:81`); PIDSMaker stores the
>   whole string.
> - ORTHRUS THEIA E3/E5 and CLEARSCOPE E3/E5 assign `dstaddr = res[0][0]` and
>   `dstport = res[0][1]`, indexing into the **uuid string** rather than the matched remote
>   address (`orthrus/src/create_database/theia_e3.py:32-33`), so the remote address is a
>   single hex character in four of six datasets. PIDSMaker's `.get()`-based extraction has
>   no such bug.
>
> **ΔK** KAIROS stored exactly **one** label field per node type and scraped CADETS E3
> subject identity from `"exec"` on Event lines rather than from `Subject` records.

### D8 — Which edge types are dropped at import

```python
# pidsmaker/utils/dataset_utils.py:18-26
exclude_edge_type = set([
    "EVENT_FCNTL",                  # EVENT_FCNTL does not have any predicate
    "EVENT_OTHER",                  # EVENT_OTHER does not have any predicate
    "EVENT_ADD_OBJECT_ATTRIBUTE",   # used to add attributes to an object incomplete at publish
    "EVENT_FLOWS_TO",               # No corresponding system call event
])
```

A blacklist of four, matched against the parsed `type` field. Everything else is stored.

⚠️ The type is read **before** the guard, positionally and unguarded:

```python
# create_database_e3.py:262
relation_type = re.findall('"type":"(.*?)"', line)[0]
```

It takes the **first** `"type"` key in the line; a matching line without one raises
`IndexError`.

> **ΔK** KAIROS used a per-dataset **whitelist** applied at graph-construction time, with
> a different type set per dataset (7 for CADETS E3, 8 for CLEARSCOPE E3, 9 or 10
> elsewhere), and excluded `EVENT_FLOWS_TO` by a whole-line substring test — so a record
> merely *mentioning* the string anywhere was dropped.
> **ΔO** Identical to ORTHRUS.

### D9 — What makes a valid edge

```python
# create_database_e3.py:271-275
if subject_uuid[0] in subject_uuid2hash and (
    predicateObject_uuid[0] in subject_uuid2hash
    or predicateObject_uuid[0] in file_uuid2hash
    or predicateObject_uuid[0] in net_uuid2hash
):
```

- Source must be a subject; destination may be a **subject**, file or netflow ⇒
  **subject→subject edges are kept**, so the process tree (`EVENT_CLONE`,
  `EVENT_EXECUTE` between processes) is present.
- Endpoint resolution is an `if/elif/else` chain, file → netflow → subject
  (`:283-288`), so no silent overwrite.
- Both endpoints must already exist as nodes; events touching un-imported entity types are
  dropped.

> **ΔK** KAIROS stored subject→(file|netflow) only, for five of its six DARPA
> experiments — the process tree was absent and attack lineage existed only through shared
> files and sockets. Its THEIA E3 notebook, the one exception, resolved endpoints through
> a chain of independent `if`s where a uuid present in more than one map took the **last**
> match.
> **ΔO** Identical to ORTHRUS.

### D10 — Which edge directions are reversed

```python
# pidsmaker/utils/dataset_utils.py:2-14
edge_reversed = [
    "EVENT_EXECUTE", "EVENT_LSEEK", "EVENT_MMAP", "EVENT_OPEN", "EVENT_ACCEPT",
    "EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_READ_SOCKET_PARAMS",
    "EVENT_CHECK_FILE_ATTRIBUTES", "READ",
]
```

Applied at import: `src` and `dst` are swapped so information flows object → process.
One shared list for every DARPA TC dataset.

OpTC reverses on `READ` **and** on inbound flows:

```python
# create_database_optc.py:189, :201-204
if operation in ["READ"]:
    reverse_flag = True
...
if (temp_dic["object"] == "FLOW") and ("direction" in temp_dic["properties"]):
    if temp_dic["properties"]["direction"] == "inbound":
        reverse_flag = True
```

> **ΔK** KAIROS had a different reversal list per dataset and never reversed `EVENT_OPEN`
> or `EVENT_EXECUTE` — those pointed process → object. `EVENT_READ` was reversed in some
> datasets and not others. For CADETS E3, `EVENT_ACCEPT` and `EVENT_RECVMSG` were reversed
> at import but then excluded by the type whitelist, making the reversal dead code.
> KAIROS declared an OpTC reversal list and never applied it.
> **ΔO** ORTHRUS's list is the same 10 DARPA entries; the bare `"READ"` is added here for
> OpTC.

### D11 — What is stored per edge

```sql
-- dataset_preprocessing/create_database.sh:14-22
CREATE TABLE event_table (
    src_node VARCHAR, src_index_id VARCHAR, operation VARCHAR,
    dst_node VARCHAR, dst_index_id VARCHAR, event_uuid VARCHAR NOT NULL,
    timestamp_rec BIGINT, _id SERIAL PRIMARY KEY);
```

Extracted per event: subject uuid, predicateObject uuid, `type`, `timestampNanos`, and
the event's own uuid.

**Discarded from every event**: `size`, `sequence`, `threadId`, `programPoint`,
`predicateObject2` (so the second endpoint of two-object events is lost), `properties`,
`hostId`, `location`, `name`, `parameters`.

> **ΔK** KAIROS had no `event_uuid` column — edges were not individually addressable, so
> node-level ground truth could not be joined back to specific events.

### D12 — Timestamp representation

- DARPA TC: `timestampNanos` verbatim, int64 nanoseconds.
- OpTC: `OPTC_datetime_to_timestamp_US` returns `int(seconds*1000 + ms) * 1_000_000`
  (`utils/utils.py:127-140`) — nanoseconds, unifying OpTC with DARPA TC. Parsing assumes
  exactly a `-04:00` suffix and a 3-digit fraction.

> **ΔK** KAIROS left OpTC timestamps in **milliseconds** while DARPA TC was in
> nanoseconds, and StreamSpot had no timestamps at all — `t` was the Postgres row serial.

### D13 — Which days are built

```python
# pidsmaker/config/pipeline.py:858-869
dates = cfg.dataset.train_dates + cfg.dataset.val_dates + cfg.dataset.test_dates
```

Only train, val and test dates are built. **`unused_dates` are never constructed at all**
— see §3.3 for the per-dataset lists.

> **ΔO** ORTHRUS builds **every** day in `start_end_day_range` and applies the split later
> by selecting graph directories (`orthrus/src/graph_construction/build_orthrus_graphs.py:107`).
> **ΔK** KAIROS built every day in a fixed range and had no validation split; its test
> script re-ran training days to seed node-IDF statistics.

### D14 — Day boundaries and event ordering

```python
# build_default_graphs.py:255-259
sql = """select * from event_table
         where timestamp_rec>'%s' and timestamp_rec<'%s'
         ORDER BY timestamp_rec, event_uuid;"""
```

- Boundaries at **US/Eastern midnight** (`utils/utils.py:99-110`), strict `>` / `<`.
- Tie-break on `event_uuid` makes ordering deterministic across runs.

> **ΔK** KAIROS ordered by `timestamp_rec` alone, so events sharing a nanosecond were
> ordered arbitrarily.

### D15 — How time windows are cut

```python
# build_default_graphs.py:315-326
BATCH = 1024
window_size_in_ns = cfg.construction.time_window_size * 60_000_000_000
...
if (len(batch_edges) < BATCH) or (temp_list[-1] == events_list[-1]):
    last_batch = True
if (batch_edges[-1][-2] > start_time + window_size_in_ns) or last_batch:
```

- A window closes at the first **1024-edge boundary after** `time_window_size` has
  elapsed. Windows are therefore ≥ the nominal size, with data-dependent length.
- `time_window_size: 15.0` (minutes) in every shipped config.
- Windows are **materialised at construction**: changing the size requires re-running the
  whole construction stage.
- The `or last_batch` branch flushes the day's trailing partial window.

> **ΔO** ORTHRUS has no `last_batch` branch, so the trailing partial window of **every
> day is silently discarded** (`orthrus/src/graph_construction/build_orthrus_graphs.py:146`).
> **ΔK** KAIROS built one graph per *day* and applied the 15-minute window only at test
> time, inside the inference loop — the window was a runtime choice, not a stored artifact.

### D16 — Whether repeated edges are collapsed

```python
# build_default_graphs.py:337-372  (fuse_edge branch)
for (src, dst), data in edge_info.items():
    sorted_data = sorted(data, key=lambda x: x[0])
    operation_list = [entry[1] for entry in sorted_data]
    # ... collect the start index of each run of identical operations ...
    for k in indices:
        edge_list.append({"src": src, "dst": dst, "time": sorted_data[k][0],
                          "label": sorted_data[k][1], "event_uuid": sorted_data[k][2]})
```

Within a window, edges are grouped by `(src, dst)`, sorted by time, and **each run of
consecutive identical operations collapses to one edge** carrying the run's first
timestamp and `event_uuid`. `fuse_edge: True` in every shipped config.

Repetition counts are destroyed: a process reading a file 10,000 times in a window becomes
one edge.

> **ΔO** In ORTHRUS fusion is unconditional — there is no flag. PIDSMaker made it
> configurable but left it on.
> **ΔK** KAIROS did not fuse; every event was an edge.

### D17 — Which edge types survive into the graphs

```python
# build_default_graphs.py:200
include_edge_type = rel2id
```

`rel2id_darpa_tc` (`dataset_utils.py:27-48`) — 10 types, and simultaneously the one-hot
vocabulary and the classifier's label set: `CONNECT, EXECUTE, OPEN, READ, RECVFROM,
RECVMSG, SENDMSG, SENDTO, WRITE, CLONE`.

- Consequence: `EVENT_CLOSE`, `EVENT_MMAP` and `EVENT_LSEEK` are stored in Postgres but
  dropped at construction.
- `rel2id_optc` (10 types) is instead applied at import (D3).
- `possible_events_darpa_tc` (`dataset_utils.py:213-259`) defines legal
  `(src_type, dst_type) → event` triplets, used only when `batching.edge_features`
  contains `edge_type_triplet`, which expands the vocabulary to 27 typed triplets. Not
  enabled in the shipped configs.

> **ΔK** KAIROS's whitelist differed per dataset and included `EVENT_CLOSE` for CADETS E3
> and CLEARSCOPE E3 but not `EVENT_CLONE` anywhere.
> **ΔO** Identical to ORTHRUS.

### D18 — Which attributes become the node label

```python
# build_default_graphs.py:52-62
label_str = " ".join([attrs[label_used] for label_used in node_label_features[node_type]])
if use_hashed_label:
    label_str = stringtomd5(label_str)
```

Attributes available and what may be selected (`config/config.py:815-833`):

| Node type | Available in `attrs` | Selectable |
| --- | --- | --- |
| subject | `type`, `path`, `cmd_line` | `type, path, cmd_line` |
| file | `type`, `path` | `type, path` |
| netflow | `type`, `local_ip`, `local_port`, `remote_ip`, `remote_port` | **only** `type, remote_ip, remote_port` |

⇒ `local_ip` / `local_port` are read from the database into `attrs`
(`build_default_graphs.py:76-82`) but **cannot be selected** — the source address is never
featurized.

Per-system selections (`config/*.yml`), all with `use_hashed_label: False`:

| Config | subject | file | netflow |
| --- | --- | --- | --- |
| `orthrus`, `threatrace`, `ocrapt` | `type, path, cmd_line` | `type, path` | `type, remote_ip, remote_port` |
| `kairos` | `path` | `path` | `remote_ip, remote_port` |
| `flash` | `path, cmd_line` | `path` | `remote_ip, remote_port` |
| `magic`, `rcaid` | `path, cmd_line` | `path` | `remote_ip` |
| `nodlink` | `cmd_line` | `path` | `remote_ip, remote_port` |

`velox.yml` inherits from `orthrus_non_snooped` via `_include_yml`.

> **ΔO** ORTHRUS maintains **two independent label pipelines**: the config-driven label is
> written onto graph nodes with `use_hashed_label: True` (so it is a SHA-256 digest), while
> the features actually fed to the model come from a *separate* database read
> (`orthrus/src/provnet_utils.py:369-421`) with its own `use_cmd` / `use_port` flags,
> ignoring `node_label_features` entirely. PIDSMaker unified them: one
> `compute_indexid2msg` result drives both, and `use_hashed_label` now really does control
> the features.
> **ΔK** KAIROS hardcoded one label field per node type per dataset, with no configuration.

### D19 — How the label becomes a vector

**Tokenization** (`utils/utils.py:375-399`):

```python
def tokenize_subject(sentence):   # tokenize_file is identical
    new_sentence = re.sub(r"\\+", "/", sentence)
    return word_tokenize(new_sentence.replace("/", " / "))

def tokenize_netflow(sentence):
    return word_tokenize(sentence.replace(":", " : ").replace(".", " . "))
```

Empty label → `[""]`.

**Corpus** (`featurization_utils.py:29-57`, `:15-27`) — the tokenized label of every node,
**deduplicated by label string**, over the splits named by `featurization.training_split`:

```python
splits = get_splits_to_train_featurization(cfg)     # "all" -> ["train","val","test"]
nodes_to_include = set().union(*(split2nodes[split] for split in splits))
```

`training_split: all` in the shipped ORTHRUS config, so the word2vec vocabulary sees
test-set node labels. Unsupervised and label-free, so not label leakage, but transductive.

**Embedding** — `featurization.used_method` ∈ `word2vec` (ORTHRUS/VELOX default),
`doc2vec`, `fasttext`, `flash`, `alacarte`, `temporal_rw`, `hierarchical_hashing`,
`ocrapt_features`, `only_type`, `only_ones`.

word2vec: `emb_dim=128`, skip-gram, `epochs=50`, `seed=0`, `min_count=1`, `window=5`,
`negative=5`. Node vector = weighted mean of token vectors with linearly declining weights
(`decline_rate=30`), L2-normalized:

```python
# feat_inference_word2vec.py:34-40
word_vectors = [model.wv[word] if word in model.wv else zeros for word in tokens]
weighted_vectors = [weight * word_vec for weight, word_vec in zip(weight_list, word_vectors)]
sentence_vector = np.mean(weighted_vectors, axis=0)
normalized_vector = sentence_vector / (np.linalg.norm(sentence_vector) + 1e-12)
```

`hierarchical_hashing` (`feat_inference_HFH.py`) is an alternative that expands the label
into cumulative path/IP prefixes, concatenates them into one string, and applies
`FeatureHasher(n_features=emb_dim, input_type="string")`.

> ⚠️ `Dockerfile:57` pins `scikit-learn==1.2.0`, under which
> `FeatureHasher(input_type="string").transform([s])` with a bare `str` iterates the string
> **per character**. The HFH vector is therefore a signed character histogram, not a hash of
> path components — two paths that are character anagrams produce identical vectors. This
> was verified directly against the pinned version. It affects only the
> `hierarchical_hashing` path, not the word2vec default.

> **ΔO** ORTHRUS has only `feature_word2vec`, always trains on the whole database with no
> split filter, and its inference has neither an out-of-vocabulary guard nor an epsilon in
> the normalization (`orthrus/src/edge_featurization/embed_edges_feature_word2vec.py:45-49`)
> — an unseen token raises `KeyError` and a zero-norm vector yields `NaN`.
> **ΔK** KAIROS had no trained embedding at all: the node feature was a 16-dimensional
> `FeatureHasher` output over the concatenated path hierarchy, i.e. exactly the
> `hierarchical_hashing` option above, with the same character-level behaviour and the same
> pinned sklearn. Structure, component order and token boundaries were discarded.

### D20 — What the model receives per edge

```python
# tasks/feat_inference.py:57-69
msg.append(torch.cat([
    ntype2oh[graph.nodes[u]["node_type"]],
    torch.from_numpy(indexid2vec[u]),
    edge_label,
    ntype2oh[graph.nodes[v]["node_type"]],
    torch.from_numpy(indexid2vec[v]),
]))
```

```
msg = [ src_type(3) | src_emb(128) | edge_type(10) | dst_type(3) | dst_emb(128) ]   = 272
```

Validated at `data_utils.py:203-207` against
`(emb_dim*2) + (num_node_types*2) + num_edge_types`.

The `CollatableTemporalData` handed to training carries:

| Tensor | Dim | Source | Meaning |
| --- | --- | --- | --- |
| `src`, `dst` | 1 each, int64 | `index_id` | node id / TGN memory slot |
| `t` | 1, int64 | `timestamp_rec` | event time, ns |
| `msg` | 272 | above | per-edge message |
| `y` | 1, int64 | edge attr `y` | per-edge label slot, 0 at construction |

**Not featurized anywhere**: event `size` / `sequence` / `threadId` / `predicateObject2` /
`programPoint` / `properties`; subject pid / ppid / tgid / user / parent; file CDM subtype
/ permissions / size; netflow `local_ip` / `local_port` / protocol / byte counts; node
degree or age, except via the opt-in `edges_distribution` feature.

> **ΔK** KAIROS's `msg` was `[src_feat(16) | edge_type(k) | dst_feat(16)]` with no node-type
> one-hot — the type entered only as three literal characters prepended to the hashed
> label string. It also had no `y` tensor.
> **ΔO** Identical layout to ORTHRUS.

### D21 — Which slices become features, and the leakage guard

```python
# utils/data_utils.py:278-281
# If we want to predict the edge type, we remove the edge type from the message
if "predict_edge_type" in cfg.training.decoder.used_methods:
    msg = torch.cat([x_src, x_dst], dim=-1)
else:
    msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)
```

`batching.node_features` (ORTHRUS: `node_emb,node_type`) slices `msg` into
`x_src` / `x_dst` = `[emb(128) | type(3)]` = 131 dims. Alternatives: `edges_distribution`
(a ThreaTrace-style in/out edge-type histogram, max-normalized per window) and
`only_ones`.

The edge-type one-hot is **dropped from the message whenever the objective is to predict
it**, so the label is not present in the input it is predicted from.

> **ΔO** ORTHRUS has the same guard (`orthrus/src/data_utils.py:134-138`); it originates
> there, not here.
> **ΔK** KAIROS left the edge-type one-hot in `msg` unconditionally. That was not direct
> leakage — its `link_pred` saw only the two node states, and `memory.update_state(...)`
> ran after the prediction — but the property held by call ordering rather than by
> construction.

### D22 — Optional post-construction transformations

Run between construction and featurization (`transformation.used_methods`, default
`none`), each rewriting the graph the model sees:

- `undirected` — adds a reverse edge with the **same timestamp**; the docstring notes it
  is incompatible with TGN.
- `dag` — removes cycles and self-loops and deduplicates edges.
- `rcaid_pseudo_graph` — rewrites the graph structure, with optional pruning.
- `synthetic_attack_naive` — injects synthetic attacks.

Separately, `construction.mimicry_edge_num > 0` injects mimicry edges into windows
overlapping `attack_to_time_window` (`build_default_graphs.py:240-252`). Default 0.

> **ΔK / ΔO** Neither KAIROS nor ORTHRUS has any of these.

---

## 3. Reference tables

### 3.1 What a node is, per import script

| Script | Identity | subject fields | file field | netflow fields |
| --- | --- | --- | --- | --- |
| E3 | `sha256(uuid)` | `properties.map.path`, `cmdLine` | 4-way fallback | all four, `.get()`-guarded |
| E5 | `sha256(uuid)` | regex `path`, `cmdLine` | `filename` | all four, regex |
| OpTC | uuid → `index_id` | actor `image_path`; object `parent_image_path`, `command_line` | `file_path` | `src_ip, src_port, dest_ip, dest_port` |

Which of these become the *feature* label is decided later by
`construction.node_label_features` (D18), not by the import.

### 3.2 Edges, per import script

| Script | subj→subj | Type filter at import | Type filter at construction | Reversal | Edge id |
| --- | --- | --- | --- | --- | --- |
| E3 | yes | blacklist of 4 | `rel2id_darpa_tc` (10) | shared 11-entry list | `event_uuid` |
| E5 | yes | blacklist of 4 | `rel2id_darpa_tc` (10) | shared 11-entry list | `event_uuid` |
| OpTC | yes | `rel2id_optc` (10) | `rel2id_optc` (10) | `READ` + inbound `FLOW` | `temp_dic["id"]` |

### 3.3 Splits (`config/config.py:3-420`)

| Dataset | train | val | test | unused |
| --- | --- | --- | --- | --- |
| CADETS_E3 | 04-02…04-05, 04-07…04-09 | 04-10 | 04-06, 04-11, 04-12, 04-13 | — |
| THEIA_E3 | 04-02…04-08 | 04-09 | 04-10, 04-12, 04-13 | 04-11 |
| CLEARSCOPE_E3 | 04-03…04-05, 04-07…04-10 | 04-02 | 04-11, 04-12 | 04-06, 04-13 |
| CADETS_E5 | 05-08, 05-09, 05-11 | 05-12 | 05-16, 05-17 | 05-10, 05-13…05-15 |
| THEIA_E5 | 05-08, 05-09, 05-10 | 05-11 | 05-14, 05-15 | 05-12, 05-13, 05-16, 05-17 |
| CLEARSCOPE_E5 | 05-08…05-12 | 05-13 | 05-14, 05-15, 05-17 | 05-16 |
| optc_h201/h501/h051 | 09-19, 09-20, 09-21 | 09-22 | 09-23, 09-24, 09-25 | 09-16…09-18 |

All datasets: `num_node_types = 3`, `num_edge_types = 10`, window `15.0` min, US/Eastern.

Test days are **not** strictly after train days in the E3 datasets — CADETS_E3 trains on
04-02…04-09 and tests on 04-06.

> **ΔO** ORTHRUS's E5 and CLEARSCOPE_E3 splits are identical. Two differ:
> CADETS_E3 (ORTHRUS trains on days 3–10, validates on day 2; PIDSMaker trains on days
> 2–9 and validates on day 10) and THEIA_E3 (ORTHRUS trains on days 2–5; PIDSMaker extends
> training to days 2–8).
> **ΔK** KAIROS had no validation split.

---

## 4. Catalogue of exclusions

**Entity types never imported:** see D4.

**Attribute values dropped from kept records:** see D11 and D20.

| Rule | Where | Effect |
| --- | --- | --- |
| `exclude_edge_type` (4 types) | import (D8) | FCNTL / OTHER / ADD_OBJECT_ATTRIBUTE / FLOWS_TO never stored |
| `include_edge_type = rel2id` (10 types) | `build_default_graphs.py:200` (D17) | `EVENT_CLOSE`, `EVENT_MMAP`, `EVENT_LSEEK` and all others dropped at construction |
| `if len(i) != 64` | all node inserts | selects uuid keys out of the bidirectional dicts |
| both endpoints must resolve | `store_event` (D9) | events touching un-imported entities dropped |
| `unused_dates` | `get_dates_from_cfg` (D13) | whole days never built |
| host allow-list, `object` allow-list, `action not in rel2id` | OpTC import (D3) | one host, three object types, ten actions |
| `fuse_edge` | `build_default_graphs.py:337-372` (D16) | consecutive same-type edges collapsed |
| `save_indexid2msg` filter | `build_default_graphs.py:126-146` | nodes not appearing in any built graph dropped from the feature map |
| `_test_mode` `NUM_TEST_EDGES = 2000` | `build_default_graphs.py:450-451` | unit-test path truncates graphs |

**Silent-failure surface.** E3 counts successes and failures per node type and logs them
(`create_database_e3.py:88`, `:147`, `:200`), though `store_file`'s success counter is
initialised and logged but never incremented. E5 and all event parsers use bare
`except` with no counters. Nothing asserts that import loss is within bounds.

> **ΔK** KAIROS reported no counters at all and used bare `except: pass` everywhere.

---

## 5. Findings that can change downstream outcomes

### 5.1 Node identity is the UUID, so nodes are entities

D5. Every process instance is its own node with its own memory slot and neighbour list;
process lineage survives, and subject→subject edges (D9) mean the process tree is present.
This is the semantics a temporal graph model assumes.

> **ΔK** Under KAIROS's label keying, "memory per node" meant memory per *label class* —
> one slot shared by every process that ever ran `sshd`. Its published memory results are
> therefore about a different object, and node counts here are roughly 2.3× larger on
> CADETS E3, so memory-bank size and neighbour-sampling density are not comparable.

### 5.2 Features are configuration, not code — and that is a confound

D18. `node_label_features` changes what a node *is* to the model, and the shipped configs
disagree: `orthrus.yml` uses `type, path, cmd_line` for subjects, `kairos.yml` uses `path`
alone, `nodlink.yml` uses `cmd_line` alone. Two systems compared inside this framework can
differ in feature content before any architectural difference applies. Any comparison must
fix `node_label_features` across arms or report it as a variable.

### 5.3 `fuse_edge: True` destroys event multiplicity

D16, on in every shipped config. Edge counts are not comparable to the raw logs;
burst and frequency signal is removed; and a memory-based model sees one update where the
log had thousands, which directly bounds what temporal dynamics it can learn.

### 5.4 Windows are batch-quantized and baked into the artifacts

D15. Windows are ≥15 minutes with data-dependent length, quantized to 1024 edges, and
stored. Changing the window size means re-running construction. This is the reference case
for the "do time windows help" question: here the window is a materialised artifact rather
than a runtime parameter.

### 5.5 Featurization is transductive by default

D19. `training_split: all` unions train, val and test node labels into the word2vec
corpus. Label-free, so not label leakage, but a strictly inductive comparison needs
`training_split: train`. The existence of a separate `orthrus_non_snooped.yml` implies
snooping is treated as a deliberate axis here.

### 5.6 The E5 import is materially weaker than E3, and has a latent crash

D2, D7. E5 kept the older regex parser: unanchored gates, positional regexes, bare
`except`, no counters. Worse:

```python
# create_database_e5.py:85-88
except:
    try:
        subject_obj2hash[subject_uuid[0][0]] = "null"   # a string ...
...
# create_database_e5.py:96-98
datalist.append([i] + [stringtomd5(i)] + subject_obj2hash[i] + [index_id])   # ... list + str
```

The success path stores a 2-element list; the fallback stores a string. `list + str` raises
`TypeError`, so any Subject line failing the regex aborts the whole import. Any E5 result
from this repo either used the pre-built database or ran on logs where the regex never
failed.

### 5.7 Only four datasets can be built from scratch

D1. CLEARSCOPE, TRACE and FIVEDIRECTIONS have no file list and fall through to `None`,
despite being fully configured. Their databases are downloaded, so their import
provenance is not inspectable from this repo.

### 5.8 Test days are not strictly after train days

§3.3. Defensible for a per-window anomaly detector, but the split is not chronological and
does not support claims about temporal generalization.

### 5.9 OpTC process labels are role-dependent

D7. A `PROCESS` object is labelled with `parent_image_path` while an actor is labelled
with `image_path`, and the dict write is unconditional, so a uuid seen in both roles keeps
whichever record was read last. Two nodes for the same process can carry different labels
depending on file order.

### 5.10 The `hierarchical_hashing` featurization is character-level

D19. If used, the node feature is a signed character histogram, not a hash of path
components. Anagram paths are indistinguishable. Not the default here, but it is the path
a KAIROS-comparable run would take.

---

## 6. Downstream filters that also shape the results

- **Ground truth** is node-UUID CSVs under `Ground_Truth/orthrus/<DATASET>/…csv`, resolved
  to `index_id` via `get_uuid2nids` (`utils/labelling.py:94-122`), plus per-attack time
  windows in `DATASET_DEFAULT_CONFIG.attack_to_time_window`. A `threatrace` ground-truth
  version is selectable via `evaluation.ground_truth_version`, so the labelling convention
  is an explicit variable.
- **Evaluation granularity** is a choice: `node_evaluation`, `node_tw_evaluation`,
  `edge_evaluation`, `tw_evaluation`, `queue_evaluation`.
- **Thresholding** is a named method — `max_val_loss`, `mean_val_loss`, `threatrace`,
  `magic`, `flash`, `nodlink`, `ocrapt` (`config/config.py:797`) — with optional k-means
  top-K post-processing (`kmeans_top_K: 30` for node evaluation).
- **KAIROS's IDF queue survives as one option**:
  `evaluation.queue_evaluation.used_method: kairos_idf_queue`, with
  `include_test_set_in_IDF: True` and `queue_threshold: 20` (`config/default.yml`).

> **ΔK** KAIROS scored detection only at 15-minute-window granularity, against four
> hand-listed attack windows and nine hard-coded keyword strings, with a fixed
> `mean + 1.5·std` threshold. It also floored the IDF of nodes matching a per-dataset
> hand-tuned path list (`is_include_key_word`), several entries of which were broad
> substrings such as `usr`, `var`, `proc` and `Windows`. **No equivalent keyword filter
> exists in PIDSMaker.**

---

## 7. Reproduction note

Not re-verified experimentally in this pass, except the character-level `FeatureHasher`
behaviour noted in D19, which was confirmed against the pinned `scikit-learn==1.2.0`.

Worth writing as targeted checks:

1. Print the **effective window length distribution** from built graphs, against the
   nominal 15 minutes (D15, §5.4).
2. Count edges **with and without `fuse_edge`** on one day, to size the multiplicity loss
   (D16, §5.3).
3. Assert node count equals distinct uuid count on one dataset, confirming D5 — and
   compare against the KAIROS label-collapse ratio.
4. Diff the corpus produced by `training_split: all` vs `train`, to size the transductive
   gap (D19, §5.5).
5. Re-run the E5 import on a log slice containing a `Subject` record without `cmdLine`, to
   confirm the `TypeError` in §5.6.
