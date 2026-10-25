# Native CLIP4Clip checkpoint inference

This runner evaluates an explicitly supplied native ViT-B/32, 2D, loose meanP
checkpoint. It is separate from the small OpenCLIP pilot and the alignment head
training path. It does not download models/data or create cloud resources.

## Inputs and trust boundary

The current host layout uses `data/metadata/msrvtt_data.zip` and
`data/metadata/msrvtt_protocol_lock.json`. Historical Kaggle bundles retain
their authenticated internal paths under `outputs/w03/`; moving the host's
experiment directories does not change those ZIP members or manifests.
The notebook builder accepts both metadata layouts. For a historical bundle,
it derives current working paths from the verified input bytes and records
their hashes in `input_path_aliases.json`; it never rebuilds a missing archive
from extracted members or relaxes its hash check.

The small pinned retrieval evaluator and MIT license are included under
`src/temporal_hoi/evaluation/reference/`. The source snapshot is stored as
`clip4clip_metrics.py.txt`, verified by SHA256, and loaded explicitly as Python
source. Metric conformance tests therefore do not require local experiment
outputs, video data, or checkpoints.

- A complete local CLIP4Clip source tree at commit
  `508ffa3de39ba0563a03199c440ab602a72e9b6f`, including the BPE vocabulary and
  MIT license. The runner verifies the hard-coded hashes of the code it imports
  and the source defining the reference pooling/loader rules before importing.
- The metadata archive and protocol lock described in
  [the retrieval protocol](msrvtt_protocol.md). Both are local inputs, not distributed
  benchmark media. IDs, raw caption strings and candidate order remain locked.
- A native fine-tuned checkpoint, its explicit SHA256 and a provenance JSON.
  Loading uses `torch.load(..., weights_only=True)`. There is no unsafe pickle
  fallback, base-CLIP fallback, automatic download or missing-key initialization.
- Nonempty `<video_id>.mp4` files for the selected pool. Decode failures stop the
  run. Alternative extensions or corrupt files are not silently substituted.

The checkpoint provenance must contain `source_url`, `training_regime`,
`architecture="CLIP4Clip_ViT-B32_meanP_loose_2d"` and `checkpoint_sha256`.
These are provenance assertions to verify against the actual release, not proof
that a compatible tensor shape was trained on the claimed data. A checkpoint
from another implementation must not be relabeled as native merely because
parameter names match.

## Preflight and configuration

Copy [the inference configuration](../../configs/msrvtt_clip4clip_inference.json)
to a local run config and supply checkpoint/provenance/source paths. Checkpoint
fields intentionally default to null; the runner cannot infer a trusted release.
The 8 GiB RSS / 1 hour limits are proposed limits for this separate workload,
not an increase of the existing alignment budget or a measured hardware minimum.
The default device is CUDA for a separately approved GPU environment. Existing
CPU dependencies and the project lockfile remain unchanged.

```text
python scripts/run_msrvtt_baseline.py --config <run.json> --source-root <CLIP4Clip> --output-dir <new-output>
```

Default execution checks metadata and file/hash availability without loading a
model. Missing inputs produce `readiness.json`, status blocked and exit code 2;
no scores or benchmark metrics are created. Every invocation needs a fresh
output directory to preserve earlier evidence. File presence does not establish
checkpoint architecture compatibility or media decodability; execution checks
those later. No automatic dependency installation occurs.

## Sampling, model and scoring

The native decoder samples frame `sec * int(fps)` at 1 Hz until the declared
frame count. Uniform integer linspace selects at most 12 samples, matching
`slice_framepos=2`, frame order 0. The runner decodes all expected 1 Hz samples
to reject mid-clip corruption, but only transforms/stores the selected frames.
This avoids storing all transformed samples from long videos. Videos with
invalid metadata or non-increasing decoder timestamps fail explicitly.

Image transforms are imported from the verified native source: RGB, bicubic
short-side 224 resize, center crop 224 and CLIP normalization. Text uses the
native BPE, start/text/end truncation and zero padding to 32 tokens. Source
captions are not replaced by a model-generated or normalized caption list.

Loose meanP has only CLIP model parameters. The runner reconstructs the exact
ViT-B/32 topology from the pinned module and strictly loads `clip.*` tensors,
optionally stripping a uniform DDP `module.` prefix. Temporal/cross head keys
are rejected. No architecture or missing weight is silently guessed.

Each valid frame feature is L2-normalized, then averaged, then normalized again.
Text features are normalized. Scores have axes `[text,video]` and are multiplied
by `exp(logit_scale)`. Encoding only valid frames is equivalent to meanP's masked
padding in this evaluation-only ViT topology. Batch size affects memory, not the
candidate pool. This implementation executes in FP32 and records that choice;
real checkpoint inference has completed and scores were replayed against the
upstream evaluator. End-to-end numeric equivalence to the original training/runtime
environment remains unverified; component checks do not establish it.

## Development probe and final evaluation

```text
python scripts/run_msrvtt_baseline.py --config <run.json> --probe-videos 2 --execute --output-dir <new-probe-output>
```

The probe selects the first N videos from the locked development validation pool,
retains all captions for them and labels results as a subset. It never selects
test videos for development. Preflight then requires only those selected
validation media files. A probe cannot be compared to full test metrics.

```text
python scripts/run_msrvtt_baseline.py --config <fixed-test-run.json> --execute --allow-final-evaluation --output-dir <new-final-output>
```

Final test also requires `selection_fixed_before_scores=true` in the run config.
This is a recorded assertion; the operator must actually fix checkpoint and
preprocessing before accessing scores. Never tune from the returned final errors.
Resource checks run before major allocations and between decode/encoder steps.
CUDA unavailability fails rather than silently falling back to CPU. A resource
exception records failure; it does not establish full benchmark completion.

Successful outputs are `scores.npz` (numeric arrays and Unicode IDs),
`provenance.json` (model/input/source hashes, sample indices/timestamps,
configuration, runtime and memory), `metrics.json` (internal evaluator) and
`run.json`. A successful inference is still marked
`reference_reproduction_verified=false` until reference comparison is performed.
For upstream metric comparison use the score/provenance interface in
[the adapter evaluation guide](msrvtt_adapter_evaluation.md); report ties and
recall units explicitly. An output matrix is not automatically evidence that the
checkpoint's provenance or the paper's quality result has been reproduced.

## Selected jaewooo baseline

The active baseline uses [jaewooo/test](https://huggingface.co/jaewooo/test/tree/de15dc5686a8628d051dae0a85b80d34ed690c68),
revision `de15dc5686a8628d051dae0a85b80d34ed690c68`, epoch4,
SHA256 `3d7e2ba508986c7347380767233b017d4ac4c9fabc135f3457de8625dc833dca`.
This is third-party native CLIP4Clip, not weights released by the paper authors.
Do not substitute OpenMMLab weights into this native pipeline.

The existing local run is retained under `outputs/experiments/msrvtt/20261021_clip4clip_baseline/`:
the frozen notebook, authenticated input bundle, result ZIP and original analysis
remain unchanged. `reference_source/` retains the complete pinned source and license;
`reference_source_manifest.json` records its hashes. Local media is retained in
`data/raw/msrvtt/` and checkpoint/license/provenance in
`weights/clip4clip_native_third_party/`. Supply those paths explicitly in a new
inference configuration. The original input ZIP also contains the public code
and locked metadata needed by the frozen notebook. No Docker image is required.

Inference completed on the locked 1,000-video/1,000-text test pool. Internal
T→V R@1/5/10 = 41.0/69.4/79.7%, V→T = 41.1/69.6/79.5%, MedR=2 both ways.
The original evaluator agrees for T→V; V→T is 40.8103/69.2688/79.1502% because
9 tied queries produce 1,012 rank entries. Keep both evaluator results.
The publisher log reports T→V 41.7/70.5/80.0%; the residual gap is unresolved.
This establishes completed checkpoint inference and score verification, not
paper-level reproduction or a completed training run.

The checkpoint declares training on train9k, which includes internal validation900.
That validation pool can support resource/functionality probes but cannot establish
independent validation quality for this checkpoint or select a learned improvement.
Training history and historical test use have not been independently audited.

Docker packaging and alternative cache/retraining preparations were retired.
`maintenance/cleanup_manifest.json` records removed paths and a compact local
archive of diagnostic evidence; historical progress links into retired folders
are retained as history, not execution instructions.

## Kaggle notebook handoff

`scripts/prepare_kaggle_baseline.py --context <reviewed-context> --output <new-folder>`
creates a local allowlisted input ZIP and notebook. It does not upload or launch a job.
The notebook accepts the authenticated original input bundle as a renamed ZIP or an
expanded Kaggle Dataset; filenames and Dataset titles are not used as identity.
The expected manifest hashes are embedded in the notebook. Current runner/media code
is also embedded, and executed-source hashes record any overlay onto an older bundle.
Ambiguous versions, traversal, altered input and inconsistent existing source fail explicitly.
If Kaggle has recursively expanded the inner BPE GZIP or MSR-VTT metadata ZIP,
the compact notebook downloads original bytes for those two non-code assets from
pinned author URLs, with a 30-second request timeout, at most three network attempts,
byte limits and download progress. Only missing archives (or directories in their
place) can use this recovery; the bytes must still match the authenticated manifest
SHA256. Verified local recovery files are reused on a repeated cell. An existing mismatched file, symlink or
unrecognized missing source remains a failure. No reconstructed ZIP/GZIP or hash
exception is used. The input receipt lists recovered paths. This keeps old Datasets
usable without another upload. Recovery requires Internet. The first executable line
prints `CELL_1_STARTED` before imports; input scanning logs directories/files and has
a separate 15-second heartbeat. A verified Dataset root is pruned from the discovery
walk rather than recursively scanning its expanded inner archives. Missing even the
first print indicates execution has not reached notebook code; kernel scheduling or
front-end submission remain possible causes and are not diagnosed by local tests.

Subprocess output streams to the version console and `baseline_results/console.log`;
`events.jsonl` records stage starts, 30-second silence heartbeats, completion and failures.
Each stage snapshots result files to `baseline_results.zip`, including on ordinary errors.
The inference runner logs each video and text progress, and disables CUDA TF32 for this
explicit FP32 adapter. Separate output directories preserve repeated runs.

Optional `feature_cache_root` stores numeric-only per-input features atomically.
Cache keys bind pool, raw caption or media hash, checkpoint, source, adapter/runner,
PyTorch version, device/GPU, precision and frame batch size. Corrupt cache fails;
changed identities recompute instead of reusing features. Reuse preserves the full
pool and scoring rule and is not a checkpoint/configuration selection mechanism.
Cache persistence requires the working directory to survive or be retained separately.
Kaggle session termination, whole-process OOM or forced kernel kill cannot be made
crash-proof by Python; these can interrupt the latest log/cache/ZIP write.

Local source conformance checks compare native tokenizer IDs, sampled frames and image
transforms on a synthetic MP4, and masked meanP scores on synthetic tensors. They do
not establish equality of checkpoint inference against the original CUDA/precision
environment, verify third-party training history, or reproduce paper metrics.
