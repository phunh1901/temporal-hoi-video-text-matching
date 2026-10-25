# Frozen global-mean video–text pilot

## Purpose

`scripts/run_pilot_baseline.py` provides a runnable B0 baseline on the small,
reviewed local pilot. It validates the existing manifest lock and source-file
hashes before loading a local OpenCLIP checkpoint. It downloads no model or
media, uses no cloud service, and does not change annotations or splits.

This tests the complete decoding → image/text embedding → pooling → retrieval
path. It does not reproduce VidHOI or MSR-VTT benchmark scores and cannot assess
rule violations. The pilot contains no HOI boxes, interaction labels, reliable
event boundaries or independently reviewed violation ground truth.

## Locked inputs and scoring

- `configs/pilot_video_loading.json`: 8 clips, 6 dev clips and 2 extension clips;
  8 frames per clip; decoder maximum image side 320; batch size 1.
- `outputs/experiments/pilot/data_review/pilot_manifest_lock.json`: exact hashes for clip, text, split,
  relevance and source manifests. Inputs must match; changed manifests fail.
- Five original source videos total 23,732,142 bytes, below the 50 MiB limit.
  Actual video content hashes and sizes are checked against `sources.csv`.
- The complete 8 × 10 candidate pool has 80 explicit binary annotations and
  16 positive pairs. Positive sets come from `relevance.csv`, including
  multiple videos relevant to the same text. Missing pairs fail rather than
  becoming implicit negatives.
- Frame selection uniformly samples eight distinct frame indices across each
  full locked `[start_s,end_s)` interval. The reader requires constant frame
  rate and checks decoder timestamps; it never repeats or substitutes frames.
- Video frames convert to RGB and, when needed, downsample with OpenCV area
  interpolation to side 320. OpenCLIP then uses bicubic shortest-side resize
  to 224, center crop to 224 and the explicit CLIP RGB mean/std.
- The frozen `ViT-B-32` encodes each image in FP32. Normalize each frame's
  embedding, average all eight valid embeddings, then normalize the mean.
  Encode texts with the ViT-B-32 tokenizer (context length 77) and normalize
  each text embedding. Zero norms or nonfinite values fail explicitly.
- Score is the cosine dot product. Candidate order follows the locked CSV;
  equal scores retain that order. Multiplying by the model's positive
  `exp(logit_scale)` would leave ranks unchanged; this runner stores cosine.

The verified cached LAION checkpoint has SHA256
`ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6`
and size 605,143,316 bytes. It is not a CLIP4Clip checkpoint fine-tuned on
MSR-VTT. The runner loads the state dictionary strictly from the supplied local
path. An alternate explicit hash identifies a custom checkpoint rather than
claiming the known LAION provenance.

## Reproduction

From the repository root, pass the local checkpoint path explicitly:

```powershell
& '.\.venv\Scripts\python.exe' scripts/run_pilot_baseline.py `
  --checkpoint 'C:\path\to\open_clip_model.safetensors' `
  --threads 2 --output-dir outputs/experiments/pilot/baseline/b0
```

The default expected checkpoint hash is the verified hash above. A new clone
must first obtain and review its own pilot inputs; raw videos, locked local
manifests and run outputs are deliberately not distributed with the code.
The runner refuses absent files or changed hashes and does not fetch them.

CPU execution uses one image/text per batch, two computation threads by
default, one interop thread and no data-loader workers. Hugging Face and
Transformers offline modes are set before model loading. The Windows memory
guard requires available commit capacity greater than twice the checkpoint
size plus 512 MiB before import/loading, and at least 256 MiB between clips.
This is a local safety policy, not a published minimum hardware requirement.

## Output contract

`baseline_run.json` records time, exact command arguments, input and checkpoint
hashes, model/library versions, preprocessing, coverage, both retrieval
directions, complete similarity scores, per-clip decoder timestamps, elapsed
time, memory measurements and limitations. Failed runs save an explicit error
record; they do not claim completed checkpoint inference.

`embeddings.npz` contains numeric FP32 video, text and frame embeddings, scores,
timestamps and ID arrays; read it with `allow_pickle=False`. `predictions.json`
contains the full ranked candidate lists and explicit positive labels in both
directions. Their hashes are included in the run record.

Metrics report all eight pilot clips and dev/extension separately. Video-to-text
uses all ten text candidates in each scope. Text-to-video uses only the videos
in that scope and excludes a query if it has no positive video there, reporting
its excluded count. R@1/5/10 are fractions; MedR is one-based and MRR is auxiliary.
The all-pilot query counts are 8 and 10, not 80 independent observations.
R@10 saturates by construction for these tiny candidate pools.

The outputs can support later temporal-model comparison if preprocessing,
candidate pools, inputs and checkpoint remain fixed. The frame embeddings are
sampled over complete pilot clips; they are not a ready-made sliding-window
cache and must not be relabeled as independent validation data.

## Observed execution and interpretation

The local run on 2026-10-03 completed all 64 frames and all ten texts with finite
8 × 10 scores. All-pilot video-to-text R@1 was 7/8 (87.5%), MRR 0.9375 and MedR 1;
text-to-video R@1 was 10/10 (100%), MRR 1 and MedR 1. Dev video-to-text R@1 was
6/6; extension was 1/2. The pairwise metric was 121/128 (94.53125%).

The cycling clip's highest-scoring description concerned walking across a
parking area; its first cycling-positive ranked second. This shows that global
scene pooling can confuse a shared outdoor context with the action. It is a
diagnostic observation on the pilot, not an instruction to tune against an
official test set. All labels remain unchanged.

Model loading took 3.02 seconds; decoding plus image/text inference, output
serialization and metric calculation took 7.73 seconds; total time including
imports and hashing was 18.55 seconds. Peak process working set was 918,179,840
bytes (875.64 MiB), and peak process commit charge was 2,818,932,736 bytes
(2.63 GiB). Commit charge is not physical RAM usage. The system's sampled
physical RAM minimum fell to 8,327,168 bytes (7.94 MiB), while available commit
remained at least 1,713,659,904 bytes (1.60 GiB). The process finished successfully
under paging pressure; these measurements do not establish that a full
benchmark or model training will fit on this machine. OS peak counters capture
the process maxima; 0.5-second sampling may miss shorter system-memory dips.

The 8 clips were selected for a technical pilot, contain related source footage
and coarse scene descriptions, and were reviewed by AI rather than a human.
Scores cannot establish generalization, SOTA performance, or real violation
detection. No numerical delta against CLIP4Clip/ST-HOI is defined because the
dataset, model checkpoint and protocol differ.
