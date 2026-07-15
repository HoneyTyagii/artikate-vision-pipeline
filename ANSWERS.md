# Artikate Studio -Written Answers

---

## Section 1 - Diagnose a Failing CV Pipeline

### Scenario A - Accuracy drops after INT8 quantization (0.91 → 0.58 mAP@0.5)

**What I check first, in order**

1. **Preprocessing parity.** Before blaming quantization, confirm the ONNX path
   feeds the model *exactly* what PyTorch did: same resize/letterbox, same
   normalization (0–1 vs 0–255), same channel order (RGB vs BGR), same layout
   (NCHW). A single mismatch here collapses mAP and is the cheapest thing to rule
   out. Test: dump the input tensor from both paths for one image and compare
   with `np.allclose`.
2. **FP32 ONNX sanity.** Evaluate the *FP32* ONNX (pre-quantization) on the same
   val set. If FP32 ONNX already scores ~0.91, the export is faithful and the
   problem is isolated to quantization. If FP32 ONNX is also low, the bug is in
   export/preprocessing, not INT8.
3. **Calibration data.** Inspect what images were used for INT8 calibration and
   how many. Calibration derives per-tensor activation ranges; a tiny or
   unrepresentative set produces bad scales.

**Three independent root causes that each produce this exact symptom**

- **Bad calibration set.** Too few images, or images not representative of the
  val distribution (wrong lighting/defect mix), so activation ranges clip real
  signal. *Distinguishing test:* re-calibrate with 300–500 representative val-like
  images and re-measure; if mAP recovers, this was it.
- **Per-tensor vs per-channel quantization on sensitive layers.** Detection heads
  and depthwise convs have wide dynamic range; per-tensor INT8 scales crush them.
  *Distinguishing test:* run mixed precision -keep the head/output layers in
  FP16/FP32, quantize only the backbone. If mAP recovers, the head was the
  casualty.
- **Preprocessing / normalization mismatch introduced during the ONNX+quant
  pipeline** (e.g. the quant tool assumes 0–255 uint8 input while the model
  expects normalized float). *Distinguishing test:* the FP32-ONNX eval in step 2
  -if FP32 ONNX is *also* wrong, quantization is a red herring and this is the
  cause.

A fourth, worth naming: an **unsupported/awkwardly-fused op** falling back or
being approximated during quantization. *Test:* compare FP32-ONNX vs INT8-ONNX
outputs layer-by-layer (ORT node output dumps) and find where divergence spikes.

**Fix + validation**

Most commonly the fix is recalibrating with a representative set *and* keeping the
detection head in higher precision (mixed-precision INT8). Validation before the
client's line:

- Re-run the full mAP@0.5 benchmark on the identical val set; accept only if the
  drop vs FP32 is **< 2%**.
- Spot-check a per-image confusion of FP32 vs INT8 predictions to confirm no
  systematic class or size bias was introduced.
- Confirm the calibration set is frozen and version-controlled so the result is
  reproducible on redeploy.

---

### Scenario B -Boxes drift on one camera feed only, offset grows toward edges

**What the pattern tells me.** The error is *systematic* (always shifted, never
random) and *edge-magnified*. That signature points away from the model (which is
identical across all 12 feeds and works on 11) and toward a **geometric transform
applied to that one feed before or after inference**. An error that scales with
distance from the image center is the fingerprint of either (a) a resize/aspect
handling difference -stretch instead of letterbox - or (b) lens-distortion
correction that is applied on the camera/decoder for this feed but not accounted
for when mapping boxes back. Center pixels move little under these transforms;
edge pixels move a lot. That's exactly the observed gradient.

**Specific things I'd check for that one feed**

- **Resolution / aspect ratio.** Is this feed 1280×720 while others are 1920×1080?
  If preprocessing does a plain resize to a square instead of aspect-preserving
  letterbox, boxes stretch outward toward the edges.
- **Letterbox padding un-mapping.** If padding offset/scale isn't reversed with
  this feed's actual dimensions, the mapping drifts most at the edges.
- **Camera-side transforms:** digital zoom, EIS (electronic image stabilization),
  or a different lens/undistortion profile enabled on that RTSP stream only.
- **Rotation/flip or a different crop/ROI** configured on that channel.
- **Pixel aspect / anamorphic encoding** in that feed's SPS (non-square pixels).

**Root-cause hypothesis + confirmation without physical access**

Hypothesis: that feed has a **different native resolution/aspect ratio**, and the
preprocessing letterbox is being computed against an assumed shape, so the
scale/pad reversal is wrong -producing a systematic, edge-amplified offset.

Confirm remotely:

1. Pull raw frame metadata from the stream (`ffprobe` on the RTSP URL): width,
   height, SAR/DAR. Compare against the 11 working feeds.
2. Overlay the model's letterboxed input for one frame of the bad feed vs a good
   feed -if the bad feed is stretched (circles become ellipses), aspect handling
   is confirmed.
3. Feed one saved frame from the bad camera through the *good* feed's exact
   preprocessing path offline; if boxes line up, the bug is per-feed config, not
   the image.

Fix: normalize all feeds through the same aspect-preserving letterbox and reverse
the transform using each feed's *actual* frame dimensions, not a hard-coded shape.

---

### Scenario C -97% → 84% over three months, zero code/model changes

**Plausible causes**

1. **Data / environmental drift.** Lighting changed (seasonal sun angle, a failed
   overhead light, a new fixture), the belt/product finish changed, or the camera
   slowly defocused / accumulated dust. The model is static; the world moved.
2. **Camera hardware degradation or auto-setting drift.** Auto-exposure /
   auto-white-balance gradually shifted, sensor aging, focus creep from vibration,
   or a firmware auto-update changing ISP behavior.
3. **Input-pipeline drift upstream of the model.** A codec/driver update changed
   frame color or compression, or throughput changes caused dropped/duplicated
   frames that skew the count.

**Evidence to confirm / rule out each**

1. Compare a histogram of recent frames vs the deployment-time baseline (mean
   brightness, contrast, per-channel means). A shifted distribution confirms
   environmental/lighting drift. Sampled recent images reviewed by eye will show
   glare, blur, or a changed product.
2. Check EXIF/stream metadata for exposure/gain/WB over time; a monotonic trend
   confirms camera auto-setting drift. A sharpness/focus metric (variance of
   Laplacian) trending down confirms defocus/dust.
3. Diff decoder/driver/firmware versions and timestamps against the accuracy
   inflection point; inspect frame-timing logs for drops/dupes.

**Lightweight monitoring signal that catches it in < 2 weeks**

Log, per hour, two cheap unlabeled signals and alert on drift from a baseline:

- **Prediction-distribution monitor:** rolling detection count per frame, mean
  confidence, and class mix. A silent drop in mean confidence or a shift in count
  distribution flags trouble *before* labeled accuracy is available.
- **Input-image statistics monitor:** rolling mean brightness/contrast + a focus
  (Laplacian variance) metric per camera.

Track both against the first-week baseline with a simple threshold or PSI-style
divergence alert. Either one crossing threshold pages an operator to sample and
review 50 recent frames -turning a 3-month silent drift into a 2-week alert.

---

## Section 3 -Find the Silent Bug

_To be completed once the provided broken repo is received (added in a follow-up commit)._

---

## Section 4 -Edge & Air-Gapped Deployment Design

**Setup:** 8× 1080p @ 15 fps, one Jetson AGX Orin (64GB), fully air-gapped,
end-to-end latency < 200 ms/frame, no cloud ever (including model updates).

### 1. Model family & precision

Target **YOLOv8s (or v8n if recall allows) at INT8** via TensorRT.

Reasoning: the AGX Orin 64GB has strong INT8 throughput on its DLA + GPU. The
< 200 ms budget is *per frame* end-to-end, which is generous per frame but must
hold while 8 streams share the box. INT8 roughly doubles throughput over FP16
with a small, usually recoverable accuracy cost -and headroom matters more than
a fraction of mAP when 8 feeds contend for one GPU. I'd start at v8s-INT8 and drop
to v8n only if aggregate throughput can't be met, or move up to v8m only if
recall on defects is short *and* the throughput math (below) still closes.

### 2. Throughput arithmetic

Aggregate input load:

```
8 cameras × 15 fps = 120 frames/sec  (required inference throughput)
```

Budget per frame if served strictly serially:

```
1000 ms / 120 = 8.33 ms/frame of inference headroom
```

So the detector must sustain **≥ 120 inferences/sec**, i.e. **≤ ~8.3 ms/frame**
of model time, to keep up without a growing queue. The 200 ms end-to-end budget
then covers decode + preprocess + inference + NMS + post, which is comfortable if
model time is ~8 ms and we pipeline the stages.

Reality check against hardware: published Orin numbers put YOLOv8s-INT8/TensorRT
roughly in the low-single-digit to ~10 ms/frame range at 640 depending on batch
and whether DLA is used, so 120 fps aggregate is *plausible* -but I would
**batch across cameras** (e.g. batch of 4–8) to hit it, since batched TensorRT
throughput is much higher than single-image latency implies. I would **not**
commit to a specific fps number without benchmarking on the actual Orin with the
actual model and batch size. This is the main thing I'd want to measure before
promising the client 120 fps: single-stream latency ≠ 8-stream throughput.

Latency-vs-throughput tension to flag honestly: batching raises throughput but
adds queuing latency. A batch of 8 at 15 fps means up to ~53 ms of batch-fill wait
before inference even starts -still inside 200 ms, but it eats the budget, so the
batch size is a tuning knob against the 200 ms ceiling, not a free win.

### 3. Air-gapped retraining loop

- **Flagging:** operators mark false positives/negatives directly on the on-prem
  UI. Each flag saves the frame + model prediction + operator correction to a
  local review store on the server. No internet needed.
- **Feedback out:** on a schedule, export the accumulated flagged samples
  (images + corrected labels) to encrypted removable media (USB) -a manual,
  audited sneakernet transfer to a connected training workstation off-site.
- **Retraining:** fine-tune the current production checkpoint on baseline data +
  the new corrected samples on the workstation. Version the dataset and the run.
- **Validation before replacing production:** the new model must clear a **frozen
  held-out test set** from the client's own floor -accept only if mAP is
  **≥** the incumbent and there is **no regression on the previously-flagged
  failure cases** (they become permanent regression tests). Also re-run the INT8
  export + a shadow latency check so a "better" model that blows the latency
  budget is caught before deployment.
- **Deploy in:** package the validated `.engine`/`.onnx` + metadata, carry it in
  on USB, and stage it alongside the current model.

### 4. Rollback plan + detection speed

- **Keep the previous model version resident on the device** ("blue/green"): the
  new model is deployed beside the known-good one, and switching back is a config
  flip + service restart, not a re-transfer. Rollback in **minutes**.
- **Shadow / canary before full cutover:** run the new model in shadow on live
  feeds (predictions logged, not acted on) for a defined window and compare
  against the incumbent before promoting it.
- **Regression detection:** reuse the Section-1C monitoring signals -rolling mean
  confidence, detection-count distribution, and per-class mix vs the incumbent's
  baseline. A significant divergence after cutover triggers auto-rollback to the
  previous version. With per-hour monitoring this catches a floor-wide regression
  **within hours**, not days.

**Things I'm genuinely unsure about and would benchmark before committing:**
exact YOLOv8s-INT8 batched throughput on *this* Orin, the accuracy cost of INT8 on
this specific defect set (may need per-channel quant or a higher-precision head),
and whether DLA offload helps or hurts once 8 decode pipelines are also running on
the same SoC.
