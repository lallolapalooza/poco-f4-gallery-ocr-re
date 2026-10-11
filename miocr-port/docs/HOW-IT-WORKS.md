# How Xiaomi Gallery's local OCR works — and how this port reproduces it

Source of truth: the unrooted adb dump of the Poco F4 (`munch`, Android 14 Global)
described in the parent project. Everything below is from `MIUIGalleryGlobal.apk` and
the code/data it contains; file offsets are into
`product/priv-app/MIUIGalleryGlobal/MIUIGalleryGlobal.apk!lib/arm64-v8a/libmiocr.so`
(10,658,368 B).

## 1. App layer (MIUI Gallery)

* Menu entry `R.id.action_OCR` → string `ocr_entrance` ("Recognize text"); removed from
  the menu unless `OCRHelper.isSupportOCR()`. `munch` is listed in
  `WhiteList.LOCAL_OCR_DEVICES`, so **on this device the local engine is used**.
* `menuitem/Ocr.onClick` → consent (`BaseGalleryPreferences.CTA`) → network check (metered
  networks confirm via `NetworkConsider`) → `OCRManager.startRequestOCR`.
* `OCRHelper.Builder.newInstance()` picks `OCRLocalRequestHelper` (local) or
  `com.miui.gallery.request.OCRRequestHelper` (cloud) depending on
  `DeviceFeature.isSupportLocalOCR()`.
* Local helper: `OCREngine.getInstance().init(cacheDir, null)`, DB cache check
  (`OcrResult` row with `error_code == 5` suppresses re-runs), `OCRDecoder.decode`
  (power-of-two downscale to 3,686,400 px, or 6,000,000 px for aspect > 5 / < 0.2, EXIF
  rotation), then detect + recognize, then `release()`; the text is upserted into the
  `OcrResult` table (`sha1, ocr_result, flag, language="CN", error_code, path`).
* Rendering: `OCRManager` → `GuideView`/`OCREditor`; quad hierarchy
  `Quadrangle → LineQuadrangle → CharQuadrangle`; `GuideView.generateOCRMatrix()`
  scales OCR-bitmap coordinates into `PhotoView.getDisplayRect()`, and every quad point
  is mapped through the PhotoView display matrix. Selection uses
  `CheckPointInside` hit-testing plus per-orientation handle geometry
  (`view/selector/Calculator*.java`).
* Cloud path (not used on `munch`): hardware-attested token
  (`/auth/v1/hardwareNonce` → `signWithDeviceCredential` → `/auth/v1/hardwareAuthToken`)
  then `POST https://api.open.ai.xiaomi.com/vision/ocr/v1/general` with
  `Authorization: HARDWARE_AUTH <token>`, body `{"image": base64}`, WEBP q100, ≤ 4 MiB.
  The response carries per-line boxes, but the app keeps only `line.text`.
* "Translate" is delegated to `com.xiaomi.aiasst.vision` (Mi AI ≥ v301).

## 2. Engine (`libmiocr.so`)

Statically links **MACE v1.1.0-31-g9250bad** (OpenCL kernels, APU/Hexagon paths),
**MNN** (flatbuffer runtime), **OpenCV 3.4.7**, protobuf-lite, FlatBuffers 1.10.0,
ClipperLib, libjpeg-turbo.

JNI surface (`libmiocr_wrapper.so` exports
`Java_com_xiaomi_ocr_sdk_1ocr_OCREngine_*`; the plain C++ symbols live in
`libmiocr.so`): `nativeInit(cacheDir, libDir)`, `nativeOCRLineDetectBitmap`,
`nativeOCRRecognizeBitmap(obj, bitmap, Location[], bool)`, `nativeOCRBitmap`
(one-shot), `nativeOCRPageOrient`, `nativeOCRRegionDetect*`, `nativeWarmup`,
`nativeGetVersion`, `nativeRelease`.

Gallery calls only **line-detect → recognise**; the region/page-orient one-shot APIs
have no callers.

## 3. Models — what is embedded, where

Inside the .so, as MNN flatbuffers with inline weights:

| model | offset | size | input | output |
|---|---|---|---|---|
| recognizer | `0x4539c8` | 3,017,516 B | `(N,1,32,W)` grayscale | `(N,7888,W)` CTC logits |
| orientation | `0x73bdd8` | 926,320 B | `(1,1,48,192)` | `(1,4)` |
| charset | `0x7344f4` | 30,948 B | 7,747 NUL-separated UTF-8 labels | |

Found by validating flatbuffer roots over `.rodata` and confirming each candidate loads
in MNN (`tools/scan_models.py`); extents were pinned by bisecting the smallest buffer
that passes MNN's verifier (`tools/carve_models.py`). The recognizer's end coincides
exactly with the charset start, and the orientation model's end coincides with the
`ocr_detector_model` string at `0x81e048` — a good cross-check.

Recognizer details: model version 2.0.0; output is log-softmax over 7,888 classes;
class→character mapping is in `miocr/labels.py` (derived in §4). The graph's node names
are TF-style (`Softmax177`, `StridedSlice132`, `SliceTf138`, …) — a TensorFlow model
converted to MNN.

Orientation details: the SDK logs `"ocr_orientation predict_orient use mnn;"`; the
carved model takes a 48×192 line strip and scores 4 rotations, matching
`OCRData.RotateFlags{ROTATE_0, ROTATE_90, ROTATE_180, ROTATE_270}`.

## 4. Preprocessing / decoding recovered by experiment

* Input: grayscale, resized to height 32 keeping aspect ratio, `(x/255 - 0.5)/0.5`
  (the graph is insensitive to the scale factor; blank = class 0).
* Decode: greedy CTC — collapse repeats, drop class 0. Per-character x-positions come
  from the timestep span of each emitted class, which is how per-char quads are derived.
* Class mapping: see README ("Label mapping"); recovered by rendering single glyphs and
  comparing the emitted class with the label's index in `charset.txt` — the offset was
  exactly +117 for every probe.

## 5. The detector gap (important)

The line detector is a **separate neural model** (`ocr_detector_model`,
`ocr_detector_model_small`, `ocr_detector_model_large` strings) executed through
**MACE** (`Create MaceEngine from model graph proto and weights data`,
`initMACE … model_name %s, apu_bin_file %s`). It is **not present in the APK**:

* flatbuffer scan of the whole `.so` finds only the two models above;
* parsing `.rodata` with MACE's own `NetDef` schema finds nothing;
* no zlib/gzip/xz streams (i.e. nothing compressed) anywhere in the file;
* the same holds for every other `libmiocr.so` in the dump
  (`MIUIAICR.apk`'s copy, `libmiocr_tokenizer.so`) — they contain the same recognizer +
  orientation + charset, no detector;
* the SDK writes only `/rec_tmp.bin` and `/orient_tmp.bin` to its cache dir — no
  detector cache file, and its loader (`getModelFromFile`, container fields
  `model_num/model name/model type/model idx/model size/model start`) is wired to
  downloaded model packs.

So the detector weights live outside the APK (downloaded to app-private storage, or in
the vendor/NPU path — `/vendor/…` is largely SELinux-denied in this unrooted dump).
`miocr/detect.py` therefore provides a **classical OpenCV line detector** behind the
`Detector` protocol; the rest of the pipeline (recognition, orientation, positions) is
the real thing.
