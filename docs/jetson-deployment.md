# Jetson Nano deployment and validation

## Status

This is a preparation runbook, not a hardware validation record. No Jetson
Nano, camera, TensorRT engine, throughput measurement, thermal result, or
display result has been verified for this project yet. The commands below make
the intended target procedure reproducible; record their actual output before
declaring Step 5 complete.

NVIDIA lists JetPack 4.6.6 with L4T 32.7.6 as supporting Jetson Nano. It is the
candidate baseline for this project, subject to confirmation on the specific
board and camera. L4T R32.7.6 is the final JetPack 4 / L4T R32 release, so it
must be treated as a compatibility baseline rather than a claim of current
support. See NVIDIA's [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive)
and [L4T 32.7.6 release page](https://developer.nvidia.com/embedded/linux-tegra-r3276).

Use the official Ultralytics JetPack 4 image as the initial headless
environment: `ultralytics/ultralytics:latest-jetson-jetpack4`. Pin its image
digest after the first successful target validation; `latest` is not a
reproducible deployment identifier. The stock image installs a headless OpenCV
wheel, so it is not the expected environment for an OpenCV display window or
CSI capture through `CAP_GSTREAMER`.

CSI/GStreamer and local display require a separate native or custom-container
environment that preserves Jetson's GUI- and GStreamer-enabled OpenCV build.
That environment remains to be established and verified on the target. Do not
install the project's `desktop` or `headless` extras on either Jetson path:
generic PyPI OpenCV, PyTorch, or TensorRT packages can replace the
Jetson-compatible builds.

## Prepare the target environment

After the repository is available on the Jetson, start an interactive
container. The mount keeps the source on the host, so the following editable
install only adds this project and leaves the image's platform packages intact.

```sh
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4
sudo docker run --rm -it --runtime=nvidia --ipc=host --network host \
    -v "$PWD:/workspace/uav-vision" \
    -w /workspace/uav-vision \
    ultralytics/ultralytics:latest-jetson-jetpack4 bash
python3 -m pip install --no-deps -e .
```

For a USB camera in the headless container, add the applicable host device, for
example `--device=/dev/video0`, to the `docker run` command. Do not use this
stock container command as the CSI or display procedure.

Before capture or export, record these checks in the validation table below.
The OpenCV build information must report GStreamer support before using a CSI
pipeline.

```sh
cat /etc/nv_tegra_release
python3 --version
python3 - <<'PY'
import cv2
import tensorrt as trt
import torch
from ultralytics import YOLO

print("OpenCV:", cv2.__version__)
print("GStreamer enabled:", "GStreamer:                   YES" in cv2.getBuildInformation())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
print("TensorRT:", trt.__version__)
print("Ultralytics model load:", type(YOLO("yolo26n.pt", task="detect")).__name__)
PY
```

If `trtexec` is installed, also record `trtexec --version`; the required
TensorRT check is the Python import above. In the stock headless container,
GStreamer support may report `False`. Before CSI or display validation, repeat
the checks in the separate environment and require GStreamer support plus a
GUI-capable OpenCV build.

## Build the TensorRT engine on the Jetson

TensorRT engines are specific to their target hardware and installed TensorRT
version. Build the nano model on the Jetson that will execute it, rather than
copying an engine made on a laptop or another Jetson:

```sh
yolo export model=yolo26n.pt format=engine device=0
```

Keep the resulting `yolo26n.engine` beside a user configuration file. The
application resolves model identifiers from the YOLO configuration rather than
accepting a model path on the command line. For example, create
`jetson-yolo.json`:

```json
{"detection": {"nano": "yolo26n.engine"}}
```

Then reference it from `jetson-app.json` and select CUDA explicitly:

```json
{
  "inference": {"model_size": "nano", "device": "cuda"},
  "yolo_config_path": "jetson-yolo.json"
}
```

```sh
uav-vision-usb --config-path jetson-app.json
```

Do not record GPU acceleration as successful merely because export completed.
The live inference run and `tegrastats` must show GPU activity; an error must
be investigated rather than replacing the selected device with a CPU fallback.
Ultralytics documents TensorRT export and the target-specific engine constraint
in its [Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) and
[TensorRT integration guide](https://docs.ultralytics.com/integrations/tensorrt/).

## Camera commands

### USB camera

Run headlessly with a USB camera at index `0`:

```sh
uav-vision-usb --config-path jetson-app.json
```

The equivalent command with an explicit OpenCV source is:

```sh
uav-vision --camera-type opencv --camera-index 0 --config-path jetson-app.json
```

### CSI camera through GStreamer

The following is an unverified starting pipeline for a supported CSI camera in
the native or custom environment that preserves Jetson OpenCV with GStreamer.
It is not expected to work in the stock headless container. The pipeline
belongs in user configuration rather than application code because sensor,
resolution, and carrier-board settings vary.

```json
{
  "capture": {
    "camera_type": "gstreamer",
    "source": "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
  },
  "inference": {"device": "cuda"},
  "yolo_config_path": "jetson-yolo.json"
}
```

Save this as `jetson-csi.json` beside `jetson-yolo.json`, then run:

```sh
uav-vision --config-path jetson-csi.json
```

Confirm camera discovery and this pipeline with the Jetson's native tools
first. A failure to open the pipeline is a target configuration issue to
resolve, not a reason to substitute the OpenCV camera adapter.

## Optional display

The processing path is headless by default. In the separate native or custom
environment, on a Jetson local graphical session with a GUI-capable OpenCV
build, add the same display options used on a laptop:

```sh
uav-vision-usb --config-path jetson-app.json \
    --display --display-width 1280 --display-height 720
```

Press `q`, `Q`, or `Esc` in the window to request a clean stop. Validate local
display first. Container X11 forwarding or a remote desktop session requires
host-specific display permissions and has not been validated for this project.

## Measurement and acceptance record

Use representative resolution, camera, model, and power conditions. Start
`tegrastats` in a second terminal before the run, then stop the application with
`Ctrl-C` after the chosen interval so its cleanup path runs:

```sh
tegrastats --interval 1000 --logfile tegrastats.log
```

Run headless inference separately:

```sh
uav-vision-usb --config-path jetson-app.json
```

`tegrastats` supplies GPU utilisation, memory, swap, power, and thermal data; it
does not supply capture rate, model latency, or end-to-end FPS. Before target
validation, add a small measurement harness around the existing
`FrameSource.read()`, `Detector.detect()`, and full processing-loop boundaries.
This will record the three timings without changing normal application output.
That instrumentation remains pending until representative hardware is
available.

Include the exact board, camera, JetPack/L4T, image digest, Python, OpenCV,
PyTorch, TensorRT, Ultralytics, and model versions. Measure optional display
separately from headless mode.

| Acceptance check | Required evidence | Current result |
| --- | --- | --- |
| Target environment | Jetson model, JetPack/L4T, image digest, package versions | Pending — Jetson unavailable |
| USB capture | Live frames from the selected `/dev/video*` device | Pending — Jetson unavailable |
| CSI capture | Native camera check and working GStreamer pipeline | Pending — Jetson unavailable |
| TensorRT inference | Engine built on target, detections, and GPU activity in `tegrastats` | Pending — Jetson unavailable |
| Headless processing | Live capture and inference without a display resource | Pending — Jetson unavailable |
| Optional display | Labels, boxes, requested dimensions, and quit keys work | Pending — Jetson unavailable |
| Shutdown | `Ctrl-C` and display quit release the camera without errors | Pending — Jetson unavailable |
| Performance and thermals | Separate capture/inference harness, end-to-end FPS, memory, and thermal data | Pending — Jetson unavailable; harness not added |
