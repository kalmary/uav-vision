# UAV Vision

UAV Vision captures frames from a camera and detects objects with an
Ultralytics YOLO model. With the optional display enabled, it draws class
labels and bounding boxes on the live image. Without the display, the same
capture and inference pipeline runs headlessly.

The default model is `yolo26n.pt`, the smallest configured detection model.
The first use may download its weights through Ultralytics.

## Laptop and USB camera

The desktop path is intended for Python 3.12 and a camera available as index
`0`. Install the desktop profile; it includes the GUI-capable OpenCV package.

```sh
uv python install 3.12
uv sync --python 3.12 --extra desktop --group test
```

Run a headless capture from camera `0`:

```sh
uv run --extra desktop uav-vision-usb
```

Enable the annotated window and set its dimensions:

```sh
uv run --extra desktop uav-vision-usb \
    --display --display-width 1280 --display-height 720
```

Press `q`, `Q`, or `Esc` while the display window has focus to stop. On macOS,
grant the terminal application permission to use the camera if the operating
system asks. `uav-vision-usb` differs from `uav-vision` only by defaulting to
the OpenCV camera at index `0`; the equivalent explicit command is:

```sh
uv run --extra desktop uav-vision --camera-type opencv --camera-source 0
```

Choose a different built-in model size with `--model-size`:

```sh
uv run --extra desktop uav-vision-usb --model-size small
```

Use `--model-path` for a specific model file. It cannot be combined with
`--model-size`:

```sh
uv run --extra desktop uav-vision-usb --model-path path/to/model.pt
```

TensorRT engines are tied to their target hardware and runtime. Build and use
the Jetson engine according to the separate deployment guide rather than
copying a laptop-built engine.

The current headless mode performs capture and inference without emitting a
per-frame result. Filtered detection logging, including an optional log-file
path, is planned separately.

## Dependency profiles

Select exactly one runtime profile in an environment:

- `desktop` is for a laptop or desktop that needs OpenCV capture and a display
  window: `uv sync --extra desktop --group test`.
- `headless` is for a generic environment without display support:
  `uv sync --extra headless --group test`.
- The profiles are intentionally mutually exclusive because GUI and headless
  OpenCV wheels must not coexist. The Jetson path uses the NVIDIA-provided
  packages instead; see [Jetson deployment](docs/jetson-deployment.md).

## Ultralytics license

Ultralytics YOLO and its models are offered under AGPL-3.0 or an Enterprise
license. Review the [Ultralytics licensing information](https://docs.ultralytics.com/help/contributing/#license)
before distributing this application or a product that includes it.
