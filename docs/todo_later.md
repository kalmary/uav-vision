- cli should have arguments: config-path with default value - under it default config is stored and if any argument is not provided, the default is used. if another config path is given, it is used instead as fallback values.
- where are models stored? it would be best to have a separate directory for models, so they are not mixed with other files. (dont code if not necessary)
- in cli: model-path unnecessary - we use only models from ultralytics for now. it is possible that in the future we'll support other model sources. program structure should make the switch easy (ensure it is done now/ do it), but no other models are required.
- in cli: extra yolo arguments are necessary:
  - work mode: classification with bboxes? depthmap? semantic segmentation?
  - if classification + bboxes options for choosing displayed/ returned bboxes should be there. top k elements/ threshold/ specific class or mutliple of it at once. as its getting
  it should be done in a way that for specific yolo modes, relevant arguments are passed - in help we can see what arguments are available for each mode.
- yolo modes to handle:
  - bboxes + classes (default, in both display and headless modes it should show class name+ confidence score)
  - semantic segmentation
  - depthmap
- logger should be implemented. as cli arguments are passed, you should have basic: info about launching configuration given once + filtered output from yolo (usefull when running in headless mode), adjusted for modes:
  - bboxes + classes: bbox, confidence, class name (string)
  - semantic segmentation: num of classes
  - depthmap: min, max, mean, std
- as yolo settings are complicated on its own, it should be moved to a separate config file - if any of arguments for given yolo mode is not provided, the default is used.
- config files with defaults are missing - add them in place easily available for users to modify
- make sure other parts of the code are still compatible with new ones.
- README.md should be provided:
  - what is uav-vision? simply, without non necessary descriptions
  - repo structure
  - how to install it?
  - what are available flags both when installing and running?
  - what is supported?
  - how to use it?

- other questions/ things to modify:
  - what is camera source? i dont think the user needs to tell. if its a laptop its 100% a laptop/ usb camera, detect automatically. if its jetson same situation - no need to multiply args.
- only processing type now is detection. i assume others are ones i asked for above. if so they should be implemented as well. otherwise, all options should be visible
- no cuda device visible error could be handled more elegantly


after 2nd update of work plan:
required working arguments for camera:
- camera type: default opencv (we will change it laaaater)
- camera source: if it is camera index, should be change to camera index (name), default 0

- device should take cuda or cpu strings as input. cpu is default
- no model path is necessary - we just choose which model to use based on processing type and model-size - its enough to download specific model/ use specific cached one
- still some cli args are missing/ are not groupped well - example display height/ width
- make sure input frame size is adjusted ALWAYS for the choosen model (not always the same)
- add to cli fps (int) value with default 30. speed of inference + display + logging is limited to this value. if no --display is set, just inference+ logging is limited. if hardware cannot handle the requested fps, it should be limited to the hardware's maximum. if fps is 0 (manually), fps should be limited to the hardware's maximum.


- frame sequence isnt necessary info and it can be cause of int overflow.
- existing tests should be modified to reflect requested modifications. building completely new tests to reflect changes on existing codebase as "test for change x" is suboptimal.
- if hardware cannot handle the requested fps, it should be limited to the hardware's maximum. if fps is 0 (manually), fps should be limited to the hardware's maximum.
- fps info should be displayed on screen (when --display, and always for logger)
- add licence in readme - this project is under mit, but ultralytics yolo not.
- change install and use guide in readme:
  - divide it into install by uv sync with proper flags (explain what they do)
  - explain usage in later section, with clear distinction between modes (there are 2 entry points)
-   --no-display - this flag is unecessary, if --display is not set, the display is not used (default)
- this could be configured better for help. current format suggests that all of those options work for every yolo mode - most is for detection.
```
processing/inference:
  --processing-type {detection,segmentation,depth}
  --model-size {nano,small,medium,large,xlarge}
  --device {cpu,cuda}
  --selected-classes SELECTED_CLASSES [SELECTED_CLASSES ...]
  --minimum-confidence MINIMUM_CONFIDENCE
  --top-k TOP_K
``` 
