from enum import Enum


class ModelSize(str, Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


MODEL_PATHS = {
    ModelSize.NANO: "yolo26n.pt",
    ModelSize.SMALL: "yolo26s.pt",
    ModelSize.MEDIUM: "yolo26m.pt",
    ModelSize.LARGE: "yolo26l.pt",
    ModelSize.XLARGE: "yolo26x.pt",
}


def model_path_for(size: ModelSize) -> str:
    return MODEL_PATHS[size]
