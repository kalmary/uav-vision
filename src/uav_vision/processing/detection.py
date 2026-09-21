from uav_vision.domain import Frame, ProcessedFrame
from uav_vision.inference import Detector


class DetectionProcessor:
    def __init__(self, detector: Detector) -> None:
        self._detector = detector

    def process(self, frame: Frame) -> ProcessedFrame:
        return ProcessedFrame(frame=frame, detections=self._detector.detect(frame))
