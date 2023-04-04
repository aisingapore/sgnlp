from typing import List


class BaseFeature:
    """This is used as base class for derived StanceClassificationFeature and RumourVerificationFeature."""

    def __init__(self) -> None:
        self.input_ids: List[int] = []
        self.input_masks: List[int] = []
        self.segment_ids: List[int] = []


class StanceClassificationFeature(BaseFeature):
    """Basic feature of a conversation thread for stance classification."""

    def __init__(self) -> None:
        super().__init__()
        self.label_ids: List[int] = []
        self.label_masks: List[int] = []


class RumourVerificationFeature(BaseFeature):
    """Basic feature of a conversation thread for rumour verification."""


class BaseFeatures:
    """This is used as base class for derived StanceClassificationFeatures and RumourVerificationFeatures."""

    def __init__(self) -> None:
        self.nested_input_ids: List[List[int]] = []
        self.nested_input_masks: List[List[int]] = []
        self.nested_segment_ids: List[List[int]] = []
        self.flatten_input_masks: List[int] = []

    def update(self, base_features: BaseFeature) -> None:
        self.nested_input_ids.append(base_features.input_ids)
        self.nested_input_masks.append(base_features.input_masks)
        self.nested_segment_ids.append(base_features.segment_ids)
        self.flatten_input_masks.extend(base_features.input_masks)


class StanceClassificationFeatures(BaseFeatures):
    """Basic features of all conversation threads for stance classification."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten_label_ids: List[int] = []
        self.flatten_label_masks: List[int] = []

    def update(self, base_features: BaseFeature) -> None:
        super().update(base_features)
        if isinstance(base_features, StanceClassificationFeature):
            self.flatten_label_ids.extend(base_features.label_ids)
            self.flatten_label_masks.extend(base_features.label_masks)


class RumourVerificationFeatures(BaseFeatures):
    """Basic features of all conversation threads for rumour verification."""

    def __init__(self) -> None:
        super().__init__()
        self.single_label_id: int
