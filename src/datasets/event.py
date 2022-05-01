from dataclasses import dataclass


@dataclass
class Event:
    label: str
    label_idx: int
    start: float
    end: float
