from enum import Enum
from dataclasses import dataclass


class FileFormat(Enum):
    WSJ = 1
    FILE = 2


@dataclass
class DiscourseTreeNode:
    span: tuple
    ns_type: str
    label: str
    text: str = None


@dataclass
class DiscourseTreeSplit:
    left: DiscourseTreeNode
    right: DiscourseTreeNode
