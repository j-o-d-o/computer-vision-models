from dataclasses import dataclass
from collections import OrderedDict

# These are the same as from the comma10k label spec, colours are in BGR!
SEMSEG_CLASS_MAPPING = OrderedDict([
  ("road", 0x202040),
  ("lane_markings", 0x0000ff),
  ("undriveable", 0x608080),
  ("movable", 0x66ff00),
  ("ego_car", 0xff00cc),
])


@dataclass
class Entry:
  img: bytes
  mask: bytes
  content_type: str # e.g. "image/png", "image/jpg", etc.
  org_source: str # describe original source of the data e.g. KITTI
  org_id: str # describe original identifier

  def get_dict(self):
    return self.__dict__
