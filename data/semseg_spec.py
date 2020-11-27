from dataclasses import dataclass
from collections import OrderedDict

# These are the same as from the comma10k label spec (https://github.com/commaai/comma10k)
# Hex values are in RGB, Tuples are in BGR!
#  1 - #402020 - road (all parts, anywhere nobody would look at you funny for driving)
#  2 - #ff0000 - lane markings (don't include non lane markings like turn arrows and crosswalks)
#  3 - #808060 - undrivable
#  4 - #00ff66 - movable (vehicles and people/animals)
#  5 - #cc00ff - my car (and anything inside it, including wires, mounts, etc. No reflections)
SEMSEG_CLASS_MAPPING = OrderedDict([
  ("road", (32, 32, 64)), # dark red
  ("lane_markings", (0, 0, 255)), # red
  ("undriveable", (96, 128, 128)), # green-brownish
  ("movable", (102, 255, 0)), # green
  ("ego_car", (255, 0, 204)), # purple
])
SEMSEG_CLASS_IDX = {k: pos for pos, k in enumerate(SEMSEG_CLASS_MAPPING)}

@dataclass
class Entry:
  img: bytes
  mask: bytes
  content_type: str # e.g. "image/png", "image/jpg", etc.
  org_source: str # describe original source of the data e.g. KITTI
  org_id: str # describe original identifier

  def get_dict(self):
    return self.__dict__
