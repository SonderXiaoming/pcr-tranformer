import json
from unit_dict import CHARA_NAME

CHARA_NAME[250] = ["开始"]
CHARA_NAME[520] = ["结束"]
CHARA_NAME[9999] = [0]

small_dict = {i: idx for i, idx in enumerate(list(CHARA_NAME))}

with open("min_unit.json", "w") as f:
    json.dump(small_dict, f)
