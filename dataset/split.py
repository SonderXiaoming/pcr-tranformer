import itertools
import json
import random

orginal_train = {}
orginal_test = {}
bigger_train = {}
bigger_test = {}

with open("./data_all.json", "r") as f:
    data: dict = json.load(f)
    data_list = list(data)
    length = len(data)
    test_list = random.sample(range(length), int(length * 0.3))
    for test_id in test_list:
        test_def = data_list[test_id]
        orginal_test[test_def] = data[test_def]
        del data[test_def]
    orginal_train = data

for def_team in orginal_train:
    for perm in itertools.permutations(def_team.split(",")):
        bigger_train[",".join(perm)] = orginal_train[def_team]

for def_team in orginal_test:
    for perm in itertools.permutations(def_team.split(",")):
        bigger_test[",".join(perm)] = orginal_test[def_team]

with open("orginal_train.json", "w") as f:
    json.dump(orginal_train, f)

with open("orginal_test.json", "w") as f:
    json.dump(orginal_test, f)

with open("bigger_train", "w") as f:
    json.dump(bigger_train, f)

with open("bigger_test.json", "w") as f:
    json.dump(bigger_test, f)
