import json
import numpy as np
from collections import defaultdict

if __name__ == "__main__":

    with open("data/test_2020-07-03_12-10-59.json", "r") as f:
        data = json.load(f)

    label = data["label"]
    result_set = data["result_sets"]

    p1, p2 = result_set[0]["players"]
    keys = set(list(p1.keys()) + list(p2.keys()))

    p1_vals, p2_vals = defaultdict(list), defaultdict(list)
    # determine changing values
    for res in result_set:
        p1, p2 = res["players"]

        for k in keys:
            if p1.get(k) is not None:
                p1_vals[k].append(p1.get(k))
            if p2.get(k) is not None:
                p2_vals[k].append(p2.get(k))

    def remove_dups(dic):
        for k, v in dic.items():
            dic[k] = list(set(v))

    remove_dups(p1_vals)
    remove_dups(p2_vals)

    def get_player_name_and_changing_keys(dic):
        name_parts = []
        changing_keys = []
        for k, v in dic.items():
            if len(v) > 1:
                changing_keys.append(k)
            elif k == "name":
                name_parts.insert(0, str(v[0]))
            else:
                name_parts.append(str(v[0]))
        return " ".join(name_parts), changing_keys

    p1_name, p1_changing = get_player_name_and_changing_keys(p1_vals)
    p2_name, p2_changing = get_player_name_and_changing_keys(p2_vals)
    print(label, p1_name, "vs", p2_name)
    print("Comparing p1 {} with p2 {}".format(p1_changing, p2_changing))

    def get_changing_label(player_dict, changing_keys):
        parts = []
        for k in changing_keys:
            parts.append(f"{k}: {str(player_dict[k])}")
        return ", ".join(parts)

    combined_results = defaultdict(list)
    for res in result_set:
        results = res["results"]
        players = res["players"]
        # mean_results = (1 + np.mean(results, axis=0)) / 2

        lbl = tuple(map(lambda x: get_changing_label(*x), zip(players, [p1_changing, p2_changing])))
        combined_results[lbl].extend(results)
        flip_results = [match[::-1] for match in results]
        combined_results[lbl[::-1]].extend(list(flip_results))

    for lbl in sorted(combined_results):
        res = combined_results[lbl]
        mean = (1 + np.mean(res, axis=0)) / 2
        print(lbl, mean, f"Games: {len(res)}")
