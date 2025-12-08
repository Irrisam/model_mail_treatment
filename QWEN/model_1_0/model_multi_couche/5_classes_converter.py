import json
import os

inp = "pi_multi_classes.jsonl"
out = "pi_5_multi_classes.jsonl"

mapping = {
    "Brevets": "Brevets",
    "Marques_DM": "Marques_DM",
    "Contentieux": "Contentieux",
    "Noms de domaine et Surveillance": "Noms de domaine et Surveillance",
    "Extensions": "Extension territoriale"
}

print("nput file:", os.path.exists(inp))

count = 0

with open(inp, "r") as fin, open(out, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj["label"] = mapping[obj["label"]]
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        count += 1

print(f"Total lines processed: {count}")
print(f"Output file: {out}")
