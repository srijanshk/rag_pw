import json

def build_example_map(jsonl_path, output_path):
    id_to_example = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            id_to_example[i] = {
                "problem": obj["problem"],
                "solution": obj["generated_solution"]
            }
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(id_to_example, out)
    print(f"✅ Saved example map to {output_path}")

build_example_map(
    "./data/openmathinstruct2/openmathinstruct2_train_streamed.jsonl",
    "./data/openmathinstruct2/example_id_to_data.json"
)