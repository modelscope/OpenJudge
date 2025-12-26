import json
import sys


def load_jsonl(filepath):
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" not in obj:
                    raise ValueError(f"Missing 'id' field in line {line_num}")
                data[obj["id"]] = obj
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} in {filepath}: {e}")
    return data


def main(file1, file2, output_file):
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    all_ids = set(data1.keys()) | set(data2.keys())
    merged = []

    for id_ in sorted(all_ids):
        if id_ in data1 and id_ in data2:
            merged_obj = {**data1[id_], **data2[id_]}
        elif id_ in data1:
            merged_obj = data1[id_]
        else:
            merged_obj = data2[id_]
        merged.append(merged_obj)

    with open(output_file, "w", encoding="utf-8") as f:
        for obj in merged:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Successfully merged {len(merged)} records into {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python generate_bfcl_tool_call_data.py <tool_call_bfcl_v3_multiple_raw.jsonl> <tool_call_bfcl_v3_multiple_gt.jsonl> <tool_call_bfcl_v3_multiple.jsonl>"
        )
        sys.exit(1)

    file1, file2, output = sys.argv[1], sys.argv[2], sys.argv[3]
    main(file1, file2, output)
