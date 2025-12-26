import argparse
import json
import re
import sys


def extract_index(id_str) -> int:
    match = re.search(r"(\d+)$", id_str)
    return int(match.group(1)) if match else 0


def resolve_ground_truth_args(gt_func_dict):
    resolved = {}
    for key, values in gt_func_dict.items():
        for v in values:
            if v != "" or isinstance(v, (int, float)):
                resolved[key] = v
                break
        else:
            resolved[key] = ""
    return resolved


def convert_to_output_format(obj: dict, min_expect_score=None, max_expect_score=None) -> dict:
    index = extract_index(obj["id"])

    user_msgs = obj["question"][0]
    query = [{"message": msg} for msg in user_msgs]

    tool_definitions = []
    for func in obj["function"]:
        tool_definitions.append(
            {
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func["parameters"],
                },
            }
        )

    tool_calls = []
    for gt in obj["ground_truth"]:
        func_name = list(gt.keys())[0]
        args_dict = resolve_ground_truth_args(gt[func_name])
        tool_calls.append(
            {
                "function": {"name": func_name, "arguments": json.dumps(args_dict, ensure_ascii=False)},
                "id": f"call_{func_name}_{abs(hash(json.dumps(args_dict, sort_keys=True))) % 1000000:06x}",
            }
        )

    data = {
        "index": index,
        "parameters": {"query": query, "tool_calls": tool_calls, "tool_definitions": tool_definitions},
    }
    if min_expect_score is not None:
        data["min_expect_score"] = min_expect_score
    if max_expect_score is not None:
        data["max_expect_score"] = max_expect_score
    return data


def main(input_path, output_path, min_expect_score, max_expect_score):
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out_obj = convert_to_output_format(
                    obj, min_expect_score=min_expect_score, max_expect_score=max_expect_score
                )
                records.append(out_obj)
            except Exception as e:
                print(f"Warning: Skip invalid line {line_num}: {e}", file=sys.stderr)

    records.sort(key=lambda x: x["index"])
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Successfully converted and sorted {len(records)} records → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BFCL dataset.")
    parser.add_argument(
        "--input_path",
        type=str,
        help="Input file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "--min_expect_score",
        type=int,
        default=None,
        help="Set minimum expected score",
    )
    parser.add_argument(
        "--max_expect_score",
        type=int,
        default=None,
        help="Set maximum expected score",
    )
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        min_expect_score=args.min_expect_score,
        max_expect_score=args.max_expect_score,
    )
