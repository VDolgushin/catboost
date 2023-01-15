import os
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file")
    parser.add_argument("--build-root")
    parser.add_argument("--source-root")
    return parser.parse_known_args()


def main():
    args, unknown_args = parse_args()
    inputs = unknown_args
    result_json = {}
    for inp in inputs:
        if os.path.exists(inp) and inp.endswith("tidyjson"):
            with open(inp, 'r') as afile:
                errors = json.load(afile)
            testing_src = errors["file"]
            if os.path.exists(os.path.join(args.source_root, testing_src)):
                # TODO remove .tidyjson concatenation after ya-bin&tt release
                result_json[testing_src + ".tidyjson"] = errors
            elif "_/" not in testing_src:
                # TODO remove .tidyjson concatenation after ya-bin&tt release
                result_json[os.path.basename(testing_src + ".tidyjson")] = errors

    with open(args.output_file, 'w') as afile:
        json.dump(result_json, afile, indent=4)  # TODO remove indent


if __name__ == "__main__":
    main()
