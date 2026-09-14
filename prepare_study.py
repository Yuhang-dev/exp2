import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from study_data import prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/study"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = prepare(AutoTokenizer.from_pretrained(args.model), args.out, args.samples, args.seed)
    metadata = {k: v for k, v in data.items() if k != "streams"}
    metadata["streams"] = [{k: v for k, v in stream.items() if k != "tokens"} for stream in data["streams"]]
    (args.out / "streams.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {args.samples} report streams: {args.out / 'streams.pt'}")
