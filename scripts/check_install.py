"""Check file layout for a selected local ASR backend without loading models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import Pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查本地视频翻译运行文件")
    parser.add_argument("--asr", choices=("whisper", "whisper_precise", "qwen3", "kotoba"),
                        default="whisper_precise", help="要检查的识别引擎")
    args = parser.parse_args()
    problems = Pipeline(ROOT).check_environment(asr_backend=args.asr)
    if problems:
        print(f"{args.asr} 缺少 {len(problems)} 项：")
        for problem in problems:
            print(" -", problem)
        return 1
    print(f"{args.asr} 所需文件已找到。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
