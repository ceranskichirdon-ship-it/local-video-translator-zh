"""Offline Qwen3-ASR + official ForcedAligner worker.

Run this module with the separate Python environment containing qwen-asr.
stdout is a UTF-8 JSON-lines event stream; dependency logs go to stderr.
The output JSON uses seconds on the original, uncut audio timeline.

Official API references verified during implementation:
https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py
https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py
https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import unicodedata
import uuid
from typing import Any, Iterable


class WorkerError(RuntimeError):
    """An actionable recognition or alignment failure."""


LANGUAGE_NAMES = {"auto": None, "en": "English", "ja": "Japanese", "ko": "Korean"}
LANGUAGE_CODES = {"english": "en", "japanese": "ja", "korean": "ko", "chinese": "zh",
                  "cantonese": "yue", "french": "fr", "german": "de", "italian": "it",
                  "portuguese": "pt", "russian": "ru", "spanish": "es"}
_SENTENCE_END = re.compile(r"[.!?。！？…](?:[\s\"'”’」』）)]*)$")


def language_code(name: str, requested: str = "auto") -> str:
    """Keep mixed/other language names instead of falsely labelling them English."""
    parts = [p.strip() for p in str(name).split(",") if p.strip()]
    mapped = [LANGUAGE_CODES.get(part.casefold(), part.casefold()) for part in parts]
    if not mapped:
        return requested
    return ",".join(dict.fromkeys(mapped))


def _normal_characters(text: str) -> tuple[str, list[int]]:
    """Match the aligner's letter/number/apostrophe tokens to original text.

    Character positions retain original spacing, punctuation and symbols in the
    subtitles. NFKC and case-folding are used only for matching, never display.
    """
    characters, positions = [], []
    for index, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char).casefold():
            if normalized == "'" or unicodedata.category(normalized)[0] in {"L", "N"}:
                characters.append(normalized)
                positions.append(index)
    return "".join(characters), positions


def _field(item: Any, name: str):
    return item[name] if isinstance(item, dict) else getattr(item, name)


@dataclass
class AlignedUnit:
    start: float
    end: float
    text: str


def project_aligned_text(text: str, items: Iterable[Any],
                         duration: float | None = None) -> list[AlignedUnit]:
    """Attach the complete original transcript to official aligned word spans.

    Exact normalized coverage is required. A tokenization mismatch raises a
    readable error instead of silently deleting words or guessing timestamps.
    Zero-duration words are preserved and merged by ``build_segments``.
    """
    if not str(text).strip():
        raise WorkerError("没有识别到可用语音；请检查视频音量，或指定源语言后重试")
    text = str(text)
    normalized, positions = _normal_characters(text)
    if not normalized:
        raise WorkerError("识别结果没有可对齐的语音文字")
    rows = list(items)
    if not rows:
        raise WorkerError("识别出了文字，但没有得到词级时间戳；请检查本地 ForcedAligner 模型")
    units = []
    original_starts = []
    cursor = 0
    previous_start = -1.0
    for row in rows:
        token = str(_field(row, "text"))
        token_normal, _ = _normal_characters(token)
        if not token_normal:
            continue
        if normalized[cursor:cursor + len(token_normal)] != token_normal:
            raise WorkerError("识别原文与词级对齐结果不一致，为避免漏字已停止生成字幕")
        start, end = float(_field(row, "start_time")), float(_field(row, "end_time"))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise WorkerError("ForcedAligner 返回了无效时间戳")
        if start < previous_start - 0.001:
            raise WorkerError("ForcedAligner 时间戳顺序异常，无法生成可靠字幕")
        if duration is not None and (start > duration + 0.5 or end > duration + 0.5):
            raise WorkerError("词级时间戳超出了原音频范围")
        original_starts.append(positions[cursor])
        units.append(AlignedUnit(start=start, end=end, text=""))
        cursor += len(token_normal)
        previous_start = start
    if cursor != len(normalized):
        raise WorkerError("词级对齐未覆盖全部识别文字，为避免遗漏已停止生成字幕")
    if not units:
        raise WorkerError("没有获得有效的语音时间戳")
    # Boundaries are each next token's original start. Thus punctuation after a
    # word stays with that word, Korean spaces are untouched, and no CJK spaces
    # are invented. Prefix symbols belong to the first token.
    for index, unit in enumerate(units):
        left = 0 if index == 0 else original_starts[index]
        right = original_starts[index + 1] if index + 1 < len(units) else len(text)
        unit.text = text[left:right]
    return units


def build_segments(text: str, items: Iterable[Any], language: str = "auto", *,
                   max_duration: float = 6.0, max_chars: int = 72,
                   pause_seconds: float = 0.65,
                   duration: float | None = None) -> list[dict]:
    """Group exact aligned words into readable, fully covered subtitle cues."""
    if max_duration <= 0 or max_chars <= 0 or pause_seconds < 0:
        raise ValueError("Invalid subtitle segmentation limits")
    units = project_aligned_text(text, items, duration=duration)
    groups: list[list[AlignedUnit]] = []
    current: list[AlignedUnit] = []
    current_size = 0
    for unit in units:
        if current:
            first, last = current[0], current[-1]
            valid_span = max(part.end for part in current) > first.start
            sentence_end = bool(_SENTENCE_END.search(last.text.rstrip()))
            gap = unit.start - last.end
            too_long = unit.end - first.start > max_duration
            too_much_text = current_size + len(unit.text) > max_chars
            if valid_span and (sentence_end or gap >= pause_seconds or too_long or too_much_text):
                groups.append(current)
                current = []
                current_size = 0
        current.append(unit)
        current_size += len(unit.text)
    if current:
        if groups and max(unit.end for unit in current) <= current[0].start:
            # The model can assign 0 ms to a trailing token; keep its text in the
            # preceding cue rather than deleting it or inventing a timed cue.
            groups[-1].extend(current)
        else:
            groups.append(current)
    result = []
    for group in groups:
        start = group[0].start
        end = max(unit.end for unit in group)
        if duration is not None:
            end = min(end, duration)
        if end <= start:
            raise WorkerError("识别文字没有有效时长，无法生成可靠字幕")
        display = "".join(unit.text for unit in group).strip()
        if not display:
            continue
        result.append({"id": len(result) + 1, "start": round(start, 3),
                       "end": round(end, 3), "text": display})
    if not result:
        raise WorkerError("没有可用字幕片段")
    # Whitespace between cues is formatting, but every non-whitespace character
    # (including punctuation and emoji) must survive segmentation exactly once.
    remove_space = lambda s: "".join(c for c in s if not c.isspace())
    if remove_space("".join(segment["text"] for segment in result)) != remove_space(text):
        raise WorkerError("字幕完整性检查失败，识别原文未被完整保留")
    # Brief aligned words can last only a few frames. Extend their display into
    # the following silence without moving starts, changing text or word spans,
    # shortening an existing cue, or crossing the next cue/audio boundary.
    for index, segment in enumerate(result):
        if segment["end"] - segment["start"] >= 0.9:
            continue
        limit = result[index + 1]["start"] if index + 1 < len(result) else math.inf
        if duration is not None:
            limit = min(limit, duration)
        display_end = min(round(segment["start"] + 0.9, 3), limit)
        segment["end"] = max(segment["end"], display_end)
    return result


def configure_offline(root: Path):
    cache = root / "work" / "qwen-cache"
    values = {"HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "huggingface" / "hub",
              "TORCH_HOME": cache / "torch", "NUMBA_CACHE_DIR": cache / "numba",
              "CUDA_CACHE_PATH": root / "work" / "cuda-cache",
              "TMP": root / "work" / "qwen-temp", "TEMP": root / "work" / "qwen-temp"}
    for key, folder in values.items():
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false")


def _publish_json(path: Path, data: dict):
    if path.exists():
        raise WorkerError(f"识别输出文件已经存在，为保护已有结果已停止：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("asr_result_" + uuid.uuid4().hex + ".json.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if os.name == "nt":
            temporary.rename(path)  # Windows rename refuses an existing destination.
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args, emit):
    root = Path(__file__).resolve().parent
    audio_path = Path(args.audio).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    aligner_path = Path(args.aligner).expanduser().resolve()
    if not audio_path.is_file():
        raise WorkerError(f"找不到本地音频文件：{audio_path}")
    if output_path == audio_path or output_path.exists():
        raise WorkerError("识别输出不能覆盖音频或已有文件，请使用新的 JSON 输出路径")
    for path, label in [(model_path, "Qwen3-ASR"), (aligner_path, "Qwen3-ForcedAligner")]:
        if not path.is_dir() or not (path / "config.json").is_file():
            raise WorkerError(f"找不到完整的本地 {label} 模型目录：{path}")
    configure_offline(root)
    started = time.monotonic()
    emit("progress", "正在加载本地 Qwen3 语音识别和时间对齐模型", 2)
    # Imports remain local so the app and unit tests need no PyTorch dependencies.
    import torch
    import soundfile as sf
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import MAX_FORCE_ALIGN_INPUT_SECONDS

    audio_info = sf.info(str(audio_path))
    duration = audio_info.frames / audio_info.samplerate
    if duration <= 0:
        raise WorkerError("音频文件没有有效时长")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_name = torch.cuda.get_device_name(0)
    else:
        dtype, device_name = torch.float32, "CPU"
        emit("log", "当前未检测到可用显卡，正在使用 CPU 识别，速度会较慢")
    emit("log", f"本地语音识别设备：{device_name}；音频 {duration:.1f} 秒")
    kwargs = dict(dtype=dtype, device_map=device, attn_implementation="sdpa", local_files_only=True)
    model = Qwen3ASRModel.from_pretrained(
        str(model_path), max_inference_batch_size=1, max_new_tokens=4096,
        forced_aligner=str(aligner_path), forced_aligner_kwargs=dict(kwargs), **kwargs)
    emit("progress", "正在完整识别音频并进行词级时间对齐", 15)
    emit("log", f"长音频由官方识别库自动分段（每段最多 {MAX_FORCE_ALIGN_INPUT_SECONDS} 秒），保留原始时间坐标")
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(15):
            emit("log", f"正在识别和对齐，已用时 {int(time.monotonic() - started)} 秒")

    pulse = threading.Thread(target=heartbeat, daemon=True, name="qwen-asr-progress")
    pulse.start()
    try:
        with torch.inference_mode():
            results = model.transcribe(audio=str(audio_path), language=LANGUAGE_NAMES[args.language],
                                       return_time_stamps=True)
    finally:
        stopped.set()
        pulse.join(timeout=1)
    if len(results) != 1:
        raise WorkerError("本地识别返回了异常数量的结果")
    recognition = results[0]
    language_name = str(recognition.language or "")
    language = language_code(language_name, args.language)
    text = str(recognition.text or "")
    timestamps = recognition.time_stamps
    items = list(timestamps.items) if timestamps is not None else []
    emit("progress", "正在按原始时间戳整理完整字幕", 92)
    segments = build_segments(text, items, language, duration=duration)
    words = [{"text": str(_field(item, "text")), "start": float(_field(item, "start_time")),
              "end": float(_field(item, "end_time"))} for item in items]
    result = {"language": language, "language_name": language_name, "text": text,
              "segments": segments, "backend": "qwen3-asr", "alignment": "qwen3-forced-aligner",
              "model": str(model_path), "aligner_model": str(aligner_path),
              "audio": str(audio_path), "duration": duration, "word_timestamps": words,
              "device": device_name, "dtype": str(dtype), "vad_used": False,
              "official_max_chunk_seconds": MAX_FORCE_ALIGN_INPUT_SECONDS,
              "elapsed_seconds": round(time.monotonic() - started, 3), "schema_version": 1}
    _publish_json(output_path, result)
    emit("progress", f"识别完成：{len(segments)} 个字幕片段，语言 {language_name or language}", 100)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地 Qwen3-ASR 与词级时间对齐工作程序")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--aligner", required=True)
    parser.add_argument("--language", choices=tuple(LANGUAGE_NAMES), default="auto")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    event_stream = sys.stdout
    lock = threading.Lock()

    def emit(kind, message, progress=None):
        event = {"type": kind, "stage": "transcribe", "message": message}
        if progress is not None:
            event["progress"] = progress
        with lock:
            event_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            event_stream.flush()

    try:
        # Prevent progress bars or dependency print() calls from corrupting the
        # stdout JSON protocol while retaining their details in the task log.
        with contextlib.redirect_stdout(sys.stderr):
            run(args, emit)
        return 0
    except KeyboardInterrupt:
        emit("error", "已取消语音识别")
        return 130
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        message = str(exc)
        if "out of memory" in message.lower():
            message = "显存不足，请关闭其他使用显卡的程序后重试。" + message
        emit("error", message or type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
