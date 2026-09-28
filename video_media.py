"""Local video, speech transcription and subtitle tools (standard library only).

Events passed to ``emit`` are dictionaries with type, stage, message and an
optional progress percentage. Segment timestamps are seconds. Existing output
files are never replaced; callers should use a fresh per-job output directory.
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Iterable
import uuid


class MediaError(RuntimeError):
    """A local media tool could not finish the requested operation."""


class CancelledError(MediaError):
    """The operation was cancelled by the user."""


def srt_timestamp(seconds: float) -> str:
    """Format an absolute time, rounding to the nearest millisecond."""
    value = float(seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError("字幕时间必须是非负有限数值")
    total = int((Decimal(str(seconds)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    sec, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{sec:02},{millis:03}"


def _clean_text(value: Any) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ord(ch) >= 32)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def write_srt(segments: Iterable[dict], path: str | Path, text_key: str = "text") -> Path:
    """Write UTF-8 SRT without changing original timestamps or replacing files."""
    rows = []
    for segment in segments:
        text = _clean_text(segment.get(text_key, ""))
        if not text:
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("字幕片段结束时间必须大于开始时间，且均为非负有限数值")
        start_label, end_label = srt_timestamp(start), srt_timestamp(end)
        if start_label == end_label:
            end_label = srt_timestamp(start + 0.001)
        rows.append(f"{len(rows) + 1}\n{start_label} --> {end_label}\n{text}\n\n")
    if not rows:
        raise MediaError("没有可保存的字幕内容")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create makes overwriting even a concurrently-created file impossible.
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(rows))
    return target


def _timestamp_seconds(value: str) -> float:
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})", str(value))
    if not match:
        raise ValueError(f"无法识别时间戳：{value}")
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse_whisper_json(data: dict) -> tuple[list[dict], str]:
    """Parse whisper.cpp -oj JSON; offsets are milliseconds, not centiseconds."""
    entries = data.get("transcription")
    if not isinstance(entries, list):
        raise MediaError("语音识别返回的数据中没有 transcription 字幕列表")
    segments = []
    silent_labels = {"[blank_audio]", "[silence]", "(silence)", "[no_speech]"}
    for entry in entries:
        text = _clean_text(entry.get("text", ""))
        if not text or text.casefold() in silent_labels:
            continue
        try:
            if "offsets" in entry:
                start = float(entry["offsets"]["from"]) / 1000.0
                end = float(entry["offsets"]["to"]) / 1000.0
            else:
                start = _timestamp_seconds(entry["timestamps"]["from"])
                end = _timestamp_seconds(entry["timestamps"]["to"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                times = entry.get('timestamps',{})
                start = _timestamp_seconds(times.get('from',''))
                end = _timestamp_seconds(times.get('to',''))
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    raise ValueError('有识别文字但时间戳无效，不能忽略此片段')
        except (KeyError, TypeError, ValueError) as exc:
            raise MediaError("语音识别返回了无效时间戳") from exc
        segments.append({"id": len(segments) + 1, "start": start, "end": end, "text": text})
    if not segments:
        raise MediaError("没有识别到可用的语音；请检查视频是否有说话声音，或手动指定源语言后重试")
    return segments, str(data.get("result", {}).get("language") or data.get("params", {}).get("language") or "auto")


def _same_file(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    try:
        return first.exists() and second.exists() and os.path.samefile(first, second)
    except OSError:
        return False


def _publish(temp_path: Path, target: Path) -> None:
    """Publish a completed result atomically, with no overwrite race."""
    try:
        os.link(temp_path, target)
    except FileExistsError:
        raise MediaError(f"输出文件已经存在，为保护原文件已停止：{target}") from None
    except OSError:
        # Windows rename is also no-clobber. Filesystems without hard links (exFAT)
        # still work, while POSIX uses exclusive create followed by a streamed copy.
        if os.name == "nt":
            try:
                temp_path.rename(target)
            except FileExistsError:
                raise MediaError(f"输出文件已经存在，为保护原文件已停止：{target}") from None
            return
        with temp_path.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
    temp_path.unlink(missing_ok=True)


class MediaTools:
    def __init__(self, config: dict, emit: Callable[[dict], None] | None = None,
                 cancel_event: threading.Event | None = None):
        self.config = dict(config)
        self.base_dir = Path(config.get("base_dir", Path(__file__).parent)).resolve()
        self.work_dir = self._path(config.get("work_dir", "work"))
        self.ffmpeg = self._path(config.get("ffmpeg_path", "tools/ffmpeg/ffmpeg.exe"))
        self.ffprobe = self._path(config.get("ffprobe_path", self.ffmpeg.with_name("ffprobe.exe")))
        self.whisper = self._path(config.get("whisper_path", "tools/whisper/whisper-cli.exe"))
        self.emit = emit or (lambda event: None)
        self.cancel_event = cancel_event or threading.Event()

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.base_dir / path).resolve()

    def _event(self, kind: str, stage: str, message: str, progress: float | None = None):
        event = {"type": kind, "stage": stage, "message": message}
        if progress is not None:
            event["progress"] = min(100.0, max(0.0, float(progress)))
        self.emit(event)

    def _check_cancelled(self):
        if self.cancel_event.is_set():
            raise CancelledError("已取消处理")

    @staticmethod
    def _require_file(path: Path, label: str):
        if not path.is_file():
            raise MediaError(f"找不到{label}：{path}")

    @staticmethod
    def _stop_process(process: subprocess.Popen):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _run(self, args: list[str | Path], stage: str, *, cwd: Path | None = None,
             duration: float = 0, capture: bool = False, timeout: float | None = None) -> str:
        self._check_cancelled()
        self._require_file(Path(args[0]), "处理程序")
        output = queue.Queue()
        history = [] if capture else deque(maxlen=80)
        kwargs = {"cwd": str(cwd or self.base_dir), "stdin": subprocess.DEVNULL,
                  "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                  "text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1}
        child_env = os.environ.copy()
        cuda_cache = self.base_dir / "work" / "cuda-cache"
        cuda_cache.mkdir(parents=True, exist_ok=True)
        child_env["CUDA_CACHE_PATH"] = str(cuda_cache)
        kwargs["env"] = child_env
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            process = subprocess.Popen([str(arg) for arg in args], **kwargs)
        except OSError as exc:
            raise MediaError(f"无法启动{Path(args[0]).name}：{exc}") from exc

        def reader():
            try:
                for line in process.stdout:
                    output.put(line.rstrip("\r\n"))
            finally:
                output.put(None)

        reader_thread = threading.Thread(target=reader, name=f"media-{stage}-reader", daemon=True)
        reader_thread.start()
        started = time.monotonic()
        last_event = started
        last_progress = -1.0
        ended = False
        try:
            while not ended:
                self._check_cancelled()
                now = time.monotonic()
                if timeout and now - started > timeout:
                    raise MediaError("读取媒体信息超时，请确认文件所在磁盘可以访问")
                try:
                    line = output.get(timeout=0.2)
                except queue.Empty:
                    if now - last_event >= 15:
                        self._event("log", stage, "仍在处理，请稍候……")
                        last_event = now
                    continue
                if line is None:
                    ended = True
                    continue
                history.append(line)
                if line.startswith('{'):
                    try:
                        event = json.loads(line)
                        if event.get('type') in ('log','progress','error') and 'message' in event:
                            event.setdefault('stage',stage)
                            self.emit(event)
                            continue
                    except (ValueError,TypeError,AttributeError):
                        pass
                progress = None
                if line.startswith("out_time_us=") and duration > 0:
                    try:
                        progress = int(line.partition("=")[2]) / 1_000_000 / duration * 100
                    except ValueError:
                        pass
                match = re.search(r"progress\s*=\s*(\d+(?:\.\d+)?)%", line)
                if match:
                    progress = float(match.group(1))
                if capture:
                    # Callers that capture full tool output report progress at
                    # their own window/group granularity.
                    progress = None
                if progress is not None and (progress >= last_progress + 0.5 or progress >= 100):
                    last_progress = progress
                    self._event("progress", stage, "正在处理", progress)
                    last_event = now
                elif not capture and line and not re.match(r"^[a-zA-Z_\d]+\s*=", line):
                    self._event("log", stage, line)
                    last_event = now
            return_code = process.wait()
            self._check_cancelled()
            if return_code:
                tail = "\n".join(list(history)[-16:])[-5000:]
                raise MediaError(f"{Path(args[0]).name} 处理失败（退出码 {return_code}）\n{tail}")
            return "\n".join(history)
        finally:
            self._stop_process(process)
            reader_thread.join(timeout=2)
            process.stdout.close()

    def probe(self, path: str | Path) -> dict:
        source = Path(path).resolve()
        self._require_file(source, "视频文件")
        raw = self._run([self.ffprobe, "-v", "error", "-show_format", "-show_streams",
                         "-of", "json", source], "probe", capture=True, timeout=60)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MediaError("无法读取该媒体文件的信息") from exc
        streams = result.get("streams", [])
        videos = [stream for stream in streams if stream.get("codec_type") == "video"
                  and not stream.get("disposition", {}).get("attached_pic")]
        audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
        def number(value):
            try:
                n = float(value)
                return n if math.isfinite(n) and n > 0 else 0.0
            except (TypeError, ValueError):
                return 0.0
        duration = number(result.get("format", {}).get("duration"))
        if not duration:
            duration = max([number(stream.get("duration")) for stream in streams] or [0])
        result.update(duration=duration, has_audio=bool(audios), has_video=bool(videos),
                      width=videos[0].get("width", 0) if videos else 0,
                      height=videos[0].get("height", 0) if videos else 0)
        return result

    def extract_audio(self, input: str | Path, wav: str | Path) -> Path:
        source, target = Path(input).resolve(), Path(wav).resolve()
        if _same_file(source, target) or target.exists():
            raise MediaError(f"输出文件已经存在，为保护原文件已停止：{target}")
        info = self.probe(source)
        if not info["has_audio"]:
            raise MediaError("所选视频没有音轨，无法进行语音识别")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_name(f"audio_{uuid.uuid4().hex}.wav")
        try:
            self._event("progress", "extract", "正在提取单声道 16 kHz 音频", 0)
            self._run([self.ffmpeg, "-hide_banner", "-nostdin", "-n", "-i", source,
                       "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                       "-c:a", "pcm_s16le", "-progress", "pipe:1", "-nostats", temp_path],
                      "extract", duration=info["duration"])
            self._check_cancelled()
            if not temp_path.is_file() or temp_path.stat().st_size <= 44:
                raise MediaError("音频提取结果为空")
            _publish(temp_path, target)
            self._event("progress", "extract", "音频提取完成", 100)
            return target
        finally:
            temp_path.unlink(missing_ok=True)

    def transcribe(self, wav: str | Path, language: str = "auto") -> tuple[list[dict], str]:
        backend = self.config.get('asr_backend','whisper')
        if backend == 'whisper_precise':
            from precise_asr import PreciseWhisper
            return PreciseWhisper(self).transcribe(Path(wav).resolve(), language)
        if backend == 'qwen3':
            return self._transcribe_qwen(wav,language)
        if backend == 'kotoba':
            return self._transcribe_kotoba(wav,language)
        source = Path(wav).resolve()
        self._require_file(source, "音频文件")
        if language not in {"auto", "en", "ja", "ko", "zh"}:
            raise MediaError("不支持的识别语言，请选择自动、英语、日语或韩语")
        model_value = self.config.get("whisper_model") or self.config.get("whisper_model_path")
        if not model_value:
            raise MediaError("尚未配置本地 Whisper 语音识别模型")
        model = self._path(model_value)
        self._require_file(model, "Whisper 模型")
        # whisper.cpp CUDA builds can crash opening -m with an absolute Chinese
        # path. The model directory as cwd and its ASCII basename are verified
        # to work; the CLI correctly accepts Unicode input and output paths.
        output_stem = f"transcript_{uuid.uuid4().hex}"
        output_base = source.parent / output_stem
        json_path = output_base.with_suffix(".json")
        threads = max(1, int(self.config.get("whisper_threads", min(8, os.cpu_count() or 4))))
        temporary_model = None
        model_name = model.name
        if not model_name.isascii():
            temporary_model = model.with_name(f"speech_model_{uuid.uuid4().hex}.bin")
            try:
                os.link(model, temporary_model)
            except OSError:
                shutil.copy2(model, temporary_model)
            model_name = temporary_model.name
        args = [self.whisper, "-m", model_name, "-f", str(source), "-l", language,
                "-t", str(threads), "-oj", "-of", str(output_base), "-pp", "-sns"]
        if not self.config.get("whisper_gpu", True):
            args.append("-ng")
        vad_model = self.config.get("whisper_vad_model")
        temporary_vad = None
        if vad_model and self.config.get("whisper_vad_enabled",False):
            vad = self._path(vad_model)
            self._require_file(vad, "语音活动检测模型")
            if vad.parent == model.parent and vad.name.isascii():
                vad_name = vad.name
            else:
                temporary_vad = model.parent / f"speech_vad_{uuid.uuid4().hex}.bin"
                try:
                    os.link(vad, temporary_vad)
                except OSError:
                    shutil.copy2(vad, temporary_vad)
                vad_name = temporary_vad.name
            args += ["--vad", "-vm", vad_name]
        try:
            self._event("progress", "transcribe", "正在本地识别语音；长视频可能需要较长时间", 0)
            self._run(args, "transcribe", cwd=model.parent)
            self._check_cancelled()
            if not json_path.is_file():
                raise MediaError("Whisper 没有生成识别结果，请检查模型及程序日志")
            try:
                data = json.loads(json_path.read_text(encoding="utf-8-sig"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise MediaError(f"无法解析语音识别结果：{json_path.name}") from exc
            segments, detected = parse_whisper_json(data)
            self._event("progress", "transcribe", f"语音识别完成：{len(segments)} 个字幕片段，语言 {detected}", 100)
            return segments, detected
        finally:
            if temporary_model:
                temporary_model.unlink(missing_ok=True)
            if temporary_vad:
                temporary_vad.unlink(missing_ok=True)

    def _transcribe_qwen(self, wav, language='auto'):
        source = Path(wav).resolve()
        self._require_file(source,'音频文件')
        python = self._path(self.config['qwen_python'])
        worker = self.base_dir/'qwen_asr_worker.py'
        self._require_file(worker,'Qwen语音模块')
        output = source.parent/('qwen_transcript_'+uuid.uuid4().hex+'.json')
        args = [python,worker,'--audio',source,'--output',output,
                '--model',self.config['qwen_asr_model'],'--aligner',self.config['qwen_aligner_model'],
                '--language',language]
        self._event('progress','transcribe','正在完整识别音轨，并对齐每句话的时间戳',0)
        self._run(args,'transcribe',cwd=self.base_dir)
        self._check_cancelled()
        if not output.is_file():
            raise MediaError('Qwen没有生成识别结果，请查看任务日志。')
        result = json.loads(output.read_text(encoding='utf-8-sig'))
        segments = result.get('segments',[])
        if not segments:
            raise MediaError('未识别到可用语音。请检查音轨，或手动指定源语言重试。')
        for i, item in enumerate(segments,1):
            if not str(item.get('text','')).strip():
                raise MediaError('识别结果包含空字幕，已停止以避免漏译。')
            if not math.isfinite(item['start']) or not math.isfinite(item['end']) or item['start'] < 0 or item['end'] <= item['start']:
                raise MediaError('识别文字存在无效时间戳，已停止以避免悄悄丢掉台词。')
            item['id'] = i
        self._event('progress','transcribe',f'完整识别完成：{len(segments)} 条字幕，语言 {result.get("language")}',100)
        return segments,result.get('language') or language

    def _transcribe_kotoba(self, wav, language='ja'):
        if language != 'ja':
            raise MediaError('Kotoba-Whisper 仅用于日语，请按视频主要语言选择识别模型。')
        source = Path(wav).resolve()
        self._require_file(source, '音频文件')
        python = self._path(self.config['qwen_python'])
        worker = self.base_dir / 'kotoba_asr_worker.py'
        self._require_file(worker, 'Kotoba 日语识别模块')
        output = source.parent / ('kotoba_transcript_' + uuid.uuid4().hex + '.json')
        args = [python, worker, '--audio', source, '--output', output, '--model', self.config['kotoba_model']]
        self._event('progress', 'transcribe', '正在用日语专用模型识别语音并对齐字幕时间戳', 0)
        self._run(args, 'transcribe', cwd=self.base_dir)
        self._check_cancelled()
        if not output.is_file():
            raise MediaError('Kotoba 没有生成识别结果，请查看任务日志。')
        result = json.loads(output.read_text(encoding='utf-8-sig'))
        segments = result.get('segments', [])
        if not segments:
            raise MediaError('Kotoba 没有识别到可用日语语音。')
        for index, item in enumerate(segments, 1):
            if not str(item.get('text', '')).strip():
                raise MediaError('识别结果包含空字幕，已停止以避免漏译。')
            if not math.isfinite(item['start']) or not math.isfinite(item['end']) or item['start'] < 0 or item['end'] <= item['start']:
                raise MediaError('识别文字存在无效时间戳，已停止以避免悄悄丢掉台词。')
            item['id'] = index
        self._event('progress', 'transcribe', f'日语识别完成：{len(segments)} 条字幕', 100)
        return segments, 'ja'

    def export(self, input: str | Path, zh_srt: str | Path, out: str | Path,
               mode: str = "burn") -> Path:
        source, subtitle, target = Path(input).resolve(), Path(zh_srt).resolve(), Path(out).resolve()
        self._require_file(source, "视频文件")
        self._require_file(subtitle, "中文字幕文件")
        if mode not in {"burn", "soft", "softsub", "srt", "subtitles"}:
            raise MediaError(f"未知的导出模式：{mode}")
        if _same_file(source, target):
            raise MediaError("输出路径不能与原视频相同")
        if mode in {"srt", "subtitles"} and _same_file(subtitle, target):
            return target
        if target.exists():
            raise MediaError(f"输出文件已经存在，为保护原文件已停止：{target}")
        self._check_cancelled()
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode in {"srt", "subtitles"}:
            with subtitle.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            self._event("progress", "export", "中文字幕已保存", 100)
            return target
        info = self.probe(source)
        if not info["has_video"]:
            raise MediaError("所选文件没有可导出的视频画面")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_name(f"video_{uuid.uuid4().hex}{'.mp4' if mode == 'burn' else '.mkv'}")
        try:
            with tempfile.TemporaryDirectory(prefix="subtitle_", dir=self.work_dir) as staging:
                stage_dir = Path(staging)
                shutil.copy2(subtitle, stage_dir / "subtitle.srt")
                args = [self.ffmpeg, "-hide_banner", "-nostdin", "-n", "-i", source]
                if mode == "burn":
                    font_dir_arg = ""
                    candidates = [self.config.get("subtitle_font_path", ""),
                                  str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"),
                                  str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/simhei.ttf")]
                    for candidate in candidates:
                        if candidate and Path(candidate).is_file():
                            fonts = stage_dir / "fonts"
                            fonts.mkdir()
                            shutil.copy2(candidate, fonts / ("font" + Path(candidate).suffix))
                            font_dir_arg = ":fontsdir=fonts"
                            break
                    # ASS uses a 288-high canvas: scale portrait captions by
                    # aspect ratio so they do not occupy half a vertical video.
                    ratio = float(info.get('width') or 16) / float(info.get('height') or 9)
                    font_size = round(22 * min(1.0, ratio), 1)
                    outline = round(max(0.6, font_size * 0.068), 1)
                    style = f"FontName=Microsoft YaHei,FontSize={font_size},Outline={outline},Shadow=0,MarginV=18"
                    vf = ("subtitles=filename=subtitle.srt:charenc=UTF-8" + font_dir_arg
                          + ":force_style='" + style + "',scale=ceil(iw/2)*2:ceil(ih/2)*2")
                    args += ["-map", "0:v:0", "-map", "0:a?", "-vf", vf,
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                             "-movflags", "+faststart", "-map_metadata", "0", "-f", "mp4"]
                else:
                    args += ["-sub_charenc", "UTF-8", "-i", "subtitle.srt",
                             "-map", "0:v:0", "-map", "0:a?", "-map", "1:0",
                             "-c:v", "copy", "-c:a", "copy", "-c:s", "srt",
                             "-metadata:s:s:0", "language=zho", "-metadata:s:s:0", "title=中文字幕",
                             "-disposition:s:0", "default", "-map_metadata", "0", "-f", "matroska"]
                args += ["-progress", "pipe:1", "-nostats", temp_path]
                self._event("progress", "export", "正在导出带中文字幕的视频", 0)
                self._run(args, "export", cwd=stage_dir, duration=info["duration"])
                self._check_cancelled()
                if not temp_path.is_file() or temp_path.stat().st_size == 0:
                    raise MediaError("导出的视频文件为空")
                _publish(temp_path, target)
            self._event("progress", "export", "视频导出完成", 100)
            return target
        finally:
            temp_path.unlink(missing_ok=True)
