"""Bounded Whisper recognition with independent speech-timing checks.

whisper.cpp's VAD restores timestamps after removing silence. On sparse long
recordings a word can inherit a several-minute silence gap. Work in bounded
windows, then re-recognize only suspicious speech islands without VAD mapping.
Every intermediate JSON is kept in the job directory so a cancelled task can
resume without repeating completed model calls.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import uuid
import wave


def speech_overlap(start: float, end: float, vad: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in vad if a < end and b > start)


def problems(rows: list[dict], vad: list[tuple[float, float]]) -> dict[int, list[str]]:
    found = {}
    for index, row in enumerate(rows):
        start, end, text = row['start'], row['end'], row['text']
        duration = end - start
        intervals = [(max(start, a), min(end, b)) for a, b in vad if a < end and b > start]
        intervals = [(a, b) for a, b in intervals if b > a]
        overlap = sum(b - a for a, b in intervals)
        gap = max((intervals[i + 1][0] - intervals[i][1] for i in range(len(intervals) - 1)), default=0)
        chars = len(re.sub(r'[\s、。！？!?.,，:：;；（）()「」『』\[\]]', '', text))
        flags = []
        if not intervals:
            flags.append('no_speech')
        if duration > max(14, len(text) / .9):
            flags.append('too_long')
        if chars >= 12 and duration < chars / 16:
            flags.append('too_short')
        if duration > 3 and overlap / duration < .2:
            flags.append('mostly_silence')
        if gap > 5:
            flags.append('crosses_silence')
        if re.fullmatch(r'[\s♪♫\[\]（）()音楽.。]+', text):
            flags.append('music')
        if chars >= 10:
            normalized = re.sub(r'\W', '', text)
            if any(re.sub(r'\W', '', previous['text']) == normalized
                   and abs(start - previous['start']) < 8
                   for previous in rows[max(0, index - 8):index]):
                flags.append('repeated')
        if flags:
            found[index] = flags
    return found


def group_speech(vad: list[tuple[float, float]]) -> list[dict]:
    groups = []
    for start, end in vad:
        if groups and start - groups[-1]['end'] <= 3 and end - groups[-1]['start'] <= 30:
            groups[-1]['end'] = end
            groups[-1]['parts'].append((start, end))
        else:
            groups.append({'start': start, 'end': end, 'parts': [(start, end)]})
    return groups


def parse_entries(data: dict, offset: float, lower: float, upper: float,
                  *, midpoint_only: bool = False) -> list[dict]:
    entries = data.get('transcription')
    if not isinstance(entries, list):
        raise ValueError('Whisper JSON 中没有字幕列表')
    rows = []
    for entry in entries:
        text = str(entry.get('text') or '').strip()
        if not text or text.casefold() in {'[blank_audio]', '[silence]', '[no_speech]'}:
            continue
        if re.fullmatch(r'[\s♪♫\[\]（）()音楽.。]+', text):
            continue
        stamp = entry.get('offsets') or {}
        start = offset + float(stamp['from']) / 1000
        end = offset + float(stamp['to']) / 1000
        if midpoint_only:
            if not lower <= (start + end) / 2 < upper:
                continue
        else:
            start, end = max(start, lower), min(end, upper)
        if 0 <= start < end <= upper + (6 if midpoint_only else 1):
            rows.append({'start': round(start, 3), 'end': round(end, 3), 'text': text})
    return rows


def extend_short_cues(rows: list[dict], audio_duration: float, minimum: float = .9) -> None:
    """Give brief answers a readable tail without shifting speech starts."""
    for index, row in enumerate(rows):
        if row['end'] - row['start'] >= minimum:
            continue
        next_start = rows[index + 1]['start'] - .04 if index + 1 < len(rows) else audio_duration
        row['end'] = round(max(row['end'], min(row['start'] + minimum, next_start, audio_duration)), 3)


def deduplicate_overlaps(rows: list[dict]) -> list[dict]:
    """Drop alternative decodes of one utterance, preserving distinct replies."""
    unique = []
    for row in rows:
        normalized = re.sub(r'\W', '', row['text'])
        previous_index = next((i for i in range(len(unique) - 1, max(-1, len(unique) - 9), -1)
                               if normalized and normalized == re.sub(r'\W', '', unique[i]['text'])
                               and min(row['end'], unique[i]['end'])
                               - max(row['start'], unique[i]['start']) > .2), None)
        if previous_index is None:
            unique.append(row)
        elif row.get('asr_origin') == 'speech-group' and unique[previous_index].get('asr_origin') != 'speech-group':
            unique[previous_index] = row
        elif row['end'] - row['start'] > unique[previous_index]['end'] - unique[previous_index]['start']:
            unique[previous_index] = row
    return sorted(unique, key=lambda x: (x['start'], x['end']))


def salvage_short_speech(row: dict, flags: list[str], vad: list[tuple[float, float]]) -> dict | None:
    """Keep a brief real reply when no model rerun survives quality checks."""
    if set(flags) & {'no_speech', 'too_short', 'repeated', 'music'}:
        return None
    text = row['text']
    chars = len(re.sub(r'\W', '', text))
    if chars > 12:
        return None
    matches = [(speech_overlap(row['start'], row['end'], [(a, b)]), a, b)
               for a, b in vad if a < row['end'] and b > row['start']]
    if not matches:
        return None
    amount, start, end = max(matches)
    if amount < .3:
        return None
    start = max(row['start'], start - .15)
    end = min(row['end'], end + .15)
    if end - start < max(.35, chars / 16):
        return None
    return {**row, 'start': round(start, 3), 'end': round(end, 3),
            'asr_origin': 'speech-clipped'}


class PreciseWhisper:
    def __init__(self, media):
        self.media = media
        self.config = media.config
        self.work = media.work_dir / 'precise_whisper'
        self.work.mkdir(parents=True, exist_ok=True)
        self.model = media._path(self.config['whisper_model'])
        self.vad_model = media._path(self.config['whisper_vad_model'])
        self.vad_exe = media.whisper.with_name('whisper-vad-speech-segments.exe')
        for path, label in ((self.model, 'Whisper 模型'), (self.vad_model, '语音检测模型'),
                            (self.vad_exe, '语音检测程序')):
            media._require_file(path, label)
        if not self.model.name.isascii() or not self.vad_model.name.isascii():
            raise ValueError('精细识别模型文件名须使用英文字母；模型路径可以包含中文。')
        if self.model.parent != self.vad_model.parent:
            raise ValueError('Whisper 与语音检测模型需要放在同一模型目录。')
        self.force = bool(self.config.get('asr_force', False))

    def _event(self, message: str, percent: float):
        self.media._event('progress', 'transcribe', message, percent)

    def _json(self, path: Path):
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def _cut(self, source: Path, start: float, end: float, target: Path):
        target.unlink(missing_ok=True)
        self.media._run([self.media.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                         '-ss', str(start), '-t', str(end - start), '-i', source,
                         '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', target],
                        'transcribe', capture=True)
        if not target.is_file() or target.stat().st_size <= 44:
            raise RuntimeError('精细识别音频裁剪失败')

    def _whisper(self, audio: Path, output: Path, language: str, *, vad: bool):
        args = [self.media.whisper, '-m', self.model.name, '-f', audio, '-l', language,
                '-t', str(max(1, int(self.config.get('whisper_threads', 8)))),
                '-oj', '-of', output.with_suffix(''), '-pp', '-sns']
        if not self.config.get('whisper_gpu', True):
            args.append('-ng')
        if vad:
            args += ['--vad', '-vm', self.vad_model.name, '-vt', '.35',
                     '-vspd', '150', '-vsd', '300', '-vmsd', '25', '-vp', '150']
        log = self.media._run(args, 'transcribe', cwd=self.model.parent, capture=True)
        output.with_suffix('.log').write_text(log, encoding='utf-8')
        if not output.is_file():
            raise RuntimeError('Whisper 未生成识别结果')
        return self._json(output)

    def _vad(self, source: Path) -> list[tuple[float, float]]:
        target = self.work / 'speech_segments.json'
        if self.force:
            target.unlink(missing_ok=True)
        if not target.is_file():
            log = self.media._run([self.vad_exe, '-f', source, '-vm', self.vad_model.name,
                                   '-vt', '.35', '-vspd', '150', '-vsd', '300',
                                   '-vmsd', '25', '-vp', '150', '-np'],
                                  'transcribe', cwd=self.model.parent, capture=True)
            matches = re.findall(r'Speech segment \d+: start = ([\d.]+), end = ([\d.]+)', log)
            if not matches and 'Detected 0 speech segments' not in log:
                raise RuntimeError('语音检测没有生成可解析的时间区段')
            # The standalone whisper.cpp tool prints centiseconds.
            vad = [[float(a) / 100, float(b) / 100] for a, b in matches]
            target.write_text(json.dumps(vad), encoding='utf-8')
        return [tuple(x) for x in self._json(target)]

    def transcribe(self, source: Path, language: str) -> tuple[list[dict], str]:
        with wave.open(str(source), 'rb') as wav:
            if wav.getnchannels() != 1 or wav.getframerate() != 16000:
                raise ValueError('精细识别需要单声道 16 kHz 音频')
            duration = wav.getnframes() / wav.getframerate()
        if duration <= 0:
            raise ValueError('音频时长无效')
        total = math.ceil(duration / 300)
        baseline = []
        self._event('正在分段识别语音并检测静音', 0)
        for index in range(total):
            self.media._check_cancelled()
            lower, upper = index * 300, min(duration, (index + 1) * 300)
            start, end = max(0, lower - 5), min(duration, upper + 5)
            output = self.work / f'window_{index:03d}.json'
            if self.force:
                output.unlink(missing_ok=True)
            if not output.is_file():
                crop = self.work / f'audio_window_{index:03d}_{uuid.uuid4().hex}.wav'
                try:
                    self._cut(source, start, end, crop)
                    data = self._whisper(crop, output, language, vad=True)
                finally:
                    crop.unlink(missing_ok=True)
            else:
                data = self._json(output)
            baseline.extend({**row, 'asr_origin': 'window'} for row in
                            parse_entries(data, start, lower, upper, midpoint_only=True))
            self._event(f'识别窗口 {index + 1}/{total}', 75 * (index + 1) / total)
        baseline.sort(key=lambda x: (x['start'], x['end']))
        vad = self._vad(source)
        bad = problems(baseline, vad)
        groups = group_speech(vad)
        selected = []
        for i, group in enumerate(groups):
            speech = sum(end - start for start, end in group['parts'])
            has_caption = any(row['start'] < group['end'] and row['end'] > group['start']
                              for row in baseline)
            matches_problem = any(baseline[j]['start'] < group['end']
                                  and baseline[j]['end'] > group['start'] for j in bad)
            if (group['end'] - group['start'] >= .5 and matches_problem) or (
                    speech >= 1.2 and not has_caption):
                selected.append(i)
        repairs = {}
        rejected = []
        for step, index in enumerate(selected, 1):
            self.media._check_cancelled()
            group = groups[index]
            start, end = max(0, group['start'] - .8), min(duration, group['end'] + .8)
            output = self.work / f'group_{index:03d}.json'
            if self.force:
                output.unlink(missing_ok=True)
            if not output.is_file():
                crop = self.work / f'audio_group_{index:03d}_{uuid.uuid4().hex}.wav'
                try:
                    self._cut(source, start, end, crop)
                    data = self._whisper(crop, output, language, vad=False)
                finally:
                    crop.unlink(missing_ok=True)
            else:
                data = self._json(output)
            rows = parse_entries(data, start, group['start'] - .2, group['end'] + .2)
            row_problems = problems(rows, vad)
            severe = {'no_speech', 'too_long', 'too_short', 'mostly_silence', 'music'}
            good_rows = [{**row, 'asr_origin': 'speech-group'} for j, row in enumerate(rows)
                         if not set(row_problems.get(j, [])) & severe]
            if good_rows:
                repairs[index] = good_rows
            else:
                rejected.append(index)
            self._event(f'复核异常语音 {step}/{len(selected)}',
                        75 + 24 * step / max(1, len(selected)))
        kept, removed, salvaged = [], [], []
        for index, row in enumerate(baseline):
            midpoint = (row['start'] + row['end']) / 2
            covered = any(groups[i]['start'] - .2 <= midpoint <= groups[i]['end'] + .2 for i in repairs)
            if index in bad or covered:
                removed.append(row)
                if index in bad and not any(row['start'] < groups[i]['end']
                                            and row['end'] > groups[i]['start'] for i in repairs):
                    rescue = salvage_short_speech(row, bad[index], vad)
                    if rescue:
                        kept.append(rescue)
                        salvaged.append(rescue)
            else:
                kept.append(row)
        combined = kept + [row for part in repairs.values() for row in part]
        combined.sort(key=lambda x: (x['start'], x['end']))
        unique = deduplicate_overlaps(combined)
        for index, row in enumerate(unique, 1):
            row['id'] = index
        extend_short_cues(unique, duration)
        remaining = problems(unique, vad)
        report = {'windows': total, 'baseline_cues': len(baseline), 'flagged_cues': len(bad),
                  'repair_groups': len(repairs), 'rejected_groups': rejected,
                  'removed_cues': len(removed), 'salvaged_short_cues': len(salvaged),
                  'final_cues': len(unique),
                  'remaining_flags': {str(i + 1): value for i, value in remaining.items()}}
        (self.work / 'quality_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                       encoding='utf-8')
        self._event(f'精细识别完成：{len(unique)} 条字幕', 100)
        return unique, language
