"""Final subtitle timing/readability pass for precision Whisper results."""
from __future__ import annotations

import json
from pathlib import Path
import re
from difflib import SequenceMatcher

from precise_asr import speech_overlap


def _plain(text: str) -> str:
    return re.sub(r'\W', '', text)


def _shared_text_ratio(first: str, second: str) -> float:
    """How much of the shorter utterance appears in the other decode."""
    shorter = min(len(first), len(second))
    if not shorter:
        return 0.0
    return sum(block.size for block in SequenceMatcher(None, first, second,
                                                       autojunk=False).get_matching_blocks()) / shorter


def _split_long(row: dict, limit: int = 44) -> list[dict]:
    text = row['text'].strip()
    if len(text) <= 90:
        return [row]
    # Prefer natural pauses; Japanese often has spaces in Whisper's output.
    units = [unit for unit in re.split(r'(?<=[。！？!?])|\s+', text) if unit]
    parts = []
    current = ''
    for unit in units:
        if current and len(current) + len(unit) > limit:
            parts.append(current)
            current = ''
        if len(unit) > limit:
            if current:
                parts.append(current)
                current = ''
            parts.extend(unit[i:i + limit] for i in range(0, len(unit), limit))
        else:
            current += (' ' if current and re.search(r'[A-Za-z0-9]$', current)
                        and re.match(r'[A-Za-z0-9]', unit) else '') + unit
    if current:
        parts.append(current)
    if len(parts) <= 1:
        return [row]
    weights = [max(1, len(_plain(part))) for part in parts]
    total = sum(weights)
    duration = row['end'] - row['start']
    result = []
    elapsed = 0
    for index, (part, weight) in enumerate(zip(parts, weights)):
        start = row['start'] + duration * elapsed / total
        elapsed += weight
        end = row['end'] if index == len(parts) - 1 else row['start'] + duration * elapsed / total
        result.append({**row, 'start': round(start, 3), 'end': round(end, 3),
                       'text': part.strip(), 'polish': 'long-cue-split'})
    return result


def _overrides(rows: list[dict], path: Path | None, report: dict) -> list[dict]:
    if not path or not path.is_file():
        return rows
    data = json.loads(path.read_text(encoding='utf-8'))
    replacements = data.get('replacements', [])
    if not isinstance(replacements, list):
        raise ValueError('人工复核字幕文件缺少 replacements 列表')
    result = []
    matched = set()
    for row in rows:
        match = next(((index, item) for index, item in enumerate(replacements)
                            if abs(float(item['start']) - row['start']) < .01
                            and abs(float(item['end']) - row['end']) < .01
                            and item['text'] == row['text']), None)
        if match:
            index, replacement = match
            if index in matched:
                raise ValueError('人工复核字幕替换项重复匹配')
            matched.add(index)
            pieces = replacement['segments']
            if not pieces or any(not part['text'].strip() or not row['start'] - .2 <= part['start']
                                 < part['end'] <= row['end'] + .2 for part in pieces):
                raise ValueError('人工复核的字幕替换片段无效')
            result.extend({**row, **part, 'polish': 'reviewed-audio'} for part in pieces)
            report['reviewed_replacements'] += 1
        else:
            result.append(row)
    if len(matched) != len(replacements):
        raise ValueError(f'有 {len(replacements) - len(matched)} 条人工复核字幕与当前识别稿不符')
    return result


def _trim_overlaps(rows: list[dict], report: dict) -> None:
    for index in range(len(rows) - 1):
        current, following = rows[index], rows[index + 1]
        if current['end'] <= following['start'] + .04:
            continue
        new_end = round(following['start'] - .04, 3)
        if new_end > current['start'] + .2:
            current['end'] = new_end
            report['trimmed_overlaps'] += 1
    report['unresolved_overlaps'] = [rows[index].get('id') for index in range(len(rows) - 1)
                                     if rows[index]['end'] > rows[index + 1]['start'] + .04]


def polish_segments(segments: list[dict], vad: list[tuple[float, float]] | None = None,
                    override_path: Path | None = None) -> tuple[list[dict], dict]:
    """Remove overlapping alternative decodes and prevent simultaneous captions."""
    vad = vad or []
    report = {'input_cues': len(segments), 'reviewed_replacements': 0,
              'split_long_cues': 0, 'removed_near_silence': [],
              'removed_duplicate_overlaps': [], 'merged_short_repetitions': [],
              'trimmed_overlaps': 0}
    rows = _overrides([dict(row) for row in segments], override_path, report)
    expanded = []
    for row in rows:
        pieces = _split_long(row)
        if len(pieces) > 1:
            report['split_long_cues'] += 1
        expanded.extend(pieces)
    rows = sorted(expanded, key=lambda row: (row['start'], row['end']))
    filtered = []
    for index, row in enumerate(rows):
        # A caption which barely touches a speech island but overlaps the
        # preceding real caption is usually a VAD-restoration artifact.
        speech = speech_overlap(row['start'], row['end'], vad) if vad else 0.0
        neighbors = [rows[neighbor] for neighbor in (index - 1, index + 1)
                     if 0 <= neighbor < len(rows)]
        nearby_repeat = any(
            min(abs(row['start'] - neighbor['end']), abs(neighbor['start'] - row['end'])) < .25
            and _shared_text_ratio(_plain(row['text']), _plain(neighbor['text'])) >= .75
            for neighbor in neighbors)
        implausibly_short = (row['end'] - row['start'] < .25 and len(_plain(row['text'])) >= 4
                             and any(neighbor['start'] < row['end'] + .1
                                     and neighbor['end'] > row['start'] - .1 for neighbor in neighbors))
        if (implausibly_short or vad and
                (speech < .02 and filtered and filtered[-1]['end'] - row['start'] > .2
                 or speech < .2 * (row['end'] - row['start']) and nearby_repeat)):
            report['removed_near_silence'].append(row.get('id'))
            continue
        filtered.append(row)
    rows = filtered
    dropped = set()
    for i, first in enumerate(rows):
        if i in dropped:
            continue
        for j in range(i + 1, min(len(rows), i + 9)):
            if j in dropped:
                continue
            if rows[j]['start'] >= first['end']:
                break
            second = rows[j]
            shared = min(first['end'], second['end']) - max(first['start'], second['start'])
            if shared <= .2:
                continue
            a, b = _plain(first['text']), _plain(second['text'])
            shorter = first if len(a) <= len(b) else second
            smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
            shorter_coverage = shared / (shorter['end'] - shorter['start'])
            same_utterance = (len(smaller) >= 5 and smaller in larger and shorter_coverage >= .5)
            same_utterance |= (len(smaller) >= 8 and shorter_coverage >= .7
                               and _shared_text_ratio(smaller, larger) >= .85)
            if same_utterance:
                if len(a) == len(b):
                    if first.get('asr_origin') == 'speech-group' and second.get('asr_origin') != 'speech-group':
                        discard = j
                    elif second.get('asr_origin') == 'speech-group' and first.get('asr_origin') != 'speech-group':
                        discard = i
                    else:
                        discard = i if first['end'] - first['start'] < second['end'] - second['start'] else j
                else:
                    discard = i if shorter is first else j
                dropped.add(discard)
                report['removed_duplicate_overlaps'].append(rows[discard].get('id'))
                if discard == i:
                    break
    rows = [row for i, row in enumerate(rows) if i not in dropped]
    _trim_overlaps(rows, report)
    # A long alternative decode can borrow the following speaker's VAD island.
    # Recheck after shortening it to the time the viewer would actually see.
    if vad:
        dropped_after_trim = set()
        for index, row in enumerate(rows):
            duration = row['end'] - row['start']
            if duration <= 0 or speech_overlap(row['start'], row['end'], vad) >= .2 * duration:
                continue
            neighbors = [rows[neighbor] for neighbor in (index - 1, index + 1)
                         if 0 <= neighbor < len(rows)]
            if any(min(abs(row['start'] - neighbor['end']),
                       abs(neighbor['start'] - row['end'])) < .25
                   and _shared_text_ratio(_plain(row['text']), _plain(neighbor['text'])) >= .75
                   for neighbor in neighbors):
                dropped_after_trim.add(index)
                report['removed_near_silence'].append(row.get('id'))
        rows = [row for index, row in enumerate(rows) if index not in dropped_after_trim]
    merged = set()
    for index in range(len(rows) - 1):
        first, following = rows[index], rows[index + 1]
        a, b = _plain(first['text']), _plain(following['text'])
        if (index in merged or index + 1 in merged or
                first['end'] - first['start'] >= 1.0 or len(a) < 12 or
                abs(following['start'] - first['end']) > .15 or
                len(a) != len(first['text']) or len(b) != len(following['text']) or
                _shared_text_ratio(a, b) < .8):
            continue
        match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match()
        if match.size < 8:
            continue
        following['start'] = first['start']
        following['text'] = first['text'][:match.a] + following['text']
        merged.add(index)
        report['merged_short_repetitions'].append(first.get('id'))
    rows = [row for index, row in enumerate(rows) if index not in merged]
    _trim_overlaps(rows, report)
    for index, row in enumerate(rows, 1):
        row['id'] = index
    report['output_cues'] = len(rows)
    return rows, report
