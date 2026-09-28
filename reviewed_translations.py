"""Apply exact, auditable subtitle corrections after local-model translation."""
from __future__ import annotations

import json
from pathlib import Path

from translation import validated_translation


def apply_reviewed_translations(rows: list[dict], language: str,
                                path: Path) -> tuple[list[dict], dict]:
    """Match each correction to its original cue so stale reviews cannot drift."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    replacements = data.get('replacements')
    if not isinstance(replacements, list):
        raise ValueError('人工复核译文文件缺少 replacements 列表')
    by_id = {row['id']: row for row in rows}
    positions = {row['id']: index for index, row in enumerate(rows)}
    seen = set()
    changes = []
    result = [dict(row) for row in rows]
    for correction in replacements:
        cue_id = correction.get('id')
        if type(cue_id) is not int or cue_id in seen or cue_id not in by_id:
            raise ValueError(f'人工复核译文编号无效或重复：{cue_id}')
        seen.add(cue_id)
        original = by_id[cue_id]
        if (original['text'] != correction.get('text') or
                abs(original['start'] - float(correction.get('start', -1))) > .01 or
                abs(original['end'] - float(correction.get('end', -1))) > .01):
            raise ValueError(f'人工复核译文与当前识别稿不符：{cue_id}')
        reviewed = validated_translation(correction.get('zh'), original['text'], language)
        if reviewed == original['zh']:
            continue
        result[positions[cue_id]]['zh'] = reviewed
        changes.append({'id': cue_id, 'start': original['start'],
                        'original_zh': original['zh'], 'reviewed_zh': reviewed,
                        'reason': correction.get('reason', '')})
    return result, {'reviewed_cues': len(changes), 'changes': changes}
