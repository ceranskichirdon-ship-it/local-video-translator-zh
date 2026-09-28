from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reviewed_translations import apply_reviewed_translations


class ReviewedTranslationTests(unittest.TestCase):
    def test_exact_review_is_applied_without_changing_timestamps(self):
        rows = [{'id': 1, 'start': 1.2, 'end': 2.3, 'text': 'どれくらい?', 'zh': '有多严重？'}]
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temporary:
            path = Path(temporary)/'review.json'
            path.write_text(json.dumps({'replacements': [
                {'id': 1, 'start': 1.2, 'end': 2.3, 'text': 'どれくらい?',
                 'zh': '有多喜欢？', 'reason': 'previous question supplies context'}
            ]}, ensure_ascii=False), encoding='utf-8')
            result, report = apply_reviewed_translations(rows, 'ja', path)
        self.assertEqual(result[0]['zh'], '有多喜欢？')
        self.assertEqual((result[0]['start'], result[0]['end']), (1.2, 2.3))
        self.assertEqual(rows[0]['zh'], '有多严重？')
        self.assertEqual(report['reviewed_cues'], 1)

    def test_changed_recognition_rejects_stale_review(self):
        rows = [{'id': 1, 'start': 1.2, 'end': 2.3, 'text': '別の言葉', 'zh': '别的话'}]
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temporary:
            path = Path(temporary)/'review.json'
            path.write_text(json.dumps({'replacements': [
                {'id': 1, 'start': 1.2, 'end': 2.3, 'text': '元の言葉', 'zh': '原话'}
            ]}, ensure_ascii=False), encoding='utf-8')
            with self.assertRaises(ValueError):
                apply_reviewed_translations(rows, 'ja', path)
