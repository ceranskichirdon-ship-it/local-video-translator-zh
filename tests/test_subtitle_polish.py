from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subtitle_polish import polish_segments


class SubtitlePolishTests(unittest.TestCase):
    def test_nested_decode_is_removed_and_different_replies_do_not_overlap(self):
        rows = [
            {'id': 1, 'start': 1.0, 'end': 5.0, 'text': 'これは同じ長い言葉です'},
            {'id': 2, 'start': 2.0, 'end': 3.0, 'text': '同じ長い言葉'},
            {'id': 3, 'start': 4.0, 'end': 6.0, 'text': '次の返事です'},
        ]
        result, report = polish_segments(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual([row['id'] for row in result], [1, 2])
        self.assertEqual(result[0]['end'], 3.96)
        self.assertEqual(report['removed_duplicate_overlaps'], [2])

    def test_long_cue_splits_without_losing_characters_or_time(self):
        text = '最初の文です。' * 18
        result, report = polish_segments([{'id': 1, 'start': 10, 'end': 28, 'text': text}])
        self.assertGreater(len(result), 1)
        self.assertEqual(''.join(row['text'] for row in result), text)
        self.assertEqual(result[0]['start'], 10)
        self.assertEqual(result[-1]['end'], 28)
        self.assertEqual(report['split_long_cues'], 1)

    def test_reviewed_audio_boundaries_replace_one_long_cue(self):
        (ROOT / 'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as temporary:
            path = Path(temporary) / 'overrides.json'
            path.write_text(json.dumps({'replacements': [{
                'start': 1, 'end': 7, 'text': '前半後半',
                'segments': [{'start': 1, 'end': 4, 'text': '前半'},
                             {'start': 4, 'end': 7, 'text': '後半'}]}]}), encoding='utf-8')
            result, report = polish_segments([{'id': 1, 'start': 1, 'end': 7, 'text': '前半後半'}],
                                             override_path=path)
        self.assertEqual([row['text'] for row in result], ['前半', '後半'])
        self.assertEqual(report['reviewed_replacements'], 1)

    def test_stale_audio_review_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as temporary:
            path = Path(temporary) / 'overrides.json'
            path.write_text(json.dumps({'replacements': [{
                'start': 1, 'end': 2, 'text': '古い識別稿',
                'segments': [{'start': 1, 'end': 2, 'text': '修正稿'}]
            }]}, ensure_ascii=False), encoding='utf-8')
            with self.assertRaises(ValueError):
                polish_segments([{'id': 1, 'start': 1, 'end': 2, 'text': '新しい識別稿'}],
                                override_path=path)

    def test_repeated_decode_in_silence_is_removed_after_clipping(self):
        rows = [
            {'id': 1, 'start': 2999.0, 'end': 3006.23,
             'text': '知らないあったばかりの男のチンポしゃぶって喜んでる'},
            {'id': 2, 'start': 3003.71, 'end': 3007.23,
             'text': '知らない、会ったばかりの男のチンポをしゃぶって喜んでいる'},
        ]
        result, report = polish_segments(rows, [(2999.3, 2999.54), (3003.91, 3007.03)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['start'], 3003.71)
        self.assertEqual(report['removed_near_silence'], [1])

    def test_short_overlap_is_dropped_and_repeated_phrase_is_merged(self):
        rows = [
            {'id': 1, 'start': 598.75, 'end': 600.76,
             'text': 'あの打ち合わせとかやっぱちょっと'},
            {'id': 2, 'start': 599.5, 'end': 603.26,
             'text': '打ち合わせとかやっぱりちょっとお互いの現場に行くときは'},
            {'id': 3, 'start': 603.26, 'end': 603.36, 'text': 'そうですね'},
            {'id': 4, 'start': 603.3, 'end': 604.0, 'text': '現場です'},
        ]
        result, report = polish_segments(rows, [(598.75, 604.0)])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['start'], 598.75)
        self.assertTrue(result[0]['text'].startswith('あの打ち合わせ'))
        self.assertEqual(report['merged_short_repetitions'], [1])
        self.assertIn(3, report['removed_near_silence'])


if __name__ == '__main__':
    unittest.main()
