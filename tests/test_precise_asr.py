"""Checks for the timing failures seen in sparse Japanese demo audio."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precise_asr import (deduplicate_overlaps, extend_short_cues, group_speech,
                         parse_entries, problems, salvage_short_speech)


class PreciseTimingTests(unittest.TestCase):
    def test_window_offsets_and_overlap_midpoints(self):
        data = {'transcription': [
            {'offsets': {'from': 1000, 'to': 3000}, 'text': '前の窓'},
            {'offsets': {'from': 4500, 'to': 6500}, 'text': '境界の台詞'},
            {'offsets': {'from': 10000, 'to': 11000}, 'text': '[BLANK_AUDIO]'},
            {'offsets': {'from': 12000, 'to': 13500}, 'text': '次の台詞'}]}
        rows = parse_entries(data, 295, 300, 600, midpoint_only=True)
        self.assertEqual([(r['start'], r['end'], r['text']) for r in rows], [
            (299.5, 301.5, '境界の台詞'), (307, 308.5, '次の台詞')])

    def test_long_silence_and_implausible_repetition_are_flagged(self):
        rows = [
            {'start': 10, 'end': 12, 'text': 'これは本当に長い日本語の台詞です'},
            {'start': 12.1, 'end': 12.2, 'text': 'これは本当に長い日本語の台詞です'},
            {'start': 20, 'end': 65, 'text': 'はい'},
        ]
        flagged = problems(rows, [(10, 12.3), (60, 61)])
        self.assertIn('too_short', flagged[1])
        self.assertIn('repeated', flagged[1])
        self.assertIn('too_long', flagged[2])
        self.assertIn('mostly_silence', flagged[2])

    def test_speech_groups_never_swallow_long_gaps(self):
        groups = group_speech([(0, 2), (3, 5), (20, 22), (24, 27)])
        self.assertEqual([(g['start'], g['end']) for g in groups], [(0, 5), (20, 27)])

    def test_brief_subtitles_are_readable_without_covering_next_speaker(self):
        rows = [{'start': 1.0, 'end': 1.1, 'text': 'はい'},
                {'start': 1.5, 'end': 2.0, 'text': '次の言葉'}]
        extend_short_cues(rows, 3)
        self.assertEqual(rows[0]['start'], 1.0)
        self.assertEqual(rows[0]['end'], 1.46)
        self.assertEqual(rows[1]['end'], 2.4)

    def test_same_utterance_from_overlap_is_not_subtitled_twice(self):
        rows = [{'start': 3.0, 'end': 5.0, 'text': '同じ言葉。', 'asr_origin': 'window'},
                {'start': 3.2, 'end': 4.9, 'text': '同じ言葉', 'asr_origin': 'speech-group'},
                {'start': 5.1, 'end': 6.0, 'text': 'はい', 'asr_origin': 'window'}]
        result = deduplicate_overlaps(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['asr_origin'], 'speech-group')

    def test_short_reply_is_clipped_to_real_speech_when_rerun_fails(self):
        row = {'start': 10, 'end': 18, 'text': 'はい'}
        fixed = salvage_short_speech(row, ['mostly_silence'], [(10.2, 10.9)])
        self.assertEqual((fixed['start'], fixed['end']), (10.05, 11.05))
        self.assertIsNone(salvage_short_speech({**row, 'text': '長い文を一瞬で言うことはできません'},
                                                ['too_short'], [(10.2, 10.9)]))


if __name__ == '__main__':
    unittest.main()
