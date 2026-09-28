from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_asr_worker import WorkerError, build_segments, language_code, project_aligned_text


def word(text, start, end):
    return SimpleNamespace(text=text, start_time=start, end_time=end)


def substantive(text):
    return ''.join(c for c in text if not c.isspace())


class SegmentationTests(unittest.TestCase):
    def test_korean_fragments_preserve_original_spaces_and_punctuation(self):
        original = '안녕하세요. 오늘은 정말 좋은 날이에요!'
        items = [word('안녕',3.2,3.4),word('하세요',3.4,4.0),word('오늘',4.3,4.6),
                 word('은',4.6,4.8),word('정말',4.9,5.4),word('좋은',5.5,6.0),
                 word('날',6.1,6.4),word('이에요',6.4,7.0)]
        segments=build_segments(original,items,'ko')
        self.assertEqual([s['text'] for s in segments],['안녕하세요.','오늘은 정말 좋은 날이에요!'])
        self.assertEqual(segments[0]['start'],3.2)
        self.assertEqual(segments[-1]['end'],7.0)

    def test_japanese_words_are_not_separated_by_invented_spaces(self):
        original='「こんにちは！」今日は晴れです。'
        items=[word('こんにちは',0.5,1.5),word('今日',2,2.5),word('は',2.5,2.6),
               word('晴れ',2.6,3.0),word('です',3,3.5)]
        segments=build_segments(original,items,'ja')
        self.assertEqual([s['text'] for s in segments],['「こんにちは！」','今日は晴れです。'])
        self.assertEqual(substantive(''.join(s['text'] for s in segments)),substantive(original))

    def test_every_english_word_and_symbol_survives_duration_splits(self):
        original='“Please, don\'t stop…” All of these words must remain! 🙂'
        tokens=['Please',"don't",'stop','All','of','these','words','must','remain']
        items=[word(t,i*.8,i*.8+.5) for i,t in enumerate(tokens)]
        segments=build_segments(original,items,'en',max_duration=2,max_chars=24)
        self.assertGreater(len(segments),2)
        self.assertEqual(substantive(''.join(s['text'] for s in segments)),substantive(original))
        self.assertTrue(segments[-1]['text'].endswith('🙂'))

    def test_long_audio_official_offsets_are_not_reset(self):
        original='First chunk. Second chunk.'
        items=[word('First',177,178),word('chunk',178,179.5),
               word('Second',182.2,183),word('chunk',183,184.1)]
        segments=build_segments(original,items,'en',duration=400)
        self.assertEqual(segments[1]['start'],182.2)
        self.assertEqual(segments[1]['end'],184.1)

    def test_pause_breaks_subtitle_without_filling_silence(self):
        segments=build_segments('hello again',[word('hello',1,2),word('again',4,5)],'en')
        self.assertEqual(len(segments),2)
        self.assertEqual(segments[0]['end'],2)
        self.assertEqual(segments[1]['start'],4)

    def test_eighty_millisecond_cue_is_readable_without_changing_word_times(self):
        items = [word('형', 34.96, 35.04)]
        original_times = [(item.start_time, item.end_time) for item in items]
        segments = build_segments('형', items, 'ko', duration=41.54)
        self.assertEqual(segments, [{'id': 1, 'start': 34.96, 'end': 35.86, 'text': '형'}])
        self.assertEqual([(item.start_time, item.end_time) for item in items], original_times)

    def test_short_cue_extension_stops_at_next_subtitle(self):
        segments = build_segments('Hey! Go.', [word('Hey', 1, 1.08), word('Go', 1.4, 2.5)], 'en')
        self.assertEqual(segments[0]['start'], 1)
        self.assertEqual(segments[0]['end'], segments[1]['start'])
        self.assertEqual(segments[0]['end'], 1.4)
        self.assertEqual(segments[1]['end'], 2.5)

    def test_short_final_cue_extension_stops_at_audio_end(self):
        segments = build_segments('Bye!', [word('Bye', 9.7, 9.78)], 'en', duration=10.1234)
        self.assertEqual(segments[0]['start'], 9.7)
        self.assertEqual(segments[0]['end'], 10.1234)

    def test_display_extension_preserves_complete_text_and_original_starts(self):
        original = '“형!” 다음 문장입니다. 🙂'
        items = [word('형', 1, 1.08), word('다음', 2, 2.3), word('문장입니다', 2.3, 3.2)]
        segments = build_segments(original, items, 'ko', duration=5)
        self.assertEqual([s['text'] for s in segments], ['“형!”', '다음 문장입니다. 🙂'])
        self.assertEqual(substantive(''.join(s['text'] for s in segments)), substantive(original))
        self.assertEqual([s['start'] for s in segments], [1, 2])
        self.assertEqual([s['end'] for s in segments], [1.9, 3.2])

    def test_zero_duration_tokens_are_kept(self):
        original='Every word stays.'
        items=[word('Every',1,1),word('word',1,2),word('stays',2,2)]
        segments=build_segments(original,items,'en',max_chars=5)
        self.assertEqual(substantive(''.join(s['text'] for s in segments)),substantive(original))
        self.assertTrue(all(s['end']>s['start'] for s in segments))

    def test_unmatched_or_omitted_words_raise_instead_of_disappearing(self):
        for items in ([word('first',0,1)], [word('wrong',0,1),word('last',1,2)]):
            with self.assertRaisesRegex(WorkerError,'遗漏|漏字'):
                build_segments('first last',items,'en')

    def test_empty_recognition_and_missing_timestamps_are_explained(self):
        with self.assertRaisesRegex(WorkerError,'没有识别到'):
            build_segments('',[])
        with self.assertRaisesRegex(WorkerError,'词级时间戳'):
            build_segments('some speech',[])

    def test_invalid_or_out_of_order_timestamps_are_rejected(self):
        for items in ([word('first',float('nan'),2)], [word('first',3,2)],
                      [word('first',2,3),word('last',1,2)]):
            text='first last' if len(items)==2 else 'first'
            with self.assertRaisesRegex(WorkerError,'时间戳'):
                build_segments(text,items)

    def test_outside_audio_timestamps_are_rejected(self):
        with self.assertRaisesRegex(WorkerError,'超出了'):
            build_segments('hello',[word('hello',0,12)],duration=5)

    def test_all_zero_timestamps_are_not_invented(self):
        with self.assertRaisesRegex(WorkerError,'有效时长'):
            build_segments('hello',[word('hello',0,0)])

    def test_projection_accepts_serialized_official_items_and_case_folding(self):
        units=project_aligned_text('HELLO, 世界!',[
            {'text':'hello','start_time':0,'end_time':1},
            {'text':'世','start_time':1,'end_time':1.2},
            {'text':'界','start_time':1.2,'end_time':1.5}])
        self.assertEqual(''.join(u.text for u in units),'HELLO, 世界!')

    def test_language_detection_does_not_disguise_unknown_or_mixed_language(self):
        self.assertEqual(language_code('Korean'),'ko')
        self.assertEqual(language_code('Japanese,English'),'ja,en')
        self.assertEqual(language_code('Thai'),'thai')
        self.assertEqual(language_code('','ko'),'ko')


if __name__=='__main__':
    unittest.main()
