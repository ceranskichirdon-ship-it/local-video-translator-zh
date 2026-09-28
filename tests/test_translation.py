"""Translation validation and batching tests; all model responses are mocked."""
from __future__ import annotations

import copy
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from translation import LocalTranslator


def completion(content, finish_reason='stop'):
    return {'choices': [{'finish_reason': finish_reason, 'message': {'content': content}}]}


def translation_response(rows):
    return completion(json.dumps({'translations': rows}, ensure_ascii=False))


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.translator = LocalTranslator({}, self.events.append, threading.Event(), PROJECT / 'work')

    def test_duplicate_missing_and_unexpected_ids_are_rejected(self):
        segments = [
            {'id': 7, 'start': 1.25, 'end': 2.0, 'text': 'Hello'},
            {'id': 11, 'start': 2.4, 'end': 3.0, 'text': 'Goodbye'},
        ]
        cases = {
            'duplicate': [{'id': 7, 'text': '你好'}, {'id': 7, 'text': '再见'}],
            'missing': [{'id': 7, 'text': '你好'}],
            'unexpected': [{'id': 7, 'text': '你好'}, {'id': 99, 'text': '再见'}],
        }
        for name, rows in cases.items():
            with self.subTest(name=name), mock.patch.object(self.translator, '_completion', return_value=translation_response(rows)) as call:
                with self.assertRaises(ValueError):
                    self.translator.translate_batch(segments, 'en')
                self.assertEqual(call.call_count, 1)

    def test_untranslated_ordinary_english_phrase_gets_chinese_second_pass(self):
        segments = [{'id': 4, 'start': 4.5, 'end': 6.0, 'text': 'He said good morning to me.'}]
        first = translation_response([{'id': 4, 'text': '他向我说 good morning。'}])
        second = completion('“早上好”')
        with mock.patch.object(self.translator, '_completion', side_effect=[first, second]) as call:
            result = self.translator.translate_batch(segments, 'en')
        self.assertEqual(result, {4: '他向我说 早上好。'})
        self.assertEqual(call.call_count, 2)
        fix_payload = call.call_args_list[1].args[0]
        repair_input = json.loads(fix_payload['messages'][1]['content'])
        self.assertEqual(repair_input['phrase'], 'good morning')
        self.assertEqual(repair_input['source_context'], segments[0]['text'])

    def test_single_acronym_is_not_sent_for_phrase_correction(self):
        segments = [{'id': 1, 'text': 'Please use USB.', 'start': 0, 'end': 1}]
        with mock.patch.object(self.translator, '_completion', return_value=translation_response([{'id': 1, 'text': '请使用 USB。'}])) as call:
            result = self.translator.translate_batch(segments, 'en')
        self.assertEqual(result[1], '请使用 USB。')
        self.assertEqual(call.call_count, 1)

    def test_invalid_second_pass_rejects_untranslated_english(self):
        segments = [{'id': 3, 'text': 'Say good morning.', 'start': 0, 'end': 2}]
        original = '说 good morning。'
        for repair in ('still English', '你好' * 100, ''):
            with self.subTest(repair=repair), mock.patch.object(self.translator, '_completion', side_effect=[translation_response([{'id': 3, 'text': original}]), completion(repair)]):
                with self.assertRaisesRegex(ValueError, '未翻译的英语短语'):
                    self.translator.translate_batch(segments, 'en')

    def test_japanese_or_korean_residue_and_cleaned_blank_are_rejected(self):
        cases = [
            ('ja', 'こんにちは', '你好、こんにちは'),
            ('ja', 'ああ', '阿ー'),
            ('ko', '안녕하세요', '你好、안녕하세요'),
            ('en', 'me', 'me'),
            ('ja', 'どうですか', '}'),
        ]
        for language, source, output in cases:
            segment = {'id': 1, 'start': 0, 'end': 2, 'text': source}
            with self.subTest(source=source, output=output), mock.patch.object(
                    self.translator, '_completion', return_value=translation_response([{'id': 1, 'text': output}])):
                with self.assertRaises(ValueError):
                    self.translator.translate_batch([segment], language)

    def test_invalid_cached_japanese_cue_is_translated_again(self):
        segments = [{'id': 1, 'start': 0, 'end': 2, 'text': 'おはようございます'}]
        cache = {'1': 'おはようございます'}
        saved = []
        with mock.patch.object(self.translator, '_completion',
                               return_value=translation_response([{'id': 1, 'text': '早上好。'}])) as call:
            result = self.translator.translate(segments, 'ja', cache, lambda items: saved.append(dict(items)))
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result[0]['zh'], '早上好。')
        self.assertEqual(saved[0], {})
        self.assertEqual(saved[-1], {'1': '早上好。'})

    def test_contextual_japanese_copy_gets_direct_then_strict_retry(self):
        segments = [{'id': 1, 'start': 0, 'end': 2, 'text': 'おはようございます'}]
        first = completion(json.dumps({'id': 1, 'text': 'おはようございます'}, ensure_ascii=False))
        second = translation_response([{'id': 1, 'text': 'おはようございます'}])
        third = translation_response([{'id': 1, 'text': '早上好。'}])
        with mock.patch.object(self.translator, '_completion', side_effect=[first, second, third]) as call:
            result = self.translator.translate(segments, 'ja', {}, lambda _: None, context_notes={})
        self.assertEqual(result[0]['zh'], '早上好。')
        self.assertEqual(call.call_count, 3)
        self.assertIn('这条字幕上次未译完整', call.call_args_list[2].args[0]['messages'][0]['content'])

    def test_japanese_parent_term_is_not_reversed(self):
        segment = {'id': 1, 'start': 0, 'end': 2, 'text': 'パパはどうする？'}
        with mock.patch.object(self.translator, '_completion',
                               return_value=translation_response([{'id': 1, 'text': '妈妈怎么办？'}])):
            with self.assertRaisesRegex(ValueError, '父亲误写成母亲'):
                self.translator.translate_batch([segment], 'ja')

    def test_truncated_response_is_rejected(self):
        segments = [{'id': 1, 'text': 'Hello.', 'start': 0, 'end': 1}]
        response = completion('{"translations":', finish_reason='length')
        with mock.patch.object(self.translator, '_completion', return_value=response):
            with self.assertRaises(ValueError):
                self.translator.translate_batch(segments, 'en')

    def test_contextual_translation_removes_leaked_json_closer_from_subtitle(self):
        segments = [{'id': 2, 'start': 1, 'end': 2, 'text': '헤어지자고?'}]
        response = completion(json.dumps({'id': 2, 'text': '什么？要分手？”}'}, ensure_ascii=False))
        with mock.patch.object(self.translator, '_completion', return_value=response):
            result = self.translator.translate_batch(segments, 'ko', context_notes={})
        self.assertEqual(result[2], '什么？要分手？')

    def test_batch_limits_preserve_all_ids_timestamps_and_source_text(self):
        # Text budget is 2,800 source characters; a single longer cue remains intact.
        cases = {'count_limit': [10] * 19, 'character_limit_and_large_cue': [1400, 1400, 1, 3001, 1]}
        for name, lengths in cases.items():
            with self.subTest(name=name):
                segments = [
                    {'id': 20 + i * 3, 'start': 1.25 + i * 2.0, 'end': 2.9 + i * 2.0, 'text': 'あ' * length}
                    for i, length in enumerate(lengths)
                ]
                original = copy.deepcopy(segments)
                requested = []
                saved = []
                cache = {}

                def respond(payload):
                    rows = json.loads(payload['messages'][1]['content'])['subtitles']
                    requested.append(rows)
                    # Deliberately reverse model order: IDs, rather than response
                    # positions, must attach translations to original timestamps.
                    return translation_response([{'id': row['id'], 'text': f'译文{row["id"]}'} for row in reversed(rows)])

                with mock.patch.object(self.translator, '_completion', side_effect=respond):
                    result = self.translator.translate(segments, 'ja', cache, lambda state: saved.append(dict(state)))
                self.assertEqual(segments, original)
                self.assertEqual([row['id'] for batch in requested for row in batch], [row['id'] for row in original])
                for batch in requested:
                    self.assertLessEqual(len(batch), 8)
                    self.assertTrue(sum(len(row['text']) for row in batch) <= 2800 or len(batch) == 1)
                self.assertEqual([{k: row[k] for k in ('id', 'start', 'end', 'text')} for row in result], original)
                self.assertEqual([row['zh'] for row in result], [f'译文{row["id"]}' for row in original])
                self.assertEqual(len(saved[-1]), len(original))
                if name == 'count_limit':
                    self.assertEqual([len(batch) for batch in requested], [8, 8, 3])
                else:
                    self.assertEqual([sum(len(row['text']) for row in batch) for batch in requested], [2800, 1, 3001, 1])


    def test_context_analysis_reads_the_full_transcript_and_returns_cautious_notes(self):
        segments = [{'id': 1, 'text': '아팠어?'}, {'id': 2, 'text': '다른 여자 생겼다고?'}]
        notes = {'summary': 'Friends discuss a breakup.', 'tone': 'casual', 'terms': [],
                 'ambiguities': ['A phrase may contain an ASR error.']}
        payload = completion(json.dumps(notes, ensure_ascii=False))
        with mock.patch.object(self.translator, '_completion', return_value=payload) as call:
            result = self.translator.analyze_context(segments, 'ko')
        self.assertEqual(result, notes)
        submitted = json.loads(call.call_args.args[0]['messages'][1]['content'])
        self.assertEqual(submitted['complete_transcript'], segments)
        self.assertIn('禁止擅自改写原文', call.call_args.args[0]['messages'][0]['content'])

    def test_context_analysis_batches_long_transcripts_before_output_limit(self):
        segments = [{'id': i, 'text': f'日本語の字幕{i}です。'} for i in range(1, 126)]
        requests = []
        def respond(payload):
            body = json.loads(payload['messages'][1]['content'])
            requests.append((body, payload))
            notes = {'summary': '字幕から確認できる会話。', 'tone': '会話調', 'terms': [], 'ambiguities': []}
            return completion(json.dumps(notes, ensure_ascii=False))
        with mock.patch.object(self.translator, '_completion', side_effect=respond):
            result = self.translator.analyze_context(segments, 'ja')
        self.assertEqual([len(body['complete_transcript']) for body, _ in requests[:4]], [40, 40, 40, 5])
        self.assertIn('合并整部视频', requests[4][0]['task'])
        self.assertTrue(all(payload['max_tokens'] == 900 for _, payload in requests))
        self.assertEqual(result['parts_reviewed'], 4)

    def test_contextual_translation_uses_notes_and_preserves_each_id(self):
        segments = [
            {'id': 1, 'start': 0, 'end': 1, 'text': '헤어지자고?'},
            {'id': 2, 'start': 2, 'end': 3, 'text': '아팠어?'},
            {'id': 3, 'start': 4, 'end': 5, 'text': '다른 여자 생겼다고?'},
        ]
        notes = {'summary': 'Friends discuss a breakup.', 'tone': 'teasing', 'terms': [], 'ambiguities': []}
        requested, saved, cache = [], [], {}
        def respond(payload):
            body = json.loads(payload['messages'][1]['content'])
            requested.append((body, payload))
            row = body['subtitles'][0]
            return completion(json.dumps({'id': row['id'], 'text': f'译文{row["id"]}'}, ensure_ascii=False))
        with mock.patch.object(self.translator, '_completion', side_effect=respond):
            translated = self.translator.translate(segments, 'ko', cache, lambda state: saved.append(dict(state)), context_notes=notes)
        self.assertEqual([row['zh'] for row in translated], ['译文1', '译文2', '译文3'])
        self.assertEqual(len(requested), 3)
        self.assertEqual(requested[0][0]['whole_video_notes'], notes)
        self.assertEqual(requested[1][0]['nearby_subtitles']['before'][0]['translation'], '译文1')
        self.assertEqual(requested[1][0]['nearby_subtitles']['after'][0]['text'], segments[2]['text'])
        self.assertEqual(requested[0][1]['top_k'], 20)
        self.assertEqual(requested[0][1]['response_format']['json_schema']['name'], 'translation')
        self.assertEqual(len(saved), 3)


if __name__ == '__main__':
    unittest.main()
