"""Fast pipeline behavior checks; no actual models, network, or video codecs needed."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import pipeline
from translation import LocalTranslator
from video_media import write_srt as real_write_srt


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.sandbox_parent = PROJECT / 'work' / 'pipeline-tests'
        self.sandbox_parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='run-', dir=self.sandbox_parent)
        self.root = Path(self.temporary.name)
        self.input = self.root / 'input' / 'source.mp4'
        self.input.parent.mkdir()
        self.input.write_bytes(b'original source video - do not modify')
        self.config = {"asr_backend": "whisper"}
        for key in ('ffmpeg_path', 'ffprobe_path', 'whisper_path', 'whisper_model', 'llama_path', 'translation_model', 'context_model'):
            target = self.root / 'dependencies' / key
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(b'mock dependency')
            self.config[key] = str(target)
        (self.root / 'config.json').write_text(json.dumps(self.config), encoding='utf-8')
        self.segments = [
            {'id': i + 1, 'start': i * 2.0 + .25, 'end': i * 2.0 + 1.8, 'text': f'source line {i + 1}'}
            for i in range(10)
        ]
        self.has_audio = True
        self.has_video = True
        self.media_cancel = False
        self.transcribe_cancel = False
        self.calls = {'probe': 0, 'extract': 0, 'transcribe': 0, 'export': [], 'translate_ids': [], 'translate_texts': [], 'start': 0, 'start_stages': [], 'analysis': 0, 'close': 0}
        self.written_subtitles = []
        self.events = []
        owner = self

        class FakeCancelledError(Exception):
            pass

        class FakeMedia:
            def __init__(self, config, emit, cancel_event):
                self.cancel = cancel_event

            def probe(self, video):
                owner.calls['probe'] += 1
                if owner.media_cancel:
                    raise FakeCancelledError('media step cancelled')
                return {'has_video': owner.has_video, 'has_audio': owner.has_audio, 'duration': 22}

            def extract_audio(self, video, wav):
                owner.calls['extract'] += 1
                Path(wav).write_bytes(b'fake audio')

            def transcribe(self, wav, language):
                owner.calls['transcribe'] += 1
                if owner.transcribe_cancel:
                    raise FakeCancelledError('speech recognition cancelled')
                return [dict(row) for row in owner.segments], 'en'

            def export(self, source, srt, output, mode):
                owner.calls['export'].append((Path(source), Path(srt), Path(output), mode))
                if Path(source).resolve() == Path(output).resolve():
                    raise AssertionError('Export must not overwrite input')
                Path(output).write_bytes(b'new exported video')

        def fake_write_srt(segments, path, text_key='text'):
            rows = [dict(row) for row in segments]
            owner.written_subtitles.append((Path(path), text_key, rows))
            return real_write_srt(rows, path, text_key)

        class FakeTranslator(LocalTranslator):
            def start(self, model_path=None, stage='translation'):
                self._check_cancel()
                owner.calls['start'] += 1
                owner.calls['start_stages'].append(stage)

            def analyze_context(self, segments, language):
                owner.calls['analysis'] += 1
                return {'summary': 'mocked full-dialogue context', 'tone': 'casual', 'terms': [], 'ambiguities': []}

            def close(self):
                owner.calls['close'] += 1

            def translate_batch(self, segments, language, context=None, context_notes=None):
                owner.calls['translate_ids'].extend(row['id'] for row in segments)
                owner.calls['translate_texts'].extend(row['text'] for row in segments)
                return {row['id']: f'Chinese line {row["id"]}' for row in segments}

        fake_media_module = types.SimpleNamespace(MediaTools=FakeMedia, write_srt=fake_write_srt, CancelledError=FakeCancelledError)
        self.media_patch = mock.patch.dict(sys.modules, {'video_media': fake_media_module})
        self.translator_patch = mock.patch.object(pipeline, 'LocalTranslator', FakeTranslator)
        self.media_patch.start()
        self.translator_patch.start()

    def tearDown(self):
        self.translator_patch.stop()
        self.media_patch.stop()
        # Verify the deletion target before cleaning our isolated test directory.
        self.assertTrue(self.root.resolve().is_relative_to(self.sandbox_parent.resolve()))
        self.assertNotEqual(self.root.resolve(), self.sandbox_parent.resolve())
        self.temporary.cleanup()

    def run_job(self, mode='srt', bilingual=False, emit=None, cancel=None, **kwargs):
        return pipeline.Pipeline(self.root, emit or self.events.append, cancel).run(
            str(self.input), language='en', mode=mode, bilingual=bilingual, **kwargs)

    def read_manifest(self):
        manifests = list((self.root / 'outputs').glob('*/*.json'))
        return next(json.loads(path.read_text(encoding='utf-8')) for path in manifests if 'status' in json.loads(path.read_text(encoding='utf-8')))

    def test_complete_resume_reuses_recognition_and_translation(self):
        first = self.run_job()
        second = self.run_job(mode='soft', bilingual=True)
        self.assertEqual(first['output_dir'], second['output_dir'])
        self.assertEqual(self.calls['extract'], 1)
        self.assertEqual(self.calls['transcribe'], 1)
        self.assertEqual(self.calls['start'], 2)
        self.assertEqual(self.calls['start_stages'], ['context', 'translation'])
        self.assertEqual(self.calls['analysis'], 1)
        self.assertTrue(Path(first['output_dir'], '翻译背景备注.json').is_file())
        self.assertEqual(self.calls['translate_ids'], list(range(1, 11)))
        self.assertTrue(Path(second['video']).is_file())
        self.assertEqual(self.calls['export'][0][1], Path(second['bilingual_srt']))
        self.assertEqual(self.read_manifest()['status'], 'complete')

    def test_cancellation_preserves_completed_batch_and_resumes_remaining(self):
        cancel = threading.Event()
        def emit(event):
            self.events.append(event)
            if event.get('type') == 'progress' and 75 < event.get('value', 0) < 76:
                cancel.set()
        with self.assertRaises(InterruptedError):
            self.run_job(emit=emit, cancel=cancel)
        self.assertEqual(self.read_manifest()['status'], 'cancelled')
        cached = json.loads(next((self.root / 'work' / 'jobs').glob('*/translations.json')).read_text(encoding='utf-8'))
        self.assertEqual(len(cached['items']), 8)
        self.assertGreaterEqual(self.calls['close'], 1)
        self.assertEqual(self.calls['translate_ids'], list(range(1, 9)))
        result = self.run_job()
        self.assertEqual(self.calls['transcribe'], 1)
        self.assertEqual(self.calls['analysis'], 1)
        self.assertEqual(self.calls['start'], 3)
        self.assertEqual(self.calls['translate_ids'], list(range(1, 11)))
        self.assertTrue(Path(result['zh_srt']).is_file())
        self.assertEqual(self.read_manifest()['status'], 'complete')

    def test_cancelled_recognition_reuses_complete_audio(self):
        self.transcribe_cancel = True
        with self.assertRaises(InterruptedError):
            self.run_job()
        self.assertEqual(self.calls['extract'], 1)
        self.assertEqual(self.calls['transcribe'], 1)
        self.transcribe_cancel = False
        self.run_job()
        self.assertEqual(self.calls['extract'], 1)
        self.assertEqual(self.calls['transcribe'], 2)
        self.assertEqual(self.read_manifest()['status'], 'complete')

    def test_pre_cancel_does_not_create_outputs_or_start_media(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            self.run_job(cancel=cancel)
        self.assertFalse((self.root / 'outputs').exists())
        self.assertEqual(self.calls['probe'], 0)

    def test_media_cancel_is_converted_to_ui_cancellation(self):
        self.media_cancel = True
        with self.assertRaises(InterruptedError):
            self.run_job()
        self.assertEqual(self.read_manifest()['status'], 'cancelled')
        self.assertEqual(self.calls['start'], 0)

    def test_no_audio_is_rejected_before_extract_or_translate(self):
        self.has_audio = False
        with self.assertRaises(ValueError):
            self.run_job()
        self.assertEqual(self.calls['extract'], 0)
        self.assertEqual(self.calls['start'], 0)
        self.assertEqual(self.read_manifest()['status'], 'failed')

    def test_empty_recognition_is_rejected(self):
        self.segments = []
        with self.assertRaises(ValueError):
            self.run_job()
        self.assertEqual(self.calls['start'], 0)
        self.assertEqual(self.read_manifest()['status'], 'failed')

    def test_export_never_modifies_source_and_timestamps_are_preserved(self):
        original = self.input.read_bytes()
        result = self.run_job(mode='burn')
        self.assertEqual(self.input.read_bytes(), original)
        self.assertNotEqual(Path(result['video']).resolve(), self.input.resolve())
        self.assertTrue(Path(result['video']).resolve().is_relative_to((self.root / 'outputs').resolve()))
        chinese = next(rows for path, key, rows in self.written_subtitles if key == 'zh')
        self.assertEqual([(s['id'], s['start'], s['end']) for s in chinese], [(s['id'], s['start'], s['end']) for s in self.segments])
        self.assertEqual(len(chinese), 10)

    def test_model_change_invalidates_translation_but_reuses_recognition(self):
        self.run_job()
        Path(self.config['translation_model']).write_bytes(b'a newer translation model')
        self.run_job()
        self.assertEqual(self.calls['transcribe'], 1)
        self.assertEqual(self.calls['start'], 3)
        self.assertEqual(self.calls['translate_ids'], list(range(1, 11)) * 2)

    def test_force_retranscribe_bypasses_both_recognition_and_translation(self):
        first = self.run_job()
        self.assertEqual(self.calls['transcribe'], 1)
        self.assertEqual(self.calls['start'], 2)
        # Even an identical second transcription must not reuse old translations.
        second = self.run_job(force_retranscribe=True)
        self.assertEqual(second['output_dir'], first['output_dir'])
        self.assertEqual(self.calls['extract'], 1)
        self.assertEqual(self.calls['transcribe'], 2)
        self.assertEqual(self.calls['start'], 3)
        self.assertEqual(self.calls['translate_ids'], list(range(1, 11)) * 2)
        self.assertEqual(self.read_manifest()['status'], 'complete')

    def test_changed_recognition_text_with_same_ids_invalidates_translation(self):
        self.run_job()
        state_path = next((self.root / 'work' / 'jobs').glob('*/recognition.json'))
        translation_path = state_path.with_name('translations.json')
        old_key = json.loads(translation_path.read_text(encoding='utf-8'))['model']
        # Simulate lost recognition checkpoint and corrected ASR on a retry,
        # retaining the previous translations and exactly the same segment IDs.
        state_path.unlink()
        self.segments[0]['text'] = 'a corrected sentence that was missing before'
        second = self.run_job()
        new_key = json.loads(translation_path.read_text(encoding='utf-8'))['model']
        self.assertNotEqual(new_key, old_key)
        self.assertEqual(self.calls['transcribe'], 2)
        self.assertEqual(self.calls['start'], 4)
        self.assertEqual(self.calls['translate_ids'], list(range(1, 11)) * 2)
        self.assertIn(self.segments[0]['text'], self.calls['translate_texts'][10:])
        self.assertIn(self.segments[0]['text'], Path(second['source_srt']).read_text(encoding='utf-8'))

    def test_switching_asr_engine_changes_recognition_fingerprint(self):
        runner = pipeline.Pipeline(self.root)
        runner.config['asr_backend'] = 'whisper'
        whisper_fingerprint = runner._recognition_fingerprint()
        for key in ('qwen_asr_model', 'qwen_aligner_model'):
            folder = self.root / 'dependencies' / key
            folder.mkdir()
            (folder / 'config.json').write_text('{}', encoding='utf-8')
            (folder / 'model.safetensors').write_bytes(b'isolated mock model metadata')
            runner.config[key] = str(folder)
        worker = self.root / 'qwen_asr_worker.py'
        worker.write_text('# isolated mock worker', encoding='utf-8')
        runner.config['asr_backend'] = 'qwen3'
        qwen_fingerprint = runner._recognition_fingerprint()
        self.assertNotEqual(whisper_fingerprint, qwen_fingerprint)
        self.assertEqual(whisper_fingerprint['backend'], 'whisper')
        self.assertEqual(qwen_fingerprint['backend'], 'qwen3')
        self.assertIn(str(worker), qwen_fingerprint['files'])
        self.assertNotIn(self.config['whisper_model'], qwen_fingerprint['files'])
        self.assertNotIn('whisper_model', qwen_fingerprint['settings'])
        # Unchanged engine and files produce a stable fingerprint for resuming.
        self.assertEqual(qwen_fingerprint, runner._recognition_fingerprint())

    def test_environment_reports_exact_missing_path(self):
        missing = Path(self.config['ffmpeg_path'])
        missing.unlink()
        issues = pipeline.Pipeline(self.root).check_environment()
        self.assertEqual(len(issues), 1)
        self.assertIn(str(missing), issues[0])


if __name__ == '__main__':
    unittest.main()
