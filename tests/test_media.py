import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_media import CancelledError, MediaError, MediaTools, parse_whisper_json, srt_timestamp, write_srt

ROOT = Path(__file__).resolve().parents[1]


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'work').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='test_media_', dir=ROOT / 'work')
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_millisecond_rounding_carries_into_hour(self):
        self.assertEqual(srt_timestamp(3599.9995), '01:00:00,000')
        self.assertEqual(srt_timestamp(0.0005), '00:00:00,001')
        self.assertEqual(srt_timestamp(360_001.125), '100:00:01,125')
        for invalid in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                srt_timestamp(invalid)

    def test_chinese_srt_preserves_timestamps_and_line_breaks(self):
        out = self.folder / '中文字幕.srt'
        write_srt([{'start': 61.234, 'end': 62.5, 'zh': '第一行\r\n\r\n第二行'},
                   {'start': 63.0, 'end': 64.999, 'zh': '你好，世界。'}], out, text_key='zh')
        text = out.read_text(encoding='utf-8')
        self.assertEqual(text, '1\n00:01:01,234 --> 00:01:02,500\n第一行\n第二行\n\n'
                         '2\n00:01:03,000 --> 00:01:04,999\n你好，世界。\n\n')

    def test_existing_output_never_replaced(self):
        out = self.folder / 'existing.srt'
        out.write_text('keep me', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            write_srt([{'start': 0, 'end': 1, 'text': 'new'}], out)
        self.assertEqual(out.read_text(), 'keep me')

    def test_invalid_subtitle_does_not_create_partial_file(self):
        out = self.folder / 'invalid.srt'
        with self.assertRaises(ValueError):
            write_srt([{'start': 1, 'end': 0, 'text': 'invalid'}], out)
        self.assertFalse(out.exists())

    def test_whisper_offsets_are_milliseconds_and_detected_language_wins(self):
        data = {'params': {'language': 'auto'}, 'result': {'language': 'ja'}, 'transcription': [
            {'offsets': {'from': 1230, 'to': 3570}, 'text': '  こんにちは  '},
            {'offsets': {'from': 4000, 'to': 9000}, 'text': '次の文です。'}]}
        segments, language = parse_whisper_json(data)
        self.assertEqual(language, 'ja')
        self.assertEqual(segments, [
            {'id': 1, 'start': 1.23, 'end': 3.57, 'text': 'こんにちは'},
            {'id': 2, 'start': 4.0, 'end': 9.0, 'text': '次の文です。'}])

    def test_whisper_timestamp_fallback_and_blank_audio(self):
        data = {'result': {'language': 'ko'}, 'transcription': [
            {'timestamps': {'from': '01:02:03,450', 'to': '01:02:05,600'}, 'text': '안녕하세요'},
            {'offsets': {'from': 0, 'to': 3000}, 'text': '[BLANK_AUDIO]'}]}
        segments, language = parse_whisper_json(data)
        self.assertEqual(segments[0]['start'], 3723.45)
        self.assertEqual(len(segments), 1)
        self.assertEqual(language, 'ko')
        with self.assertRaisesRegex(MediaError, '没有识别到'):
            parse_whisper_json({'transcription': [data['transcription'][1]]})

    def test_empty_or_invalid_transcription_is_actionable(self):
        with self.assertRaises(MediaError):
            parse_whisper_json({})
        with self.assertRaises(MediaError):
            parse_whisper_json({'transcription': []})
        with self.assertRaises(MediaError):
            parse_whisper_json({'transcription': [{'text': 'bad', 'offsets': {'from': 'x', 'to': 10}}]})


class ProcessTests(unittest.TestCase):
    def test_cancel_interrupts_process_even_without_output(self):
        cancelled = threading.Event()
        backend = MediaTools({'base_dir': str(ROOT)}, cancel_event=cancelled)
        timer = threading.Timer(0.3, cancelled.set)
        started = time.monotonic()
        timer.start()
        try:
            with self.assertRaises(CancelledError):
                backend._run([sys.executable, '-c', 'import time; time.sleep(20)'], 'test')
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 5)

    def test_failure_reports_tool_exit_code_and_error(self):
        backend = MediaTools({'base_dir': str(ROOT)})
        with self.assertRaisesRegex(MediaError, 'exit-message'):
            backend._run([sys.executable, '-c', 'import sys; print("exit-message"); sys.exit(7)'], 'test')

    def test_no_audio_rejected_before_extraction(self):
        backend = MediaTools({'base_dir': str(ROOT)})
        with patch.object(backend, 'probe', return_value={'has_audio': False}):
            with self.assertRaisesRegex(MediaError, '没有音轨'):
                backend.extract_audio(ROOT / 'work' / 'source-test.mp4', ROOT / 'work' / 'missing-out.wav')

    def test_export_rejects_original_even_when_hardlinked(self):
        temp = tempfile.TemporaryDirectory(prefix='test_export_', dir=ROOT / 'work')
        try:
            source = Path(temp.name) / 'input.mp4'
            source.write_bytes(b'original')
            subtitle = Path(temp.name) / 'subtitle.srt'
            subtitle.write_text('1\n00:00:00,000 --> 00:00:01,000\n中文\n', encoding='utf-8')
            backend = MediaTools({'base_dir': str(ROOT)})
            with self.assertRaisesRegex(MediaError, '不能与原视频相同'):
                backend.export(source, subtitle, source)
            self.assertEqual(source.read_bytes(), b'original')
        finally:
            temp.cleanup()


if __name__ == '__main__':
    unittest.main()
