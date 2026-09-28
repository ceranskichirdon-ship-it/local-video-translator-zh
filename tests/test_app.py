"""Desktop settings checks using Tk widgets and a mocked pipeline."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import tkinter as tk
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from app import ASR_BACKENDS, LANGUAGES, LANGUAGE_BACKENDS, MODES, VideoTranslatorApp


def backend_label(value):
    return next(label for label, code in ASR_BACKENDS.items() if code == value)


class DesktopSettingsTests(unittest.TestCase):
    def setUp(self):
        self.test_parent = PROJECT / 'work' / 'ui-tests'
        self.test_parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='settings-', dir=self.test_parent)
        self.root = Path(self.temporary.name)
        self.window = tk.Tk()
        self.window.withdraw()
        self.app = None

    def tearDown(self):
        if self.app:
            self.app.closing = True
        for callback in self.window.tk.call('after', 'info'):
            self.window.after_cancel(callback)
        self.window.destroy()
        self.assertTrue(self.root.resolve().is_relative_to(self.test_parent.resolve()))
        self.assertNotEqual(self.root.resolve(), self.test_parent.resolve())
        self.temporary.cleanup()

    def build(self):
        self.app = VideoTranslatorApp(self.window, self.root, check_on_start=False)
        self.window.update_idletasks()
        return self.app

    def wait_until(self, condition):
        deadline = time.monotonic() + 3
        while not condition() and time.monotonic() < deadline:
            self.window.update()
            time.sleep(.01)
        self.assertTrue(condition(), 'UI worker did not finish in time')

    def select_input(self):
        source = self.root / 'test-video.mp4'
        source.write_bytes(b'ui-only mock input')
        self.app._add_videos([(source, source.stat().st_size)])
        self.app.environment_ready = True
        self.app._refresh_controls()
        return source

    def test_missing_configuration_defaults_to_whisper_and_cache_reuse(self):
        app = self.build()
        self.assertEqual(ASR_BACKENDS[app.asr_backend.get()], 'whisper')
        self.assertFalse(app.force_retranscribe.get())
        self.assertEqual(LANGUAGES[app.language.get()], '')
        self.assertEqual(str(app.start_button.cget('state')), 'disabled')
        self.assertEqual(MODES[app.mode.get()], 'burn')
        self.assertEqual(str(app.backend_box.cget('state')), 'readonly')

    def test_language_is_required_and_selects_a_matching_asr_model(self):
        app = self.build()
        self.select_input()
        with mock.patch('app.messagebox.showinfo') as info:
            app._start()
        info.assert_called_once()
        requested = []

        class FakePipeline:
            def __init__(self, root, emit, cancel):
                pass
            def check_environment(self, asr_backend=None):
                requested.append(asr_backend)
                return []

        with mock.patch.dict(sys.modules, {'pipeline': types.SimpleNamespace(Pipeline=FakePipeline)}):
            for label, language in [('英语', 'en'), ('日语', 'ja'), ('韩语', 'ko')]:
                app.language.set(label)
                app._language_changed()
                self.wait_until(lambda: not app.checking)
                expected = LANGUAGE_BACKENDS[language]
                self.assertEqual(ASR_BACKENDS[app.asr_backend.get()], expected)
                self.assertEqual(str(app.start_button.cget('state')), 'normal')
        self.assertEqual(requested, ['whisper', 'whisper_precise', 'whisper'])

    def test_saved_whisper_configuration_is_respected(self):
        (self.root / 'config.json').write_text(json.dumps({'asr_backend': 'whisper'}), encoding='utf-8')
        app = self.build()
        self.assertEqual(ASR_BACKENDS[app.asr_backend.get()], 'whisper')
        self.assertFalse(app.force_retranscribe.get())

    def test_engine_selection_rechecks_selected_environment_and_locks_start(self):
        app = self.build()
        self.select_input()
        app.language.set('英语')
        requested = []
        release = threading.Event()

        class FakePipeline:
            def __init__(self, root, emit, cancel):
                pass
            def check_environment(self, asr_backend=None):
                requested.append(asr_backend)
                release.wait(2)
                return []

        with mock.patch.dict(sys.modules, {'pipeline': types.SimpleNamespace(Pipeline=FakePipeline)}):
            app.asr_backend.set(backend_label('whisper'))
            app.backend_box.event_generate('<<ComboboxSelected>>')
            self.assertTrue(app.checking)
            self.assertFalse(app.environment_ready)
            self.assertEqual(str(app.start_button.cget('state')), 'disabled')
            self.assertEqual(str(app.backend_box.cget('state')), 'disabled')
            release.set()
            self.wait_until(lambda: not app.checking)
        self.assertEqual(requested, ['whisper'])
        self.assertTrue(app.environment_ready)
        self.assertEqual(str(app.start_button.cget('state')), 'normal')
        self.assertEqual(str(app.backend_box.cget('state')), 'readonly')

    def test_start_passes_engine_and_force_flag_to_background_pipeline(self):
        app = self.build()
        source = self.select_input()
        app.language.set('英语')
        app.asr_backend.set(backend_label('whisper'))
        app.force_retranscribe.set(True)
        requested = []
        release = threading.Event()

        class FakePipeline:
            def __init__(self, root, emit, cancel):
                pass
            def run(self, video, **kwargs):
                requested.append((video, kwargs))
                release.wait(2)
                return {'output_dir': str(source.parent / 'outputs'), 'zh_srt': str(source.parent / 'result.srt')}

        with mock.patch.dict(sys.modules, {'pipeline': types.SimpleNamespace(Pipeline=FakePipeline)}):
            app._start()
            self.assertTrue(app.busy)
            self.assertEqual(str(app.backend_box.cget('state')), 'disabled')
            self.assertEqual(str(app.force_check.cget('state')), 'disabled')
            release.set()
            self.wait_until(lambda: not app.busy)
        self.assertEqual(requested, [(str(source), {'language': 'en', 'mode': 'burn', 'bilingual': False, 'asr_backend': 'whisper', 'force_retranscribe': True})])
        self.assertEqual(str(app.backend_box.cget('state')), 'readonly')
        self.assertEqual(str(app.force_check.cget('state')), 'normal')
        self.assertEqual(app.progress['value'], 100)


if __name__ == '__main__':
    unittest.main()
