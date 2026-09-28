from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = frozenset({'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v', '.wmv', '.flv', '.mpg', '.mpeg', '.ts', '.mts', '.m2ts', '.3gp'})
LANGUAGES = {'请选择视频语言': '', '英语': 'en', '日语': 'ja', '韩语': 'ko'}
ASR_BACKENDS = {'Whisper large-v3 + 分段校时 · 日语推荐': 'whisper_precise',
                'Whisper large-v3 · 英语/韩语': 'whisper',
                'Kotoba-Whisper v2.1 · 日语对照': 'kotoba',
                'Qwen3-ASR 1.7B · 多语种对照': 'qwen3'}
LANGUAGE_BACKENDS = {'en': 'whisper', 'ja': 'whisper_precise', 'ko': 'whisper'}
MODES = {'烧录中文字幕 · MP4（推荐）': 'burn', '可开关中文字幕 · MKV': 'soft', '只生成字幕 · SRT': 'srt'}


def format_size(size: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024 or unit == 'TB':
            return f'{size:.1f} {unit}'
        size /= 1024
    return str(size)


class VideoTranslatorApp:
    def __init__(self, window: tk.Tk, root: Path = ROOT, check_on_start: bool = True):
        self.window = window
        self.root = Path(root)
        self.events: queue.Queue[dict] = queue.Queue()
        self.cancel_event = threading.Event()
        self.scan_cancel = threading.Event()
        self.busy = False
        self.scanning = False
        self.checking = False
        self.environment_ready = False
        self.closing = False
        self.last_result: dict = {}
        self.videos: dict[str, Path] = {}
        self.known_paths: set[str] = set()
        self.item_counter = 0
        self.folder = self.root
        try:
            config = json.loads((self.root / 'config.json').read_text(encoding='utf-8-sig'))
            configured_backend = config.get('asr_backend') or 'whisper'
        except (OSError, ValueError, AttributeError):
            configured_backend = 'whisper'
        backend_label = next((label for label, value in ASR_BACKENDS.items() if value == configured_backend), next(iter(ASR_BACKENDS)))
        self.asr_backend = tk.StringVar(value=backend_label)
        self.force_retranscribe = tk.BooleanVar(value=False)
        self.language = tk.StringVar(value=next(iter(LANGUAGES)))
        self.mode = tk.StringVar(value=next(iter(MODES)))
        self.bilingual = tk.BooleanVar(value=False)
        self.recursive = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='请选择本地视频，或扫描视频文件夹。')
        self.selected_path = tk.StringVar(value='尚未选择视频')
        self.environment_status = tk.StringVar(value='正在检查运行环境……')
        self.count_text = tk.StringVar(value='0 个视频')
        self._build()
        self.window.protocol('WM_DELETE_WINDOW', self._on_close)
        self.window.after(80, self._drain_events)
        if check_on_start:
            self.window.after(150, self._check_environment)

    def _build(self):
        self.window.title('本地视频翻译 · 中文字幕')
        self.window.geometry('1060x760')
        self.window.minsize(820, 620)
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill='both', expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=3)
        outer.rowconfigure(7, weight=2)

        ttk.Label(outer, text='本地视频 → 识别语音 → 翻译中文 → 生成字幕', font=('Microsoft YaHei UI', 12, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 10))
        source = ttk.Frame(outer)
        source.grid(row=1, column=0, sticky='ew', pady=(0, 6))
        self.file_button = ttk.Button(source, text='添加视频…', command=self._choose_files)
        self.file_button.pack(side='left')
        self.folder_button = ttk.Button(source, text='扫描文件夹…', command=self._choose_folder)
        self.folder_button.pack(side='left', padx=6)
        ttk.Checkbutton(source, text='包含子文件夹', variable=self.recursive).pack(side='left', padx=5)
        self.stop_scan_button = ttk.Button(source, text='停止扫描', command=self._stop_scan, state='disabled')
        self.stop_scan_button.pack(side='left', padx=6)
        self.clear_button = ttk.Button(source, text='清空列表', command=self._clear_list)
        self.clear_button.pack(side='left')
        ttk.Label(source, textvariable=self.count_text).pack(side='right')

        video_frame = ttk.Frame(outer)
        video_frame.grid(row=2, column=0, sticky='nsew')
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(video_frame, columns=('name', 'size', 'path'), show='headings', selectmode='browse', height=8)
        for key, title, width in (('name', '视频文件', 260), ('size', '大小', 90), ('path', '所在位置', 590)):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, minwidth=65, stretch=key != 'size', anchor='w')
        self.tree.grid(row=0, column=0, sticky='nsew')
        scroll_y = ttk.Scrollbar(video_frame, orient='vertical', command=self.tree.yview)
        scroll_y.grid(row=0, column=1, sticky='ns')
        scroll_x = ttk.Scrollbar(video_frame, orient='horizontal', command=self.tree.xview)
        scroll_x.grid(row=1, column=0, sticky='ew')
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self.tree.bind('<<TreeviewSelect>>', self._selection_changed)
        self.tree.bind('<Double-1>', lambda _event: self._start())

        ttk.Label(outer, textvariable=self.selected_path, wraplength=990).grid(row=3, column=0, sticky='w', pady=(5, 9))
        options = ttk.LabelFrame(outer, text='翻译设置', padding=9)
        options.grid(row=4, column=0, sticky='ew')
        options.columnconfigure(3, weight=1)
        ttk.Label(options, text='识别引擎').grid(row=0, column=0, padx=(0, 6), sticky='w')
        self.backend_box = ttk.Combobox(options, textvariable=self.asr_backend, values=list(ASR_BACKENDS), state='readonly', width=49)
        self.backend_box.grid(row=0, column=1, columnspan=3, sticky='w', pady=(0, 8))
        self.backend_box.bind('<<ComboboxSelected>>', self._backend_changed)
        ttk.Label(options, text='原视频语言（必选）').grid(row=1, column=0, padx=(0, 6), sticky='w')
        self.language_box = ttk.Combobox(options, textvariable=self.language, values=list(LANGUAGES), state='readonly', width=25)
        self.language_box.grid(row=1, column=1, padx=(0, 14), sticky='w')
        self.language_box.bind('<<ComboboxSelected>>', self._language_changed)
        ttk.Label(options, text='输出方式').grid(row=1, column=2, padx=(0, 6), sticky='w')
        self.mode_box = ttk.Combobox(options, textvariable=self.mode, values=list(MODES), state='readonly', width=31)
        self.mode_box.grid(row=1, column=3, sticky='w')
        self.bilingual_check = ttk.Checkbutton(options, text='显示双语字幕（中文 + 原文）', variable=self.bilingual)
        self.bilingual_check.grid(row=2, column=0, columnspan=2, sticky='w', pady=(8, 0))
        self.force_check = ttk.Checkbutton(options, text='重新识别（忽略旧识别和译文）', variable=self.force_retranscribe)
        self.force_check.grid(row=2, column=2, columnspan=2, sticky='w', pady=(8, 0))
        ttk.Label(options, text='输出自动保存在软件目录下的 outputs 文件夹，每次任务单独保存。').grid(row=3, column=0, columnspan=4, sticky='w', pady=(6, 0))

        actions = ttk.Frame(outer)
        actions.grid(row=5, column=0, sticky='ew', pady=10)
        self.start_button = ttk.Button(actions, text='开始翻译', command=self._start, state='disabled')
        self.start_button.pack(side='left')
        self.cancel_button = ttk.Button(actions, text='取消任务', command=self._cancel, state='disabled')
        self.cancel_button.pack(side='left', padx=6)
        self.output_button = ttk.Button(actions, text='打开输出文件夹', command=self._open_output)
        self.output_button.pack(side='left', padx=6)
        self.subtitle_button = ttk.Button(actions, text='查看字幕', command=self._view_subtitle, state='disabled')
        self.subtitle_button.pack(side='left', padx=6)
        self.environment_button = ttk.Button(actions, text='重新检查环境', command=self._check_environment)
        self.environment_button.pack(side='right')

        progress_frame = ttk.Frame(outer)
        progress_frame.grid(row=6, column=0, sticky='ew', pady=(0, 7))
        self.progress = ttk.Progressbar(progress_frame, maximum=100, mode='determinate')
        self.progress.pack(fill='x')
        ttk.Label(progress_frame, textvariable=self.status, wraplength=990).pack(anchor='w', pady=(5, 0))
        self.log = ScrolledText(outer, height=9, wrap='word', state='disabled', font=('Consolas', 10))
        self.log.grid(row=7, column=0, sticky='nsew')
        ttk.Label(outer, textvariable=self.environment_status, wraplength=990).grid(row=8, column=0, sticky='w', pady=(7, 0))

    def _emit(self, event: dict):
        self.events.put(event)

    def _append_log(self, message: str):
        self.log.configure(state='normal')
        self.log.insert('end', f'[{time.strftime("%H:%M:%S")}] {message}\n')
        line_count = int(self.log.index('end-1c').split('.')[0])
        if line_count > 900:
            self.log.delete('1.0', f'{line_count - 800}.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def _drain_events(self):
        if self.closing:
            return
        try:
            for _ in range(150):
                event = self.events.get_nowait()
                kind = event.get('type')
                if kind == 'log':
                    self._append_log(str(event.get('message', '')))
                elif kind == 'progress':
                    self.progress['value'] = max(0, min(100, float(event.get('value', 0))))
                    self.status.set(str(event.get('message', '正在处理……')))
                elif kind == 'scan_batch':
                    self._add_videos(event['items'])
                elif kind == 'scan_done':
                    self.scanning = False
                    self._append_log(event['message'])
                    if not self.busy:
                        self.status.set(event['message'])
                    self._refresh_controls()
                elif kind == 'environment':
                    self.checking = False
                    problems = event['problems']
                    self.environment_ready = not problems
                    self.environment_status.set('环境已就绪。语音识别和翻译在本地进行。' if not problems else '环境尚未就绪，请查看日志，完成安装后点击“重新检查环境”。')
                    self._append_log('运行环境检查通过。' if not problems else '运行环境需要处理：\n' + '\n'.join('• ' + str(problem) for problem in problems))
                    self._refresh_controls()
                elif kind == 'task_done':
                    self.busy = False
                    self.last_result = event['result']
                    self.progress['value'] = 100
                    self.status.set('翻译完成。可以打开输出文件夹或查看字幕。')
                    self._append_log('任务完成，保存位置：' + str(self.last_result.get('output_dir', self.root / 'outputs')))
                    self._refresh_controls()
                elif kind == 'task_cancelled':
                    self.busy = False
                    self.status.set('任务已取消。已完成的阶段文件保存在输出目录。')
                    self._append_log('任务已取消。')
                    self._refresh_controls()
                elif kind == 'task_error':
                    self.busy = False
                    self.status.set('任务未完成，请查看日志中的错误信息。')
                    self._append_log('任务失败：' + event['message'])
                    self._refresh_controls()
                    messagebox.showerror('翻译未完成', event['message'], parent=self.window)
        except queue.Empty:
            pass
        self.window.after(80, self._drain_events)

    def _refresh_controls(self):
        selected = bool(self.tree.selection())
        language_selected = bool(LANGUAGES.get(self.language.get(), ''))
        self.start_button.configure(state='normal' if selected and language_selected and self.environment_ready and not self.busy and not self.checking else 'disabled')
        self.cancel_button.configure(state='normal' if self.busy and not self.cancel_event.is_set() else 'disabled')
        self.folder_button.configure(state='disabled' if self.scanning else 'normal')
        self.stop_scan_button.configure(state='normal' if self.scanning and not self.scan_cancel.is_set() else 'disabled')
        self.clear_button.configure(state='disabled' if self.scanning or self.busy else 'normal')
        self.environment_button.configure(state='disabled' if self.checking or self.busy else 'normal')
        self.backend_box.configure(state='disabled' if self.busy or self.checking else 'readonly')
        self.force_check.configure(state='disabled' if self.busy else 'normal')
        self.language_box.configure(state='disabled' if self.busy else 'readonly')
        self.mode_box.configure(state='disabled' if self.busy else 'readonly')
        self.bilingual_check.configure(state='disabled' if self.busy else 'normal')
        self.subtitle_button.configure(state='normal' if self.last_result.get('zh_srt') else 'disabled')

    def _selection_changed(self, _event=None):
        selection = self.tree.selection()
        self.selected_path.set(str(self.videos[selection[0]]) if selection else '尚未选择视频')
        self._refresh_controls()

    def _add_videos(self, items):
        first = None
        for path, size in items:
            path = Path(path)
            key = os.path.normcase(os.path.abspath(path))
            if key in self.known_paths:
                continue
            self.known_paths.add(key)
            self.item_counter += 1
            item = str(self.item_counter)
            self.videos[item] = path
            self.tree.insert('', 'end', iid=item, values=(path.name, format_size(size), str(path.parent)))
            if first is None:
                first = item
        self.count_text.set(f'{len(self.videos)} 个视频')
        if not self.tree.selection() and first:
            self.tree.selection_set(first)
            self.tree.focus(first)
        self._selection_changed()

    def _choose_files(self):
        paths = filedialog.askopenfilenames(parent=self.window, title='选择需要翻译的视频', initialdir=str(self.folder), filetypes=[('视频文件', ' '.join('*' + ext for ext in sorted(VIDEO_EXTENSIONS))), ('所有文件', '*.*')])
        if not paths:
            return
        self.folder = Path(paths[0]).parent
        items = []
        for path in paths:
            try:
                items.append((path, Path(path).stat().st_size))
            except OSError as exc:
                self._append_log(f'无法读取 {path}：{exc}')
        self._add_videos(items)

    def _choose_folder(self):
        if self.scanning:
            return
        folder = filedialog.askdirectory(parent=self.window, title='选择视频所在文件夹（可选择磁盘根目录）', initialdir=str(self.folder), mustexist=True)
        if not folder:
            return
        self.folder = Path(folder)
        self.scan_cancel = threading.Event()
        self.scanning = True
        self._refresh_controls()
        self._append_log(f'正在扫描：{folder}')
        if not self.busy:
            self.status.set('正在扫描文件夹，发现的视频会逐步显示。')
        threading.Thread(target=self._scan_worker, args=(self.folder, self.recursive.get(), self.scan_cancel), daemon=True, name='video-scan').start()

    def _scan_worker(self, folder: Path, recursive: bool, stop: threading.Event):
        pending = [folder]
        batch = []
        found = 0
        skipped = 0
        try:
            while pending and not stop.is_set():
                current = pending.pop()
                try:
                    with os.scandir(current) as entries:
                        for entry in entries:
                            if stop.is_set():
                                break
                            try:
                                if entry.is_dir(follow_symlinks=False):
                                    if recursive:
                                        pending.append(Path(entry.path))
                                elif Path(entry.name).suffix.lower() in VIDEO_EXTENSIONS and entry.is_file(follow_symlinks=False):
                                    batch.append((entry.path, entry.stat(follow_symlinks=False).st_size))
                                    found += 1
                                    if len(batch) >= 60:
                                        self._emit({'type': 'scan_batch', 'items': batch})
                                        batch = []
                            except OSError:
                                skipped += 1
                except OSError:
                    skipped += 1
            if batch:
                self._emit({'type': 'scan_batch', 'items': batch})
            message = f'扫描已停止，发现 {found} 个视频。' if stop.is_set() else f'扫描完成，发现 {found} 个视频。'
            if skipped:
                message += f' 已跳过 {skipped} 个无法访问的项目。'
        except Exception as exc:
            message = f'扫描中断：{exc}'
        self._emit({'type': 'scan_done', 'message': message})

    def _stop_scan(self):
        self.scan_cancel.set()
        self._refresh_controls()

    def _clear_list(self):
        if self.scanning or self.busy:
            return
        self.tree.delete(*self.tree.get_children())
        self.videos.clear()
        self.known_paths.clear()
        self.count_text.set('0 个视频')
        self._selection_changed()

    def _backend_changed(self, _event=None):
        self.environment_ready = False
        self._check_environment()

    def _language_changed(self, _event=None):
        language = LANGUAGES.get(self.language.get(), '')
        backend = LANGUAGE_BACKENDS.get(language)
        if backend:
            self.asr_backend.set(next(label for label, value in ASR_BACKENDS.items() if value == backend))
            self.environment_ready = False
            self._check_environment()
        else:
            self._refresh_controls()

    def _check_environment(self):
        if self.checking or self.busy:
            return
        self.checking = True
        self.environment_ready = False
        backend = ASR_BACKENDS[self.asr_backend.get()]
        self.environment_status.set(f'正在检查 {self.asr_backend.get()} 的运行环境……')
        self._refresh_controls()
        def worker():
            try:
                from pipeline import Pipeline
                pipeline = Pipeline(self.root, self._emit, threading.Event())
                problems = pipeline.check_environment(asr_backend=backend)
            except Exception as exc:
                problems = [str(exc)]
            self._emit({'type': 'environment', 'problems': problems})
        threading.Thread(target=worker, daemon=True, name='environment-check').start()

    def _start(self):
        if self.busy or self.checking or not self.environment_ready:
            return
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo('选择视频', '请先从列表中选择一个视频。', parent=self.window)
            return
        video = self.videos[selection[0]]
        if not video.is_file():
            messagebox.showerror('文件不存在', f'无法找到视频：\n{video}', parent=self.window)
            return
        language = LANGUAGES.get(self.language.get(), '')
        if not language:
            messagebox.showinfo('选择视频语言', '请先选择视频中主要说话者使用的语言。', parent=self.window)
            return
        mode = MODES[self.mode.get()]
        bilingual = self.bilingual.get()
        backend = ASR_BACKENDS[self.asr_backend.get()]
        if backend == 'kotoba' and language != 'ja':
            messagebox.showinfo('识别模型与语言不匹配', 'Kotoba 日语专用模型只能识别日语。请先选择日语，或切换到对应语言的识别模型。', parent=self.window)
            return
        force_retranscribe = self.force_retranscribe.get()
        self.cancel_event = threading.Event()
        self.busy = True
        self.progress['value'] = 0
        self.status.set('任务正在启动……')
        self._append_log(f'开始翻译：{video}\n语言：{self.language.get()}；输出：{self.mode.get()}；双语：{"是" if bilingual else "否"}')
        self._append_log(f'识别引擎：{self.asr_backend.get()}；重新识别：{"是" if force_retranscribe else "否"}')
        self._refresh_controls()
        threading.Thread(target=self._run_worker, args=(str(video), language, mode, bilingual, self.cancel_event, backend, force_retranscribe), daemon=True, name='translation-task').start()

    def _run_worker(self, video: str, language: str, mode: str, bilingual: bool, stop: threading.Event, asr_backend: str | None = None, force_retranscribe: bool = False):
        try:
            from pipeline import Pipeline
            result = Pipeline(self.root, self._emit, stop).run(video, language=language, mode=mode, bilingual=bilingual, asr_backend=asr_backend, force_retranscribe=force_retranscribe)
            self._emit({'type': 'task_done', 'result': result})
        except InterruptedError:
            self._emit({'type': 'task_cancelled'})
        except Exception as exc:
            self._emit({'type': 'task_error', 'message': str(exc) or type(exc).__name__})

    def _cancel(self):
        if self.busy:
            self.cancel_event.set()
            self.status.set('正在取消，请等待当前步骤安全停止……')
            self._append_log('已请求取消任务。')
            self._refresh_controls()

    def _open_output(self):
        path = Path(self.last_result.get('output_dir') or self.root / 'outputs')
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))
        except OSError as exc:
            messagebox.showerror('无法打开文件夹', str(exc), parent=self.window)

    def _view_subtitle(self):
        path = self.last_result.get('bilingual_srt') if self.bilingual.get() else self.last_result.get('zh_srt')
        path = path or self.last_result.get('zh_srt')
        if not path or not Path(path).is_file():
            messagebox.showinfo('暂无字幕', '请先完成一次翻译。', parent=self.window)
            return
        try:
            subprocess.Popen(['notepad.exe', str(path)], cwd=str(self.root))
        except OSError as exc:
            messagebox.showerror('无法打开字幕', str(exc), parent=self.window)

    def _on_close(self):
        if self.busy:
            if not messagebox.askyesno('取消并退出', '视频仍在处理中。取消当前任务并退出？', parent=self.window):
                return
            self.cancel_event.set()
            self.scan_cancel.set()
            self.status.set('正在停止任务，完成后自动退出……')
            self._refresh_controls()
            self.window.after(150, self._wait_then_close)
            return
        self.scan_cancel.set()
        self.closing = True
        self.window.destroy()

    def _wait_then_close(self):
        if self.busy:
            self.window.after(150, self._wait_then_close)
        else:
            self.closing = True
            self.window.destroy()


def main():
    os.chdir(ROOT)
    window = tk.Tk()
    try:
        ttk.Style(window).theme_use('vista')
    except tk.TclError:
        pass
    VideoTranslatorApp(window)
    window.mainloop()


if __name__ == '__main__':
    main()
