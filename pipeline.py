from __future__ import annotations
import hashlib, json, os, re, threading, time, uuid
from pathlib import Path
from translation import LocalTranslator, discard_invalid_cached_translations

ROOT = Path(__file__).resolve().parent
LANGUAGE_BACKENDS = {'en': 'whisper', 'ja': 'whisper_precise', 'ko': 'whisper'}

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def read_json(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return fallback

class Pipeline:
    def __init__(self, root=ROOT, emit=None, cancel_event=None):
        self.root = Path(root).resolve()
        self.emit = emit or (lambda e: None)
        self.cancel = cancel_event or threading.Event()
        self.config = read_json(self.root/'config.json', {})
        defaults = {'llama_path':str(self.root/'tools/llama/llama-server.exe'),
                    'translation_model':str(self.root/'models/Hy-MT2-7B-Q4_K_M.gguf'),
                    'context_model':str(self.root/'models/Qwen3-8B-Q4_K_M.gguf'),
                    'ffmpeg_path':str(self.root/'tools/ffmpeg/ffmpeg.exe'),
                    'ffprobe_path':str(self.root/'tools/ffmpeg/ffprobe.exe'),
                    'whisper_path':str(self.root/'tools/whisper/whisper-cli.exe'),
                    'whisper_model':str(self.root/'models/ggml-large-v3.bin'),
                    'whisper_vad_model':str(self.root/'models/ggml-silero-v6.2.0.bin'),
                    'asr_backend':'whisper',
                    'whisper_vad_enabled':False,
                    'qwen_python':str(self.root/'tools/qwen-asr/python.exe'),
                    'kotoba_model':str(self.root/'models/kotoba-whisper-v2.1'),
                    'qwen_asr_model':str(self.root/'models/Qwen3-ASR-1.7B'),
                    'qwen_aligner_model':str(self.root/'models/Qwen3-ForcedAligner-0.6B')}
        self.config = {**defaults, **self.config, 'base_dir':str(self.root)}
        self.log_path = None
        self.progress = 0

    def check_environment(self, asr_backend=None):
        problems = []
        backend = asr_backend or self.config.get('asr_backend', 'whisper')
        if backend not in ('qwen3', 'whisper', 'whisper_precise', 'kotoba'):
            return ['不支持的语音识别引擎：' + str(backend)]
        required = [('ffmpeg_path', '视频处理程序'), ('ffprobe_path', '视频信息程序'),
                    ('llama_path', '本地翻译程序'), ('translation_model', 'Hy-MT2 中文翻译模型'),
                    ('context_model', 'Qwen3 全片理解模型')]
        if backend in ('whisper', 'whisper_precise'):
            required += [('whisper_path', '语音识别程序'), ('whisper_model', 'Whisper large-v3 语音模型')]
            if backend == 'whisper_precise':
                required += [('whisper_vad_model', '语音活动模型')]
                vad_tool = Path(self.config['whisper_path']).with_name('whisper-vad-speech-segments.exe')
                if not vad_tool.is_file():
                    problems.append(f'缺少精细识别的语音检测程序：{vad_tool}')
                if not (self.root / 'precise_asr.py').is_file():
                    problems.append('缺少精细识别模块：precise_asr.py')
        elif backend == 'qwen3':
            required += [('qwen_python', 'Qwen 语音运行环境')]
            if not (self.root / 'qwen_asr_worker.py').is_file():
                problems.append('缺少 Qwen3 语音识别模块。')
            for key, label in [('qwen_asr_model', 'Qwen3 语音模型'), ('qwen_aligner_model', 'Qwen3 时间对齐模型')]:
                folder = Path(self.config[key])
                if not (folder / 'config.json').is_file() or not any(folder.glob('*.safetensors')):
                    problems.append(f'{label}尚未安装完整：{folder}')
        else:
            required += [('qwen_python', 'Kotoba 语音运行环境')]
            folder = Path(self.config['kotoba_model'])
            for file_name, label in [('config.json', 'Kotoba 模型配置'), ('model.safetensors', 'Kotoba 语音权重'),
                                     ('kotoba_whisper.py', 'Kotoba 时间戳适配')]:
                if not (folder / file_name).is_file() or (folder / file_name).stat().st_size == 0:
                    problems.append(f'{label}不存在：{folder / file_name}')
            if not (self.root / 'kotoba_asr_worker.py').is_file():
                problems.append('缺少 Kotoba 日语识别模块。')
        for name, label in required:
            path = Path(self.config[name])
            if not path.is_file() or path.stat().st_size == 0:
                problems.append(f'{label}不存在：{path}')
        vad = self.config.get('whisper_vad_model')
        if backend == 'whisper' and self.config.get('whisper_vad_enabled', False) and vad and not Path(vad).is_file():
            problems.append(f'语音活动模型不存在：{vad}')
        return problems

    def _recognition_fingerprint(self):
        backend = self.config.get('asr_backend', 'whisper')
        prefixes = {'qwen3': 'qwen_', 'whisper': 'whisper_',
                    'whisper_precise': 'whisper_', 'kotoba': 'kotoba_'}
        prefix = prefixes.get(backend, '')
        settings = {key: value for key, value in self.config.items() if prefix and key.startswith(prefix)}
        if backend == 'qwen3':
            paths = [self.config['qwen_asr_model'], self.config['qwen_aligner_model'], self.root / 'qwen_asr_worker.py']
        elif backend == 'kotoba':
            paths = [self.config['kotoba_model'], self.root / 'kotoba_asr_worker.py']
        elif backend == 'whisper_precise':
            paths = [self.config['whisper_model'], self.config.get('whisper_vad_model', ''),
                     self.root / 'precise_asr.py',
                     Path(self.config['whisper_path']).with_name('whisper-vad-speech-segments.exe')]
        else:
            paths = [self.config['whisper_model'], self.config.get('whisper_vad_model', '')]
        files = {}
        for value in paths:
            if not value:
                continue
            path = Path(value)
            items = sorted(path.glob('*')) if path.is_dir() else [path]
            for item in items:
                if item.is_file():
                    stat = item.stat()
                    files[str(item)] = [stat.st_size, stat.st_mtime_ns]
        return {'backend': backend, 'settings': settings, 'files': files, 'version': 7}

    def _event(self, e):
        if e.get('type') == 'progress':
            if 'value' not in e:
                progress = e.get('progress',0)
                low, high = {'probe':(1,3),'extract':(3,10),'transcribe':(10,50),'export':(84,99)}.get(e.get('stage'), (0,100))
                e = dict(e, value=low+(high-low)*progress/100)
            self.progress = max(self.progress, min(100,e['value']))
            e = dict(e, value=self.progress)
        if self.log_path and e.get('message'):
            with self.log_path.open('a',encoding='utf-8') as f:
                f.write(time.strftime('%H:%M:%S ') + e['message'] + '\n')
        self.emit(e)

    def _check_cancel(self):
        if self.cancel.is_set():
            raise InterruptedError('已取消。再次选择同一视频开始，可以利用已保存进度。')

    def run(self, video, language='', mode='burn', bilingual=False, asr_backend=None, force_retranscribe=False):
        from video_media import MediaTools, write_srt as media_write_srt, CancelledError
        def write_srt(segments, target, text_key='text'):
            # These are this job's generated subtitles, never the input video.
            target = Path(target)
            temporary = target.with_name('subtitle_' + uuid.uuid4().hex + '.srt')
            try:
                media_write_srt(segments, temporary, text_key)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if language not in ('en', 'ja', 'ko'):
            raise ValueError('请先选择视频中的主要语言：英语、日语或韩语。')
        if mode not in ('burn','soft','srt'):
            raise ValueError('不支持的输出模式')
        self._check_cancel()
        self.config['asr_backend'] = asr_backend or LANGUAGE_BACKENDS[language]
        self.config['asr_force'] = force_retranscribe
        if self.config['asr_backend'] == 'kotoba' and language != 'ja':
            raise ValueError('Kotoba-Whisper 是日语专用模型，请选择日语或切换识别模型。')
        problems = self.check_environment()
        if problems:
            raise RuntimeError('\n'.join(problems))
        source = Path(video).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError('所选视频不存在')
        stat = source.stat()
        signature = {'path':str(source),'size':stat.st_size,'mtime':stat.st_mtime_ns,'language':language,
                     'recognition':self._recognition_fingerprint(), 'pipeline_version':7}
        job_id = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:12]
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', source.stem).strip(' .')[:65] or 'video'
        output_dir = self.root/'outputs'/f'{stem}_{job_id}'
        work_dir = self.root/'work'/'jobs'/job_id
        output_dir.mkdir(parents=True,exist_ok=True)
        work_dir.mkdir(parents=True,exist_ok=True)
        self.log_path = output_dir/'任务日志.txt'
        self.config['work_dir'] = str(work_dir)
        media = MediaTools(self.config, self._event, self.cancel)
        state_path = work_dir/'recognition.json'
        translation_path = work_dir/'translations.json'
        manifest = {'source':str(source),'output_dir':str(output_dir),'language_requested':language,
                    'mode':mode,'bilingual':bilingual,'status':'running','job_id':job_id,
                    'asr_backend':self.config.get('asr_backend','whisper')}
        atomic_json(output_dir/'任务信息.json', manifest)
        self._event({'type':'log','message':f'视频：{source}\n输出目录：{output_dir}'})
        translator = None
        try:
            info = media.probe(source)
            if not info.get('has_video') or not info.get('has_audio'):
                raise ValueError('请选择同时包含画面和音轨的视频。')
            recognition = None if force_retranscribe else read_json(state_path, None)
            if recognition and recognition.get('signature') == signature:
                segments, detected = recognition['segments'], recognition['language']
                self._event({'type':'progress','value':50,'message':'已读取上次保存的识别结果。'})
            else:
                wav = work_dir/'audio.wav'
                if wav.exists():
                    self._event({'type':'log','message':'复用上次完整提取的音频。'})
                else:
                    media.extract_audio(source, wav)
                segments, detected = media.transcribe(wav, language)
                if not segments:
                    raise ValueError('视频中未识别到有效语音，不能生成空字幕。')
                atomic_json(state_path,{'signature':signature,'segments':segments,'language':detected})
            self._check_cancel()
            if self.config['asr_backend'] == 'whisper_precise':
                from subtitle_polish import polish_segments
                vad_values = read_json(work_dir/'precise_whisper'/'speech_segments.json', [])
                vad_intervals = [tuple(pair) for pair in vad_values if isinstance(pair, list) and len(pair) == 2]
                segments, polish_report = polish_segments(
                    segments, vad_intervals, work_dir/'subtitle_overrides.json')
                atomic_json(output_dir/'字幕时间整理报告.json', polish_report)
            source_srt = output_dir/'原文字幕.srt'
            write_srt(segments,source_srt)
            if self.config['asr_backend'] == 'whisper_precise':
                quality = read_json(work_dir/'precise_whisper'/'quality_report.json', None)
                if isinstance(quality, dict):
                    atomic_json(output_dir/'语音识别质量报告.json', quality)
            translation_path_value = Path(self.config['translation_model'])
            context_path_value = Path(self.config['context_model'])
            translation_stat = translation_path_value.stat()
            context_stat = context_path_value.stat()
            recognition_key = hashlib.sha256(json.dumps(segments,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            model_key = [str(translation_path_value),translation_stat.st_size,translation_stat.st_mtime_ns,
                         str(context_path_value),context_stat.st_size,context_stat.st_mtime_ns,8,recognition_key]
            stored = {} if force_retranscribe else read_json(translation_path,{})
            cache = stored.get('items',{}) if stored.get('model') == model_key else {}
            if not isinstance(cache, dict):
                cache = {}
            def save(items):
                atomic_json(translation_path,{'model':model_key,'items':items})
            removed_cache = discard_invalid_cached_translations(segments, detected, cache)
            if removed_cache:
                save(cache)
                self._event({'type':'log','message':f'旧缓存中 {len(removed_cache)} 条译文仍含原语言，正在重新翻译。'})
            if any(str(s['id']) not in cache for s in segments):
                notes_state_path = work_dir/'translation_context.json'
                notes_key = [str(context_path_value),context_stat.st_size,context_stat.st_mtime_ns,recognition_key,2]
                notes_state = read_json(notes_state_path,{})
                notes = notes_state.get('notes') if notes_state.get('model') == notes_key else None
                translator = LocalTranslator(self.config,self._event,self.cancel,output_dir)
                if not notes:
                    self._event({'type':'progress','value':52,'message':'正在理解完整对话并整理翻译备注。'})
                    translator.start(model_path=context_path_value,stage='context')
                    notes = translator.analyze_context(segments,detected)
                    translator.close()
                    translator = None
                    atomic_json(notes_state_path,{'model':notes_key,'notes':notes})
                else:
                    self._event({'type':'log','message':'复用已保存的全片理解备注。'})
                atomic_json(output_dir/'翻译背景备注.json',{'source_language':detected,'notes':notes})
                translator = LocalTranslator(self.config,self._event,self.cancel,output_dir)
                translator.start()
                translated = translator.translate(segments,detected,cache,save,context_notes=notes)
                translator.close()
                translator = None
            else:
                translated = [dict(s,zh=cache[str(s['id'])]) for s in segments]
                self._event({'type':'progress','value':82,'message':'已读取上次保存的中文字幕。'})
            self._check_cancel()
            reviewed_path = work_dir/'translation_overrides.json'
            if reviewed_path.is_file():
                from reviewed_translations import apply_reviewed_translations
                translated, reviewed_report = apply_reviewed_translations(
                    translated, detected, reviewed_path)
                atomic_json(output_dir/'译文人工复核报告.json', reviewed_report)
            zh_srt = output_dir/'中文字幕.srt'
            bi_srt = output_dir/'双语字幕.srt'
            write_srt(translated,zh_srt,'zh')
            bilingual_segments = [dict(s,display=s['zh']+'\n'+s['text']) for s in translated]
            write_srt(bilingual_segments,bi_srt,'display')
            atomic_json(output_dir/'字幕对照.json',translated)
            result = {'output_dir':str(output_dir),'zh_srt':str(zh_srt),
                      'bilingual_srt':str(bi_srt),'source_srt':str(source_srt),'detected_language':detected}
            if mode != 'srt':
                suffix = '.mp4' if mode == 'burn' else '.mkv'
                out = output_dir/(('双语字幕视频' if bilingual else '中文字幕视频') + suffix)
                if out.exists():
                    out = out.with_stem(out.stem+'_'+time.strftime('%Y%m%d_%H%M%S'))
                media.export(source,bi_srt if bilingual else zh_srt,out,mode)
                result['video'] = str(out)
            manifest.update(result,status='complete',completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            atomic_json(output_dir/'任务信息.json',manifest)
            self._event({'type':'progress','value':100,'message':'完成，视频和字幕已保存。'})
            return result
        except CancelledError as exc:
            manifest.update(status='cancelled',error=str(exc))
            atomic_json(output_dir/'任务信息.json',manifest)
            raise InterruptedError(str(exc)) from exc
        except BaseException as exc:
            manifest.update(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
            atomic_json(output_dir/'任务信息.json',manifest)
            raise
        finally:
            if translator:
                translator.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='本地视频中文字幕工具')
    parser.add_argument('video')
    parser.add_argument('--language',choices=['en','ja','ko'],required=True,help='视频主要说话语言，必选')
    parser.add_argument('--mode',choices=['burn','soft','srt'],default='burn')
    parser.add_argument('--bilingual',action='store_true')
    parser.add_argument('--asr',choices=['qwen3','whisper','whisper_precise','kotoba'])
    parser.add_argument('--force-retranscribe',action='store_true')
    args = parser.parse_args()
    result = Pipeline(emit=lambda e:print(e.get('message',''),flush=True)).run(args.video,args.language,args.mode,args.bilingual,args.asr,args.force_retranscribe)
    print(json.dumps(result,ensure_ascii=False,indent=2))
