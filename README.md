# 本地视频翻译 · 中文字幕

一个面向 Windows 的本地视频翻译工具：从电脑选择视频，指定英语、日语或韩语，识别语音并翻译为中文，最后生成 SRT 或带字幕的视频。界面由 Python 标准库 Tkinter 编写；处理视频时不需要云端翻译 API。

> 当前版本以 **Windows 11 + NVIDIA RTX 5060 Ti 16GB** 为主要验证环境。项目提供源码与安装脚本，不在仓库中分发模型权重、FFmpeg、whisper.cpp、llama.cpp 或视频素材。首次安装需要下载约十余 GB 的模型及运行组件；建议预留至少 30GB 磁盘空间。其他显卡和 CPU 运行尚未完整验证。

## 能做什么

- 添加单个视频，或扫描本地文件夹；处理前必须选择主要说话语言。
- 英语、韩语默认使用 Whisper large-v3；日语默认使用 Whisper large-v3 加语音检测和分段校时。另可手动选择 Kotoba-Whisper v2.1 或 Qwen3-ASR 1.7B 对照识别。
- Qwen3-8B 先阅读整份识别稿，生成语境和称呼备注；Hy-MT2 7B 再结合备注与相邻语句翻译成中文。两个模型依次加载，以适应 16GB 显存。
- 输出原文字幕、中文字幕、双语字幕、对照 JSON、识别质量报告及任务日志。可选择仅生成 SRT、生成可开关字幕的 MKV，或烧录字幕为 MP4。
- 识别与翻译结果保存在本机 `work/`，中断后可按相同设置继续。原始视频不会被覆盖。

变声、背景音乐、重叠说话、俚语或低质量音轨仍可能导致漏识别和时间偏移。完成后请抽查 `原文字幕.srt`、`中文字幕.srt` 和 `语音识别质量报告.json`；机器翻译不能自动修正错误的原文识别。

## 快速安装（Windows）

1. 安装 [Python 3](https://www.python.org/downloads/windows/)（包含 Tkinter）与 [PowerShell 7](https://github.com/PowerShell/PowerShell#get-powershell)。在 PowerShell 7 中进入项目目录。
2. 运行 `pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_core.ps1`。脚本下载并校验固定版本的 FFmpeg、whisper.cpp CUDA 版、llama.cpp Vulkan 版及三个核心模型，并放入 `tools/` 和 `models/`。下载过程可能较长。
3. 运行 `py -3 .\scripts\check_install.py --asr whisper_precise` 检查日语默认识别所需文件。
4. 双击 `start_windows.cmd`，或执行 `py -3 app.py`。选择视频、语言、输出形式，点击“开始翻译”。

已配置好模型的现有安装目录可直接使用；安装脚本不会覆盖通过校验的模型文件。详细目录结构、可选识别引擎与故障排查见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 处理流程

```text
本地视频 → FFmpeg 提取音频 → Whisper / 可选 ASR 识别
        → 时间校正与原文字幕 → Qwen3-8B 全片语境备注
        → Hy-MT2 7B 逐条翻译 → SRT / MKV / MP4
```

主要代码：`app.py` 是桌面界面，`pipeline.py` 管理任务和缓存，`video_media.py` 负责音视频与字幕，`precise_asr.py` 和 `subtitle_polish.py` 处理时间戳，`translation.py` 管理两个本地翻译阶段。`qwen_asr_worker.py`、`kotoba_asr_worker.py` 是可选识别引擎的独立进程。

命令行也可运行：

```powershell
py -3 pipeline.py "D:\Videos\example.mp4" --language ja --mode srt
```

可选 `--mode burn|soft|srt`、`--asr whisper|whisper_precise|qwen3|kotoba`、`--bilingual`、`--force-retranscribe`。若直接调用命令行，请在项目目录执行，或为自定义路径使用绝对路径的 `config.json`。

## 隐私与授权

视频、音频、识别内容和译文都保存在本机。翻译时 `llama.cpp` 只监听 `127.0.0.1` 的临时端口，任务结束后关闭。首次安装会向所列项目下载运行组件和模型；运行任务本身不向云端上传视频。

本项目**原创程序代码**采用 [MIT 许可证](LICENSE)。下载的模型和第三方程序**不由本仓库重新授权**，使用前请查看各自许可：

| 用途 | 上游项目 |
| --- | --- |
| 语音识别 | [whisper.cpp](https://github.com/ggml-org/whisper.cpp)、[Whisper large-v3 GGML](https://huggingface.co/ggerganov/whisper.cpp)、[Silero VAD](https://huggingface.co/ggml-org/whisper-vad) |
| 全片语境 | [Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| 中文翻译 | [Hy-MT2-7B-GGUF](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) |
| 本地推理 | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| 视频处理 | [FFmpeg](https://ffmpeg.org/legal.html)；实际许可取决于所下载的构建 |
| 可选识别 | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[Kotoba-Whisper](https://github.com/kotoba-tech/kotoba-whisper) |

项目不附带上述模型、二进制或其许可保证，也不附带任何演示视频。翻译前请确认你有权处理素材。

## 开发与测试

核心流程只用 Python 标准库；可选 ASR 引擎有独立依赖。无需下载模型即可运行模拟测试：

```powershell
py -3 -m unittest discover -s tests -v
```

问题反馈请附上使用的语言和识别引擎、最短可复现步骤，以及去除个人路径与素材内容后的错误日志。不要在公开 Issue 上传视频、字幕或模型文件。
