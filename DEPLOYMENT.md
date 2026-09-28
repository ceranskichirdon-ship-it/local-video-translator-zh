# Windows 部署说明

这份说明安装能完成英语、日语、韩语视频翻译的**核心流程**。Kotoba 与 Qwen3-ASR 属于可选的对照识别引擎，不影响核心安装。脚本下载第三方程序和权重到本地，不把它们提交到 Git。

## 环境要求

- Windows 10/11 x64；已验证环境为 Windows 11、RTX 5060 Ti 16GB。
- Python 3.10 或更新版本，安装时包含 Tkinter；本机使用 Python 3.14，模型相关的可选环境使用 Python 3.12。
- PowerShell 7，安装脚本使用并行分段下载。
- 适用于 NVIDIA 显卡的驱动。当前固定的 whisper.cpp 是 CUDA 12.4 构建，llama.cpp 是 Vulkan 构建。需要可用的 NVIDIA 驱动与 Vulkan 运行支持；其他平台未验证。
- 至少 30GB 可用磁盘空间，下载缓存和解压过程会暂时占用更多空间。

## 一次安装

从 GitHub 下载源码或克隆仓库，打开 PowerShell 7，进入仓库目录：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_core.ps1
py -3 .\scripts\check_install.py --asr whisper_precise
py -3 app.py
```

也可在资源管理器中双击 `start_windows.cmd`。首次启动时，界面会检查所选识别引擎所需的文件。日语默认使用 `whisper_precise`，所以安装脚本也下载 Silero VAD。

下载来自固定的上游版本，脚本验证 SHA-256。若已有同名但校验不符的文件，脚本会停止并请你检查该文件；不会静默覆盖。网络中断后，分段下载缓存位于 `downloads/`，重新执行脚本可继续。`downloads/`、`models/`、`tools/`、`work/` 和 `outputs/` 均被 `.gitignore` 排除。

核心安装后的目录应类似：

```text
项目目录/
  app.py, pipeline.py, ...
  models/
    ggml-large-v3.bin
    ggml-silero-v6.2.0.bin
    Hy-MT2-7B-Q4_K_M.gguf
    Qwen3-8B-Q4_K_M.gguf
  tools/
    ffmpeg/ffmpeg.exe, ffprobe.exe
    whisper/whisper-cli.exe, whisper-vad-speech-segments.exe, ...DLL
    llama/llama-server.exe, ...DLL
  work/       # 自动生成的缓存
  outputs/    # 任务输出
```

不要只复制 `.exe` 而遗漏同目录 DLL。`setup_core.ps1` 会把完整解压目录保留在本地。

## 配置

默认模型与工具位置都相对于项目根目录，不需要创建 `config.json`。要修改线程、显卡设备或模型位置，可将 `config.example.json` 复制为 `config.json`，并在其中加入覆盖项。`config.json` 是本机文件，不会上传。

默认翻译阶段通过 `llama.cpp` 使用 `Vulkan0`，语音识别通过 CUDA 版 whisper.cpp 使用 GPU。若机器的 Vulkan 设备名称不同，修改 `config.json` 的 `llama_device`。模型服务启动失败时查看输出任务目录中的 `context_analysis_server.log` 或 `translation_server.log`。若缺少文件，运行 `py -3 .\scripts\check_install.py --asr whisper_precise` 可看到具体路径。

## 可选识别引擎

在界面选择其他识别引擎前，需另备 Python 3.12 虚拟环境及相应模型。建议使用 `tools/qwen-asr/` 存放该环境，再在本机 `config.json` 中设置 `qwen_python` 为其 `Scripts/python.exe` 的**绝对路径**。核心程序仍使用系统 Python。可选环境至少需要与本机已验证组合相符的 `torch 2.8.0+cu128`、`torchaudio 2.8.0+cu128`、`qwen-asr 0.0.6`、`transformers 4.57.6`、`accelerate 1.12.0`、`numpy`、`soundfile`、`librosa`。安装方式与 GPU 版本请参照 [PyTorch](https://pytorch.org/get-started/locally/) 和 [Qwen3-ASR 官方项目](https://github.com/QwenLM/Qwen3-ASR)。

- **Qwen3-ASR 1.7B**：将 [识别模型](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 的完整仓库内容放入 `models/Qwen3-ASR-1.7B/`，将 [ForcedAligner 0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) 放入 `models/Qwen3-ForcedAligner-0.6B/`。两处都需要 `config.json` 与权重文件，其他 tokenizer 文件也应保留。
- **Kotoba-Whisper v2.1**：将 [完整模型仓库](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.1) 放入 `models/kotoba-whisper-v2.1/`，确保包含 `config.json`、`model.safetensors` 和 `kotoba_whisper.py`。只在已选择日语时使用。

可执行 `py -3 .\scripts\check_install.py --asr qwen3` 或 `--asr kotoba` 检查文件布局。以上模型目录是本地副本，仓库不会代为分发权重。可选识别依赖体积大，第一次安装后请先用短视频验证。

## 使用与排查

1. 在软件里添加视频或扫描文件夹，必须选英语、日语、韩语之一。
2. 选输出形式：`SRT` 便于先检查翻译；`MKV` 可开关字幕；`MP4` 会把字幕烧入画面。
3. 点击“开始翻译”。每个任务的字幕、报告、日志保存在 `outputs/` 的独立目录中。原视频保持原样。
4. 若字幕漏句，先看 `原文字幕.srt`。原文缺句属于识别问题；可切换识别引擎或勾选“重新识别”后重试。若原文正确但译文不对，再查看 `翻译背景备注.json` 和 `字幕对照.json`。

命令行示例：

```powershell
py -3 pipeline.py "D:\Videos\example.mp4" --language ja --mode srt
```

同一个视频和相同设置会复用本地进度；更换识别引擎、识别模型或勾选“重新识别”会重新识别。不要同时开启多个程序窗口处理同一任务。

## 第三方来源

安装脚本使用 [FFmpeg Windows 构建](https://www.gyan.dev/ffmpeg/builds/)、[whisper.cpp 发布文件](https://github.com/ggml-org/whisper.cpp/releases)、[llama.cpp 发布文件](https://github.com/ggml-org/llama.cpp/releases) 及 [Hugging Face 模型仓库](https://huggingface.co/models)。具体版本、下载地址与预期哈希写在 `scripts/setup_media.ps1` 和 `scripts/setup_core.ps1`。这些依赖分别遵守其上游许可；本仓库的 MIT 许可证只覆盖原创程序代码。
