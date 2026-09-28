param([switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($PSVersionTable.PSVersion.Major -lt 7) {
  throw '此安装脚本需要 PowerShell 7。请用 pwsh -File scripts/setup_core.ps1 运行。'
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ModelRoot = Join-Path $ProjectRoot 'models'
$Downloads = Join-Path $ProjectRoot 'downloads'
$LlamaHome = Join-Path $ProjectRoot 'tools/llama'
New-Item -ItemType Directory -Path $ModelRoot,$Downloads,$LlamaHome -Force | Out-Null

function Confirm-File {
  param([string]$Path,[string]$Sha256,[long]$Size = 0)
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
  $File = Get-Item -LiteralPath $Path
  if ($Size -and $File.Length -ne $Size) {
    throw "文件大小不符：$Path。请核实后手动移走这个文件，再重新运行。"
  }
  $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($Actual -ne $Sha256.ToLowerInvariant()) {
    throw "SHA-256 不符：$Path。请核实后手动移走这个文件，再重新运行。"
  }
  return $true
}

function Get-VerifiedArtifact {
  param([string]$Name,[string]$Url,[string]$Path,[string]$Sha256,[long]$Size = 0)
  if (Confirm-File -Path $Path -Sha256 $Sha256 -Size $Size) {
    Write-Output "已验证：$Name"
    return
  }
  if ($VerifyOnly) { throw "缺少：$Name ($Path)" }
  Write-Output "正在下载：$Name"
  if ($Size) {
    & (Join-Path $PSScriptRoot 'download_ranges.ps1') -Url $Url -Destination $Path -Size $Size -Sha256 $Sha256
  } else {
    Invoke-WebRequest -Uri $Url -OutFile $Path -TimeoutSec 1800
  }
  if (-not (Confirm-File -Path $Path -Sha256 $Sha256 -Size $Size)) {
    throw "下载未完成：$Name"
  }
}

$WhisperCli = Join-Path $ProjectRoot 'tools/whisper/whisper-cli.exe'
$WhisperVad = Join-Path $ProjectRoot 'tools/whisper/whisper-vad-speech-segments.exe'
$Ffmpeg = Join-Path $ProjectRoot 'tools/ffmpeg/ffmpeg.exe'
$Ffprobe = Join-Path $ProjectRoot 'tools/ffmpeg/ffprobe.exe'
$WhisperModel = Join-Path $ModelRoot 'ggml-large-v3.bin'
if ($VerifyOnly) {
  foreach ($File in @($WhisperCli,$WhisperVad,$Ffmpeg,$Ffprobe,$WhisperModel)) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "缺少：$File" }
  }
  Write-Output '已找到 FFmpeg、whisper.cpp 与 Whisper 模型。'
} elseif (-not ((Test-Path -LiteralPath $WhisperCli -PathType Leaf) -and
              (Test-Path -LiteralPath $WhisperVad -PathType Leaf) -and
              (Test-Path -LiteralPath $Ffmpeg -PathType Leaf) -and
              (Test-Path -LiteralPath $Ffprobe -PathType Leaf) -and
              (Test-Path -LiteralPath $WhisperModel -PathType Leaf))) {
  & (Join-Path $PSScriptRoot 'setup_media.ps1')
}

$LlamaServer = Join-Path $LlamaHome 'llama-server.exe'
if (-not (Test-Path -LiteralPath $LlamaServer -PathType Leaf)) {
  $LlamaArchive = Join-Path $Downloads 'llama-b10964-win-vulkan-x64.zip'
  Get-VerifiedArtifact -Name 'llama.cpp Vulkan' `
    -Url 'https://github.com/ggml-org/llama.cpp/releases/download/b10964/llama-b10964-bin-win-vulkan-x64.zip' `
    -Path $LlamaArchive -Sha256 '1ee3ad952f4ba71f438bd6d7bebef19e1c7af04adcaa35d08b4ddabb27d4c642'
  if ($VerifyOnly) { throw "缺少：$LlamaServer" }
  Expand-Archive -LiteralPath $LlamaArchive -DestinationPath $LlamaHome -Force
  $Server = Get-ChildItem -LiteralPath $LlamaHome -Recurse -File -Filter 'llama-server.exe' | Select-Object -First 1
  if (-not $Server) { throw 'llama.cpp 压缩包中找不到 llama-server.exe' }
  if ($Server.DirectoryName -ne $LlamaHome) {
    Get-ChildItem -LiteralPath $Server.DirectoryName -File | Copy-Item -Destination $LlamaHome -Force
  }
}
if (-not (Test-Path -LiteralPath $LlamaServer -PathType Leaf)) { throw "缺少：$LlamaServer" }

$Assets = @(
  @{Name='Silero VAD'; Url='https://huggingface.co/ggml-org/whisper-vad/resolve/9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v6.2.0.bin'; Path=(Join-Path $ModelRoot 'ggml-silero-v6.2.0.bin'); Size=885098; Sha256='2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987'},
  @{Name='Hy-MT2 7B'; Url='https://huggingface.co/tencent/Hy-MT2-7B-GGUF/resolve/main/Hy-MT2-7B-Q4_K_M.gguf'; Path=(Join-Path $ModelRoot 'Hy-MT2-7B-Q4_K_M.gguf'); Size=4624648896; Sha256='9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b'},
  @{Name='Qwen3 8B'; Url='https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf'; Path=(Join-Path $ModelRoot 'Qwen3-8B-Q4_K_M.gguf'); Size=5027783488; Sha256='d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785'}
)
foreach ($Asset in $Assets) {
  Get-VerifiedArtifact -Name $Asset.Name -Url $Asset.Url -Path $Asset.Path -Size $Asset.Size -Sha256 $Asset.Sha256
}
Write-Output '核心组件和模型已就绪。请运行：py -3 .\scripts\check_install.py --asr whisper_precise'
