param([switch]$SkipDownload)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$ProjectRoot=Split-Path -Parent $PSScriptRoot
$ModelRoot=Join-Path $ProjectRoot 'models'
$Downloads=Join-Path $ProjectRoot 'downloads'
$WhisperHome=Join-Path $ProjectRoot 'tools\whisper'
$FFmpegHome=Join-Path $ProjectRoot 'tools\ffmpeg'
$Work=Join-Path $ProjectRoot 'work'
$env:TEMP=$Work
$env:TMP=$Work
New-Item -ItemType Directory -Path $Downloads,$WhisperHome,$FFmpegHome,$Work,$ModelRoot -Force | Out-Null
$Assets=@(
  @{name='whisper';url='https://github.com/ggml-org/whisper.cpp/releases/download/b5130/whisper-cublas-12.4.0-bin-x64.zip';path=(Join-Path $Downloads 'whisper-cublas-12.4.0-b5130.zip');size=674539285;sha256='af520ddd034d985b55dfeea3e465ed93653ba2aee1a55e865033edc548c272a7'},
  @{name='ffmpeg';url='https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip';path=(Join-Path $Downloads 'ffmpeg-release-essentials.zip');size=114768076;sha256='60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba'},
  @{name='whisper_model';url='https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin';path=(Join-Path $ModelRoot 'ggml-large-v3.bin');size=3095033483;sha256='64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2'}
)
foreach($Asset in $Assets){
  if(!(Test-Path -LiteralPath $Asset.path)){
    if($SkipDownload){throw "Missing $($Asset.path)"}
    Write-Output "Downloading $($Asset.name)"
    if($Asset.size){ & (Join-Path $PSScriptRoot 'download_ranges.ps1') -Url $Asset.url -Destination $Asset.path -Size $Asset.size -Sha256 $Asset.sha256 }else{ Invoke-WebRequest -Uri $Asset.url -OutFile $Asset.path -TimeoutSec 1800 }
  }
  $Digest=(Get-FileHash -LiteralPath $Asset.path -Algorithm SHA256).Hash.ToLowerInvariant()
  if($Digest -ne $Asset.sha256){throw "SHA256 mismatch for $($Asset.name). Remove the incomplete file and retry."}
}
Expand-Archive -LiteralPath $Assets[0].path -DestinationPath $WhisperHome -Force
Expand-Archive -LiteralPath $Assets[1].path -DestinationPath $FFmpegHome -Force
$WhisperSource=(Get-ChildItem -LiteralPath $WhisperHome -Recurse -Filter 'whisper-cli.exe' | Select-Object -First 1).Directory.FullName
if(!$WhisperSource){throw 'whisper-cli.exe missing in archive'}
if($WhisperSource -ne $WhisperHome){ Get-ChildItem -LiteralPath $WhisperSource -File | Copy-Item -Destination $WhisperHome -Force }
$FFmpegSource=(Get-ChildItem -LiteralPath $FFmpegHome -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1).Directory.FullName
if(!$FFmpegSource){throw 'ffmpeg.exe missing in archive'}
if($FFmpegSource -ne $FFmpegHome){ Get-ChildItem -LiteralPath $FFmpegSource -File | Copy-Item -Destination $FFmpegHome -Force }
$WhisperCLI=Join-Path $WhisperHome 'whisper-cli.exe'
$FFmpegCLI=Join-Path $FFmpegHome 'ffmpeg.exe'
$FFprobeCLI=Join-Path $FFmpegHome 'ffprobe.exe'
$WhisperHelp=(& $WhisperCLI --help 2>&1 | Out-String)
$FFmpegVersion=(& $FFmpegCLI -version 2>&1 | Out-String)
$WhisperHelp | Set-Content -LiteralPath (Join-Path $Work 'whisper-help.txt') -Encoding UTF8
$FFmpegVersion | Set-Content -LiteralPath (Join-Path $Work 'ffmpeg-version.txt') -Encoding UTF8
$Manifest=[ordered]@{
  ffmpeg=$FFmpegCLI;ffprobe=$FFprobeCLI;whisper_cli=$WhisperCLI;whisper_model=$Assets[2].path
  versions=@{whisper='v1.9.4/b5130 CUDA 12.4';ffmpeg='9.0.2 essentials';whisper_model='Whisper large-v3 GGML multilingual'}
  sources=@{whisper=$Assets[0].url;ffmpeg=$Assets[1].url;whisper_model=$Assets[2].url}
  sha256=@{whisper_archive=$Assets[0].sha256;ffmpeg_archive=$Assets[1].sha256;whisper_model=$Assets[2].sha256}
  verification=@{sha256_verified=$true;cli_help_verified=($WhisperHelp -match 'output-json');ffmpeg_version_verified=($FFmpegVersion -match '9.0.2');created_at=(Get-Date -Format o)}
}
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $ProjectRoot 'tools\media_manifest.json') -Encoding UTF8
Write-Output ($Manifest | ConvertTo-Json -Depth 6)
