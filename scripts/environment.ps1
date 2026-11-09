# Dot-source this file: . .\scripts\environment.ps1
$domainTuneRoot = Split-Path -Parent $PSScriptRoot
if ([IO.Path]::GetPathRoot($domainTuneRoot) -ne 'D:\') {
    throw 'DomainTune must run from D: to honor its storage constraints.'
}
$domainTuneStorage = Join-Path $domainTuneRoot '.local'
$domainTunePaths = @{
    UV_CACHE_DIR = "$domainTuneStorage\cache\uv"
    UV_PYTHON_INSTALL_DIR = "$domainTuneStorage\python"
    UV_PYTHON_BIN_DIR = "$domainTuneStorage\bin"
    UV_TOOL_DIR = "$domainTuneStorage\tools"
    UV_TOOL_BIN_DIR = "$domainTuneStorage\bin"
    UV_PROJECT_ENVIRONMENT = "$domainTuneRoot\.venv"
    PIP_CACHE_DIR = "$domainTuneStorage\cache\pip"
    HF_HOME = "$domainTuneStorage\cache\huggingface"
    HF_HUB_CACHE = "$domainTuneStorage\cache\huggingface\hub"
    HF_DATASETS_CACHE = "$domainTuneStorage\cache\huggingface\datasets"
    HF_ASSETS_CACHE = "$domainTuneStorage\cache\huggingface\assets"
    HF_XET_CACHE = "$domainTuneStorage\cache\huggingface\xet"
    TORCH_HOME = "$domainTuneStorage\cache\torch"
    TORCH_EXTENSIONS_DIR = "$domainTuneStorage\cache\torch-extensions"
    TORCHINDUCTOR_CACHE_DIR = "$domainTuneStorage\cache\torchinductor"
    TRITON_CACHE_DIR = "$domainTuneStorage\cache\triton"
    CUDA_CACHE_PATH = "$domainTuneStorage\cache\cuda"
    XDG_CACHE_HOME = "$domainTuneStorage\cache"
    MPLCONFIGDIR = "$domainTuneStorage\cache\matplotlib"
    TEMP = "$domainTuneStorage\tmp"
    TMP = "$domainTuneStorage\tmp"
    TMPDIR = "$domainTuneStorage\tmp"
}
foreach ($domainTuneEntry in $domainTunePaths.GetEnumerator()) {
    New-Item -ItemType Directory -Force -Path $domainTuneEntry.Value | Out-Null
    [Environment]::SetEnvironmentVariable($domainTuneEntry.Key, $domainTuneEntry.Value, 'Process')
}
$env:UV_LINK_MODE = 'copy'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:DO_NOT_TRACK = '1'
$env:MLFLOW_TRACKING_URI = ([Uri](Join-Path $domainTuneRoot 'mlruns')).AbsoluteUri
Write-Host "DomainTune storage configured on D: ($domainTuneStorage)"
