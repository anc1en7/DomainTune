# Run one authorized phase locally, verify it, publish small evidence, then stop.
param(
    [string]$RunDirectory = 'outputs/adapters/phase3-r8-continuation',
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
. "$PSScriptRoot\environment.ps1"
$domainTunePython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$domainTuneRuff = Join-Path (Get-Location) '.venv\Scripts\ruff.exe'
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Executable (exit $LASTEXITCODE)" }
}
try {
    # Process-scoped sleep inhibition expires when this worker exits.
    Add-Type -Namespace DomainTune -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
    [DomainTune.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    if (-not $VerifyOnly) {
        Invoke-Checked $domainTunePython @('scripts/train_peft.py', '--config', 'configs/lora_r8_continuation.yaml', '--output', $RunDirectory, '--initialize-from', 'outputs/adapters/phase3-r8/epoch-1', '--start-epoch', '2', '--initial-validation', 'outputs/phase3-epoch1-validation/reload-validation.json')
    }
    $domainTuneRun = Get-Content (Join-Path $RunDirectory 'run.json') -Raw | ConvertFrom-Json
    if ($domainTuneRun.status -ne 'trained') {
        "Phase 3 stopped safely: $($domainTuneRun.status). No evaluation or publishing started." | Set-Content 'outputs/phase3-worker-status.txt'
        return
    }
    if (-not (Test-Path (Join-Path $RunDirectory 'reload-validation.json'))) {
        Invoke-Checked $domainTunePython @('scripts/evaluate_model.py', '--checkpoint', $domainTuneRun.selected_checkpoint, '--split', 'validation')
    }
    # Avoid mixing new user edits into automated report publication.
    $domainTuneChanges = & git status --porcelain
    if ($LASTEXITCODE -ne 0 -or $domainTuneChanges) { throw 'Git changed during training; review before publishing results.' }
    Invoke-Checked $domainTunePython @('scripts/summarize_training.py', '--run', $RunDirectory)
    Invoke-Checked $domainTunePython @('-m', 'pytest', '-q', '-p', 'no:cacheprovider')
    Invoke-Checked $domainTuneRuff @('check', '.')
    Invoke-Checked $domainTuneRuff @('format', '--check', '.')
    Invoke-Checked 'git' @('diff', '--check')
    Invoke-Checked 'git' @('add', 'reports/phase3-run.json', 'reports/phase3-validation.json', 'reports/phase3-training.png', 'reports/training.md', 'reports/experiment_log.md', 'docs/session-handoff.md', 'README.md')
    Invoke-Checked 'git' @('commit', '-m', 'Complete Phase 3 reset-state LoRA continuation with verified adapter reload')
    Invoke-Checked 'git' @('push', 'origin', 'main')
    'Phase 3 complete and pushed. Stop before Phase 4.' | Set-Content 'outputs/phase3-worker-status.txt'
} catch {
    $_ | Out-String | Set-Content 'outputs/phase3-worker-status.txt'
    throw
} finally {
    [DomainTune.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
