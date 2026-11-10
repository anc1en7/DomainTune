# Run one authorized phase locally, verify it, publish small evidence, then stop.
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
    Invoke-Checked $domainTunePython @('scripts/train_peft.py', '--config', 'configs/lora_r8.yaml')
    $domainTuneRun = Get-Content 'outputs/adapters/phase3-r8/run.json' -Raw | ConvertFrom-Json
    Invoke-Checked $domainTunePython @('scripts/evaluate_model.py', '--checkpoint', $domainTuneRun.selected_checkpoint, '--split', 'validation')
    # Avoid mixing new user edits into automated report publication.
    $domainTuneChanges = & git status --porcelain
    if ($LASTEXITCODE -ne 0 -or $domainTuneChanges) { throw 'Git changed during training; review before publishing results.' }
    Invoke-Checked $domainTunePython @('scripts/summarize_training.py')
    Invoke-Checked $domainTunePython @('-m', 'pytest', '-q', '-p', 'no:cacheprovider')
    Invoke-Checked $domainTuneRuff @('check', '.')
    Invoke-Checked $domainTuneRuff @('format', '--check', '.')
    Invoke-Checked 'git' @('diff', '--check')
    Invoke-Checked 'git' @('add', 'reports/phase3-run.json', 'reports/phase3-validation.json', 'reports/phase3-training.png', 'reports/training.md', 'reports/experiment_log.md', 'docs/session-handoff.md', 'README.md')
    Invoke-Checked 'git' @('commit', '-m', 'Complete Phase 3 primary LoRA experiment with verified adapter reload')
    Invoke-Checked 'git' @('push', 'origin', 'main')
    'Phase 3 complete and pushed. Stop before Phase 4.' | Set-Content 'outputs/phase3-worker-status.txt'
} catch {
    $_ | Out-String | Set-Content 'outputs/phase3-worker-status.txt'
    throw
} finally {
    [DomainTune.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
