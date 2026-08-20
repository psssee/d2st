# Windows local development machine only. Do not run on the Linux training server.
[CmdletBinding()]
param(
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    # This file lives in <repo>/scripts/windows.
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & git -C $RepoRoot @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: git -C `"$RepoRoot`" $($Arguments -join ' ')"
    }
}

$trackedLogs = @(
    & git -C $RepoRoot ls-files --cached --full-name -- output |
        Where-Object { $_ -match '(?i)\.log$' } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to list tracked files under output."
}

if ($trackedLogs.Count -eq 0) {
    Write-Warning "No tracked output/*.log files found. Untracked logs are intentionally ignored."
    exit 0
}

foreach ($relativePath in $trackedLogs) {
    Invoke-Git @("update-index", "--skip-worktree", "--", $relativePath)
}

Write-Host "Marked $($trackedLogs.Count) tracked output log file(s) as skip-worktree."
Write-Host "Untracked .log files were not changed. Checkpoints and .pth files were not touched."
