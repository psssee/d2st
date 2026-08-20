# Windows local development machine only. Do not run on the Linux training server.
[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$Remote = "origin",
    [string]$Branch = "main"
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

function Get-TrackedLogs {
    @(
        & git -C $RepoRoot ls-files --cached --full-name -- output |
            Where-Object { $_ -match '(?i)\.log$' } |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to list tracked files under output."
    }
}

$currentBranch = (& git -C $RepoRoot branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to determine the current branch."
}
if ($currentBranch -ne $Branch) {
    throw "Current branch is '$currentBranch'; expected '$Branch'. Pull was not run."
}

$trackedLogs = @(Get-TrackedLogs)
$locallyChangedLogs = @()
foreach ($relativePath in $trackedLogs) {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $relativePath) -PathType Leaf)) {
        $locallyChangedLogs += $relativePath
        continue
    }

    $worktreeHash = (& git -C $RepoRoot hash-object -- $relativePath).Trim()
    $headHash = (& git -C $RepoRoot rev-parse "HEAD:$relativePath").Trim()
    if ($LASTEXITCODE -ne 0 -or $worktreeHash -ne $headHash) {
        $locallyChangedLogs += $relativePath
    }
}

if ($locallyChangedLogs.Count -gt 0) {
    Write-Error "Local edits were found in tracked output log(s). Pull was aborted to avoid losing them:"
    $locallyChangedLogs | ForEach-Object { Write-Error "  $_" }
    Write-Host "To inspect or preserve them, run disable-output-log-skip-worktree.ps1 first."
    exit 1
}

$disableScript = Join-Path $PSScriptRoot "disable-output-log-skip-worktree.ps1"
$enableScript = Join-Path $PSScriptRoot "enable-output-log-skip-worktree.ps1"

try {
    & $disableScript -RepoRoot $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to clear skip-worktree before pull."
    }

    Invoke-Git @("pull", "--ff-only", $Remote, $Branch)
}
finally {
    & $enableScript -RepoRoot $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Pull finished, but skip-worktree could not be restored. Run enable-output-log-skip-worktree.ps1 manually."
    }
}

Write-Host "Pull completed and skip-worktree was restored for tracked output logs."
