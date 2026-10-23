# Publishes a release from this machine: every package of the tag v<Version> to nuget.org, then the GitHub release (the
# version's CHANGELOG section as its text, the packages attached). The Release workflow runs only by hand, so pushing a
# tag publishes nothing; this script is the normal way to publish.
#
#   powershell -ExecutionPolicy Bypass -File tools\publish.ps1 -Version 0.4.0               (Windows PowerShell)
#   pwsh -File tools/publish.ps1 -Version 0.4.0                                              (PowerShell 7)
#   ... -SkipGitHub     nuget.org only           ... -SkipNuGet     the GitHub release only (packs again to attach them)
#
# The nuget.org API key (API Keys > Create: scope Push, glob Idrak*, short expiry) comes from -ApiKey, else from the
# NUGET_API_KEY environment variable, else it is asked for; it is never stored. The GitHub release needs the GitHub CLI
# (winget install GitHub.cli; gh auth login). The script checks out the tag, checks that its VersionPrefix is <Version>,
# packs every project under src/ with that version, shows what it will push, pushes after a confirmation, creates the
# GitHub release (tools/github-release.ps1), and returns to the branch it started on.
param(
    [Parameter(Mandatory)][string]$Version,
    [string]$ApiKey,
    [switch]$SkipNuGet,
    [switch]$SkipGitHub
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$Version = $Version.Trim().TrimStart('v')
if ($SkipNuGet -and $SkipGitHub) { throw 'Both -SkipNuGet and -SkipGitHub: there is nothing to do.' }

if (git status --porcelain) { throw 'The working tree has changes: commit or stash them first.' }
$tag = "v$Version"
git fetch origin --tags --quiet
if ($LASTEXITCODE -ne 0) { throw 'git fetch failed.' }
if (-not (git tag --list $tag)) { throw "There is no tag $tag (git tag $tag; git push origin $tag)." }
if (-not $SkipGitHub -and -not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'The GitHub CLI (gh) is not installed: winget install GitHub.cli, then gh auth login (or run with -SkipGitHub).'
}

$branch = git rev-parse --abbrev-ref HEAD
$artifacts = Join-Path (Get-Location) 'artifacts'
git checkout --quiet $tag
if ($LASTEXITCODE -ne 0) { throw "git checkout $tag failed." }
try {
    $prefix = ([xml](Get-Content -Raw Directory.Build.props)).Project.PropertyGroup.VersionPrefix | Where-Object { $_ } | Select-Object -First 1
    if ($prefix -ne $Version) { throw "The tag $tag has VersionPrefix $prefix in Directory.Build.props, not $Version." }

    if (Test-Path $artifacts) { Remove-Item -Recurse -Force $artifacts }
    $projects = @(Get-ChildItem src -Recurse -Filter *.csproj)
    $i = 0
    foreach ($project in $projects) {
        $i++
        Write-Host "[$i/$($projects.Count)] Packing $($project.BaseName) $Version"
        dotnet pack $project.FullName -c Release "-p:Version=$Version" -o $artifacts --nologo -v quiet
        if ($LASTEXITCODE -ne 0) { throw "dotnet pack failed for $($project.Name)." }
    }

    $packages = @(Get-ChildItem $artifacts -Filter *.nupkg)
    if ($packages.Count -ne $projects.Count) { throw "$($packages.Count) packages for $($projects.Count) projects." }
    if ($packages | Where-Object { $_.Name -notlike "*.$Version.nupkg" }) { throw "A package does not have version $Version." }

    if (-not $SkipNuGet) {
        Write-Host "`nReady to publish to nuget.org (this cannot be undone):"
        $packages | ForEach-Object { Write-Host "  $($_.Name)" }
        $answer = (Read-Host "`nType the version ($Version) and press Enter to publish").Trim().TrimStart('v')
        if ($answer -ne $Version) { throw "Not confirmed (got '$answer'): nothing was published." }

        $key = if ($ApiKey) { $ApiKey } elseif ($env:NUGET_API_KEY) { $env:NUGET_API_KEY } else {
            [System.Net.NetworkCredential]::new('', (Read-Host 'nuget.org API key' -AsSecureString)).Password
        }
        try {
            # The .snupkg symbol packages beside the .nupkg files are pushed with them.
            dotnet nuget push (Join-Path $artifacts '*.nupkg') --api-key $key --source https://api.nuget.org/v3/index.json --skip-duplicate
            if ($LASTEXITCODE -ne 0) { throw 'dotnet nuget push failed.' }
        }
        finally {
            $key = $null
        }

        Write-Host "`nPublished $($packages.Count) packages. They appear on nuget.org after validation (usually 10-30 minutes)."
        Write-Host 'Delete the API key on nuget.org now.'
    }

    if (-not $SkipGitHub) {
        & (Join-Path $PSScriptRoot 'github-release.ps1') -Version $Version -Packages $artifacts
    }
}
finally {
    git checkout --quiet $branch
}
