# Publishes every package of a release tag to nuget.org from this machine (when the Release workflow cannot run).
#
#   pwsh tools/publish.ps1 -Version 0.1.0          (or: powershell -File tools\publish.ps1 -Version 0.1.0)
#
# Needs a nuget.org API key (API Keys > Create: scope Push, glob Idrak*, short expiry); it is asked for, not stored.
# The script checks out the tag v<Version>, packs every project under src/ with that version, shows what it will push,
# pushes after a confirmation, and returns to the branch it started on.
param([Parameter(Mandatory)][string]$Version)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

if (git status --porcelain) { throw 'The working tree has changes: commit or stash them first.' }
$tag = "v$Version"
git fetch origin --tags --quiet
if (-not (git tag --list $tag)) { throw "There is no tag $tag." }

$branch = git rev-parse --abbrev-ref HEAD
$artifacts = Join-Path (Get-Location) 'artifacts'
git checkout --quiet $tag
try {
    if (Test-Path $artifacts) { Remove-Item -Recurse -Force $artifacts }
    foreach ($project in Get-ChildItem src -Recurse -Filter *.csproj) {
        dotnet pack $project.FullName -c Release "-p:Version=$Version" -o $artifacts
        if ($LASTEXITCODE -ne 0) { throw "dotnet pack failed for $($project.Name)." }
    }

    $packages = Get-ChildItem $artifacts -Filter *.nupkg
    $projects = (Get-ChildItem src -Recurse -Filter *.csproj).Count
    if ($packages.Count -ne $projects) { throw "$($packages.Count) packages for $projects projects." }
    if ($packages | Where-Object { $_.Name -notlike "*.$Version.nupkg" }) { throw "A package does not have version $Version." }

    Write-Host "`nReady to publish to nuget.org (this cannot be undone):"
    $packages | ForEach-Object { Write-Host "  $($_.Name)" }
    $answer = (Read-Host "`nType the version ($Version) and press Enter to publish").Trim().TrimStart('v')
    if ($answer -ne $Version) { throw "Not confirmed (got '$answer'): nothing was published." }

    $secure = Read-Host 'nuget.org API key' -AsSecureString
    $key = [System.Net.NetworkCredential]::new('', $secure).Password
    try {
        dotnet nuget push (Join-Path $artifacts '*.nupkg') --api-key $key --source https://api.nuget.org/v3/index.json --skip-duplicate
        if ($LASTEXITCODE -ne 0) { throw 'dotnet nuget push failed.' }
    }
    finally {
        $key = $null
    }

    Write-Host "`nPublished $($packages.Count) packages. They appear on nuget.org after validation (usually 10-30 minutes)."
    Write-Host 'Delete the API key on nuget.org now.'
}
finally {
    git checkout --quiet $branch
}
