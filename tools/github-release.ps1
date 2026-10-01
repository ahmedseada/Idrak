# Creates the GitHub release of a version: tag v<Version>, title "Idrak <Version>", the version's CHANGELOG section as
# the notes, and the packages attached when given. tools/publish.ps1 calls it after pushing to nuget.org; on its own it
# backfills a release (Actions not running, or a release published from a machine):
#
#   pwsh tools/github-release.ps1 -Version 0.1.5                      (notes only)
#   pwsh tools/github-release.ps1 -Version 0.1.5 -Packages artifacts   (and the .nupkg files in that folder)
#
# Needs the GitHub CLI (winget install GitHub.cli; gh auth login). A release that exists already is left as it is.
param(
    [Parameter(Mandatory)][string]$Version,
    [string]$Packages,
    [switch]$NotesOnly                                   # print the notes and stop (to check them)
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$tag = "v$Version"

# The section of CHANGELOG.md under "## <Version>" up to the next "## " heading. A version that was never published
# (its heading says so) has its changes in the next one: those sections are added to its notes.
function Get-Notes([string]$version) {
    $text = [System.IO.File]::ReadAllText((Join-Path (Get-Location) 'CHANGELOG.md'))
    $sections = [regex]::Matches($text, '(?ms)^## (\S+)([^\r\n]*)\r?\n(.*?)(?=^## |\z)')
    $notes = @()
    $collect = $false
    for ($i = $sections.Count - 1; $i -ge 0; $i--) {
        $section = $sections[$i]
        $name, $rest, $body = $section.Groups[1].Value, $section.Groups[2].Value, $section.Groups[3].Value.Trim()
        if ($rest -match 'never published') { $notes += "### From $name (not published on its own)`n`n$body"; $collect = $true; continue }
        if ($name -eq $version) { return (@($body) + $notes) -join "`n`n" }
        if ($collect) { $notes = @(); $collect = $false }
    }

    throw "CHANGELOG.md has no section for $version."
}

$notes = Get-Notes $Version
if ($NotesOnly) { $notes; return }

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw 'The GitHub CLI (gh) is not installed: winget install GitHub.cli, then gh auth login.' }
git fetch origin --tags --quiet
if (-not (git tag --list $tag)) { throw "There is no tag $tag." }
gh release view $tag *> $null
if ($LASTEXITCODE -eq 0) { Write-Host "The release $tag exists already: left as it is."; return }

$file = New-TemporaryFile
try {
    [System.IO.File]::WriteAllText($file.FullName, $notes)
    $assets = if ($Packages) { @(Get-ChildItem $Packages -Filter "*.$Version.nupkg" | ForEach-Object FullName) } else { @() }
    gh release create $tag --verify-tag --title "Idrak $Version" --notes-file $file.FullName @assets
    if ($LASTEXITCODE -ne 0) { throw 'gh release create failed.' }
    Write-Host "Created the GitHub release $tag$(if ($assets.Count) { " with $($assets.Count) packages" })."
}
finally {
    Remove-Item $file -ErrorAction SilentlyContinue
}
