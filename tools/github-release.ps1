# Creates the GitHub release of a version: tag v<Version>, title "Idrak <Version>", the version's CHANGELOG section as
# the notes, and the packages attached when given. tools/publish.ps1 calls it after pushing to nuget.org; on its own it
# backfills a release (Actions not running, or a release published from a machine):
#
#   powershell -ExecutionPolicy Bypass -File tools\github-release.ps1 -Version 0.4.0                       (no files attached)
#   powershell -ExecutionPolicy Bypass -File tools\github-release.ps1 -Version 0.4.0 -Packages artifacts   (and the .nupkg files)
#   ... -NotesOnly                                                        (print the release text and stop, to check it)
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
# Windows PowerShell turns a native command's redirected error output into errors, which 'Stop' would throw: gh says
# "release not found" there when there is none yet.
$ErrorActionPreference = 'Continue'
gh release view $tag *> $null
$exists = $LASTEXITCODE -eq 0
$ErrorActionPreference = 'Stop'
if ($exists) { Write-Host "The release $tag exists already: left as it is (delete it on GitHub to make it again)."; return }

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
