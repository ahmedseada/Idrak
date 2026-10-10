<#
.SYNOPSIS
    Renders text into OCR training pairs: page images with their text (a ShareGPT train.json / val.json for `idrak tune`)
    and line images with a .txt beside each (for the ArabicOcr sample's `train`).

.DESCRIPTION
    The text is laid out by a headless Edge or Chrome (it shapes Arabic: joined letters, ligatures, right to left) in
    fonts installed on this machine, with sizes, margins, spacing and fonts varied page to page. Every document (a block
    of the input separated by a blank line) starts on a new page; its first paragraph is a centered heading.

    -Text: a UTF-8 text file, one paragraph per line, documents separated by blank lines; or a .json file holding an
           array of strings, each string one document (its lines its paragraphs).
    -Html: a saved HTML page; its text is taken (scripts and styles dropped, block elements as paragraphs).

    Output in -Out:
      pages\page-BBBBB-PP.jpg        page images (a little blurred, rotated and noisy unless -Clean)
      train.json, val.json           ShareGPT records: "<image>" + -Prompt, the page's text as the answer
      lines\NAME-line-NN.png + .txt  line images of the training pages, clean (the trainer's --augment adds damage)
      lines-val\...                  line images of the held-out pages
    Every -ValEvery'th batch of documents is held out, so no document has pages on both sides.
    Run again to continue: finished batches (batches\*.done) are kept, and train.json / val.json are rewritten.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\ocr\render-text.ps1 -Text D:\Data\boe\laws.txt -Out D:\Data\boe\rendered -MaxPages 50
#>
param(
    [string]$Text,
    [string]$Html,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Browser,
    [string[]]$Fonts = @("Arial", "Times New Roman", "Tahoma", "Segoe UI", "Traditional Arabic", "Simplified Arabic",
        "Sakkal Majalla", "Arabic Typesetting", "Microsoft Uighur", "Aldhabi", "Andalus", "Courier New",
        "Amiri", "Noto Naskh Arabic", "Noto Sans Arabic", "Noto Kufi Arabic", "KacstBook", "KacstOne", "KacstOffice"),
    [string]$Prompt = "استخرج النص من هذه الصورة كما هو، سطراً بسطر.",
    [int]$MaxPages = 0,
    [int]$CharsPerBatch = 12000,
    [int]$ValEvery = 20,
    [int]$Parallel = 4,
    [int]$Seed = 1,
    [switch]$Clean,
    [switch]$NoLines
)

$ErrorActionPreference = "Stop"
$utf8 = New-Object System.Text.UTF8Encoding($false)
if (-not $Text -and -not $Html) { throw "Give -Text FILE or -Html FILE." }
$source = if ($Text) { $Text } else { $Html }
$source = (Resolve-Path $source).Path
$Out = [System.IO.Path]::GetFullPath($Out)

if (-not $Browser) {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe")
    $Browser = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $Browser) { throw "No Edge or Chrome found; give -Browser PATH." }
}

# --- The text: documents of paragraphs -------------------------------------------------------------------------------
$raw = [System.IO.File]::ReadAllText($source, [System.Text.Encoding]::UTF8)
if ($Text -and [System.IO.Path]::GetExtension($source) -eq ".json") {
    $raw = (($raw | ConvertFrom-Json) | ForEach-Object { ([string]$_).Replace("`n`n", "`n") }) -join "`n`n"
}
if ($Html) {
    $o = [System.Text.RegularExpressions.RegexOptions]'IgnoreCase, Singleline'
    $raw = [regex]::Replace($raw, '<(script|style|head|noscript)\b.*?</\1\s*>', ' ', $o)
    $raw = [regex]::Replace($raw, '<br\s*/?>|</(p|div|li|h[1-6]|tr|td|th|section|article|table|ul|ol|pre|blockquote)\s*>', "`n", $o)
    $raw = [regex]::Replace($raw, '<(hr|article|section)\b[^>]*>', "`n`n", $o)
    $raw = [regex]::Replace($raw, '<[^>]+>', ' ', $o)
    $raw = [System.Net.WebUtility]::HtmlDecode($raw)
}
# Logical Unicode as typed: NFC, no invisible direction marks or zero-width characters, white space as single spaces.
$raw = $raw.Normalize([System.Text.NormalizationForm]::FormC)
$raw = [regex]::Replace($raw, '[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u00AD]', '')
$raw = [regex]::Replace($raw, '[^\S\n]+', ' ')

$docs = New-Object System.Collections.Generic.List[object]
$current = New-Object System.Collections.Generic.List[string]
foreach ($line in ($raw -split "\r?\n")) {
    $t = $line.Trim()
    if ($t) { $current.Add($t) }
    elseif ($current.Count -gt 0) { $docs.Add($current.ToArray()); $current.Clear() }
}
if ($current.Count -gt 0) { $docs.Add($current.ToArray()) }
if ($docs.Count -eq 0) { throw "No text in $source." }

# Batches of about -CharsPerBatch characters; a long document is split at a paragraph and continues in the next batch.
$batches = New-Object System.Collections.Generic.List[object]
$batch = New-Object System.Collections.Generic.List[object]
$size = 0
foreach ($doc in $docs) {
    $part = New-Object System.Collections.Generic.List[string]
    $cont = $false
    foreach ($p in $doc) {
        $part.Add($p); $size += $p.Length
        if ($size -ge $CharsPerBatch) {
            $batch.Add(@{ paras = $part.ToArray(); cont = $cont }); $batches.Add($batch.ToArray())
            $batch.Clear(); $part.Clear(); $size = 0; $cont = $true
        }
    }
    if ($part.Count -gt 0) { $batch.Add(@{ paras = $part.ToArray(); cont = $cont }) }
}
if ($batch.Count -gt 0) { $batches.Add($batch.ToArray()) }
$chars = ($docs | ForEach-Object { $_ } | Measure-Object -Property Length -Sum).Sum
Write-Host ("{0:N0} documents, {1:N0} characters, {2:N0} batches; browser {3}" -f $docs.Count, $chars, $batches.Count, $Browser)

function Quote([string]$s) {
    $b = New-Object System.Text.StringBuilder($s.Length + 2)
    [void]$b.Append('"')
    foreach ($c in $s.ToCharArray()) {
        switch ($c) {
            '"' { [void]$b.Append('\"') } '\' { [void]$b.Append('\\') }
            '<' { [void]$b.Append('\u003c') } '>' { [void]$b.Append('\u003e') } '&' { [void]$b.Append('\u0026') }
            default { if ([int]$c -lt 32) { [void]$b.Append(('\u{0:x4}' -f [int]$c)) } else { [void]$b.Append($c) } }
        }
    }
    [void]$b.Append('"')
    $b.ToString()
}

$script = [System.IO.File]::ReadAllText((Join-Path $PSScriptRoot "render.js"), $utf8)
foreach ($d in "pages", "lines", "lines-val", "batches", "work") { [void](New-Item -ItemType Directory -Force (Join-Path $Out $d)) }
$work = Join-Path $Out "work"

function Start-Batch([int]$index) {
    $docsJson = ($batches[$index] | ForEach-Object {
        '{"paras":[' + (($_.paras | ForEach-Object { Quote $_ }) -join ",") + '],"cont":' + $(if ($_.cont) { "true" } else { "false" }) + '}'
    }) -join ","
    $job = '{"docs":[' + $docsJson + '],"seed":' + ($Seed * 100003 + $index) + ',"fonts":[' + (($Fonts | ForEach-Object { Quote $_ }) -join ",") +
        '],"degrade":' + $(if ($Clean) { "false" } else { "true" }) + ',"lines":' + $(if ($NoLines) { "false" } else { "true" }) + '}'
    $page = "<!doctype html><html><head><meta charset=""utf-8""></head><body><pre id=""out""></pre><script>window.ocrBatch = $job;</script><script>$script</script></body></html>"
    $file = Join-Path $work ("batch-{0:D5}.html" -f $index)
    [System.IO.File]::WriteAllText($file, $page, $utf8)
    $userData = Join-Path $work ("profile-{0:D5}" -f $index)
    $result = Join-Path $work ("batch-{0:D5}.out" -f $index)
    $uri = ([System.Uri]::new($file, [System.UriKind]::Absolute)).AbsoluteUri
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Browser
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.CreateNoWindow = $true
    $arguments = @("--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--disable-extensions",
        "--allow-file-access-from-files", "--disable-background-networking", "--disable-component-update",
        "--user-data-dir=$userData", "--dump-dom", $uri)
    if ($IsLinux -or $IsMacOS) { $arguments = @("--no-sandbox") + $arguments }
    if ($info.PSObject.Properties["ArgumentList"]) { foreach ($x in $arguments) { $info.ArgumentList.Add($x) } }
    else { $info.Arguments = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join " " } # Windows PowerShell 5.1
    $process = [System.Diagnostics.Process]::Start($info)
    $output = $process.StandardOutput.ReadToEndAsync()
    $errors = $process.StandardError.ReadToEndAsync()
    @{ index = $index; process = $process; output = $output; errors = $errors; result = $result; profile = $userData; file = $file }
}

function Complete-Batch($run) {
    $run.process.WaitForExit()
    $index = $run.index
    $val = $ValEvery -gt 0 -and ($index % $ValEvery) -eq ($ValEvery - 1)
    $lineFolder = Join-Path $Out $(if ($val) { "lines-val" } else { "lines" })
    $records = New-Object System.Collections.Generic.List[string]
    $pages = 0
    $dom = $run.output.Result
    $from = $dom.IndexOf('<pre id="out">')
    $to = $dom.IndexOf('</pre>')
    $body = if ($from -ge 0 -and $to -gt $from) { $dom.Substring($from + 14, $to - $from - 14) } else { "" }
    foreach ($row in ($body -split "`r?`n")) {
        $f = $row.Split("`t")
        if ($f[0] -eq "E") { throw $f[1] }
        $name = "page-{0:D5}-{1:D2}" -f $index, ([int]$f[1] + 1)
        if ($f[0] -eq "L" -and $f.Count -eq 5) {
            $line = "{0}-line-{1:D2}" -f $name, ([int]$f[2] + 1)
            [System.IO.File]::WriteAllBytes((Join-Path $lineFolder "$line.png"), [Convert]::FromBase64String($f[4]))
            [System.IO.File]::WriteAllText((Join-Path $lineFolder "$line.txt"), $utf8.GetString([Convert]::FromBase64String($f[3])), $utf8)
        }
        elseif ($f[0] -eq "P" -and $f.Count -eq 4) {
            [System.IO.File]::WriteAllBytes((Join-Path $Out "pages\$name.jpg"), [Convert]::FromBase64String($f[3]))
            $answer = $utf8.GetString([Convert]::FromBase64String($f[2]))
            $records.Add('{"messages":[{"role":"user","content":' + (Quote ("<image>" + $Prompt)) + '},{"role":"assistant","content":' +
                (Quote $answer) + '}],"images":["pages/' + $name + '.jpg"]}')
            $pages++
        }
    }
    if ($pages -eq 0) {
        [System.IO.File]::WriteAllText($run.result, $run.output.Result)
        [System.IO.File]::WriteAllText(($run.result -replace '\.out$', '.err'), $run.errors.Result)
        throw "Batch $index produced no pages; see $($run.result) and the .err beside it."
    }
    [System.IO.File]::WriteAllLines((Join-Path $Out ("batches\{0:D5}.done" -f $index)), $records, $utf8)
    Remove-Item -Recurse -Force $run.profile, $run.file -ErrorAction SilentlyContinue
    $pages
}

$total = 0
foreach ($done in Get-ChildItem (Join-Path $Out "batches") -Filter *.done) { $total += ([System.IO.File]::ReadAllLines($done.FullName)).Count }
$running = New-Object System.Collections.Generic.List[object]
$next = 0
$clock = [System.Diagnostics.Stopwatch]::StartNew()
while ($next -lt $batches.Count -or $running.Count -gt 0) {
    while ($running.Count -lt [Math]::Max(1, $Parallel) -and $next -lt $batches.Count -and ($MaxPages -le 0 -or $total -lt $MaxPages)) {
        if (-not (Test-Path (Join-Path $Out ("batches\{0:D5}.done" -f $next)))) { $running.Add((Start-Batch $next)) }
        $next++
    }
    if ($running.Count -eq 0) { break }
    $run = $running[0]; $running.RemoveAt(0)
    $total += Complete-Batch $run
    Write-Host ("batch {0}/{1}: {2:N0} pages so far, {3:N0} s" -f ($run.index + 1), $batches.Count, $total, $clock.Elapsed.TotalSeconds)
}

# train.json and val.json from every finished batch, in order.
$train = New-Object System.Collections.Generic.List[string]
$heldOut = New-Object System.Collections.Generic.List[string]
foreach ($done in Get-ChildItem (Join-Path $Out "batches") -Filter *.done | Sort-Object Name) {
    $index = [int]$done.BaseName
    $target = $train # (assigned, not an if expression: that would unroll an empty list to $null)
    if ($ValEvery -gt 0 -and ($index % $ValEvery) -eq ($ValEvery - 1)) { $target = $heldOut }
    foreach ($r in [System.IO.File]::ReadAllLines($done.FullName, $utf8)) { if ($r) { $target.Add($r) } }
}
[System.IO.File]::WriteAllText((Join-Path $Out "train.json"), "[`n" + ($train -join ",`n") + "`n]`n", $utf8)
[System.IO.File]::WriteAllText((Join-Path $Out "val.json"), "[`n" + ($heldOut -join ",`n") + "`n]`n", $utf8)
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
Write-Host ("{0:N0} training pages, {1:N0} held out; lines in {2}" -f $train.Count, $heldOut.Count, (Join-Path $Out "lines"))
