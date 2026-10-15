<#
.SYNOPSIS
  Plan 19, B6: a vision-language model of your own, from scratch, taught to read in three stages (single words, two or
  three words, full lines).

.DESCRIPTION
  Builds the CLI and the IdrakVision plug-in, then makes whatever the work folder is missing, in order:
    1. the text corpus, its tokenizer and token file (idrak data text, data tokenizer, data tokens);
    2. the text model (A5: idrak create, then pretrain);
    3. the vision model on it (create init --vision at -ImageSize pixels, create --decoder);
    4. the fonts (Arial and Times New Roman from Windows, or -Fonts);
    5. the reading data: clean pages cut to their ink (data synth --page crop=1), training and held-out sets;
    6. the three stages, each from the previous stage's checkpoint, every weight trained, each evaluated by its
       character error rate (tune evaluate --metric cer) on held-out images.
  What exists is kept, and a training run that stopped is resumed, so rerunning it is safe. Everything it prints also
  goes to <work>\b6.log.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File samples\IdrakVision\scripts\b6-curriculum.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File samples\IdrakVision\scripts\b6-curriculum.ps1 -Device cuda:0 -Fonts D:\fonts
#>
param(
    [string]$Device = "cuda:0",
    [string]$Work = "work",
    [string]$Fonts = "",
    [int]$CorpusPages = 20000,
    [int]$Vocab = 32000,
    [int]$TextSteps = 2000,
    [int]$ImageSize = 448,
    [int]$Words = 40000,
    [int]$Lines = 30000,
    [int]$Held = 200,
    [int]$Steps1 = 3000,
    [int]$Steps2 = 3000,
    [int]$Steps3 = 4000,
    [string]$Rate1 = "1e-3",
    [string]$Rate2 = "5e-4",
    [string]$Rate3 = "3e-4",
    [string]$Page = "size=90,crop=1,crop-padding=12..24"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
Set-Location $root
New-Item -ItemType Directory -Force $Work | Out-Null
$Work = (Resolve-Path $Work).Path
$utf8 = New-Object System.Text.UTF8Encoding $false
Start-Transcript -Path (Join-Path $Work "b6.log") -Append | Out-Null

function Step([string]$what) { Write-Host ""; Write-Host "== $what" -ForegroundColor Cyan }
function W([string]$name) { Join-Path $Work $name }
function Idrak {
    & dotnet $script:idrak @args
    if ($LASTEXITCODE -ne 0) { throw "idrak $($args -join ' ') failed (exit code $LASTEXITCODE)" }
}

# The run's last checkpoint folder, or $null.
function Last([string]$run) {
    $c = Get-ChildItem $run -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1
    if ($c) { $c.FullName } else { $null }
}

# Trains $from into $out for $steps steps: done when its last step's checkpoint is there, resumed when it stopped after
# a checkpoint, started again when it stopped before one.
function Train([string]$from, [string]$out, [int]$steps, [string[]]$more) {
    if (Test-Path (Join-Path $out "checkpoint-$steps")) { Write-Host "$out is done ($steps steps)"; return }
    if ((Test-Path (Join-Path $out "run.json")) -and (Last $out)) {
        Idrak resume $out -d $Device -P $script:plugin
        return
    }

    Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
    Idrak pretrain $from -o $out --steps $steps -d $Device @more
}

# The corpus's text lines (no blank lines, no "---" document breaks).
function TextLines([string]$path) {
    [IO.File]::ReadAllLines($path, $utf8) | Where-Object { $_.Trim().Length -gt 0 -and $_.Trim() -ne '---' }
}

# Documents (one per page) written between "---" lines.
function WriteDocs([string]$path, [string[]]$docs) {
    [IO.File]::WriteAllText($path, ($docs -join "`n---`n") + "`n", $utf8)
    Write-Host "$path : $($docs.Count) documents"
}

try {
    Step "Building the command-line tool and the IdrakVision plug-in"
    dotnet build -c Release src/Idrak.Cli
    if ($LASTEXITCODE -ne 0) { throw "building src/Idrak.Cli failed" }
    dotnet build -c Release samples/IdrakVision/Idrak.IdrakVision
    if ($LASTEXITCODE -ne 0) { throw "building the IdrakVision plug-in failed" }
    $script:idrak = Join-Path $root "src/Idrak.Cli/bin/Release/net10.0/Idrak.Cli.dll"
    $script:plugin = Join-Path $root "samples/IdrakVision/Idrak.IdrakVision/bin/Release/net10.0/Idrak.IdrakVision.dll"

    Step "1. Text: corpus, tokenizer, token file"
    if (-not (Test-Path (W "corpus.txt"))) { Idrak data text -n $CorpusPages --unit pages -o (W "corpus.txt") }
    if (-not (Test-Path (W "tokenizer.json"))) { Idrak data tokenizer (W "corpus.txt") --vocab $Vocab -o (W "tokenizer.json") }
    if (-not (Test-Path (W "tokens.bin"))) { Idrak data tokens (W "corpus.txt") --tokenizer (W "tokenizer.json") -o (W "tokens.bin") }

    Step "2. The text model (A5)"
    if (-not (Test-Path (W "spec.json"))) { Idrak create init -o (W "spec.json") }
    if (-not (Test-Path (W "model/config.json"))) { Idrak create (W "spec.json") -o (W "model") }
    Train (W "model") (W "run") $TextSteps @("--tokens", (W "tokens.bin"), "--lr", "6e-4")
    $text = Last (W "run")
    Write-Host "text model: $text"

    Step "3. The vision model on it ($ImageSize pixels)"
    $visionSpec = W "vision$ImageSize-spec.json"
    if (-not (Test-Path $visionSpec)) {
        if (-not (Test-Path (W "vision-spec.json"))) { Idrak create init --vision -o (W "vision-spec.json") }
        $spec = [IO.File]::ReadAllText((W "vision-spec.json"), $utf8) -replace '"image_size": \d+', ('"image_size": ' + $ImageSize)
        [IO.File]::WriteAllText($visionSpec, $spec, $utf8)
    }

    $vlm = W "vlm$ImageSize"
    if (-not (Test-Path (Join-Path $vlm "config.json"))) { Idrak create $visionSpec -o $vlm --decoder $text -P $plugin }

    Step "4. Fonts"
    if ($Fonts -eq "") {
        $Fonts = W "fonts"
        New-Item -ItemType Directory -Force $Fonts | Out-Null
        if ($env:windir -and -not (Get-ChildItem $Fonts -File -ErrorAction SilentlyContinue)) {
            foreach ($name in "arial.ttf", "times.ttf") {
                $font = Join-Path (Join-Path $env:windir "Fonts") $name
                if (Test-Path $font) { Copy-Item $font $Fonts }
            }
        }
    }

    if (-not (Get-ChildItem $Fonts -File -Include *.ttf, *.otf, *.ttc -Recurse -ErrorAction SilentlyContinue)) {
        throw "No fonts in ${Fonts}: give -Fonts FOLDER (fonts with Arabic letters, .ttf or .otf)."
    }
    Write-Host "fonts: $Fonts"

    Step "5. Reading data: clean pages cut to their ink"
    $all = @(TextLines (W "corpus.txt"))
    if ($all.Count -lt $Held + 10) { throw "The corpus has $($all.Count) lines: too few for $Held held-out ones." }
    $trainLines = $all[0..([Math]::Min($Lines, $all.Count - $Held) - 1)]
    $heldLines = $all[($all.Count - $Held)..($all.Count - 1)]
    $rng = New-Object System.Random 1
    function Pick([string[]]$lines, [int]$count, [int]$least, [int]$most) {
        $pool = @($lines -split '\s+' | Where-Object { $_.Length -gt 0 })
        for ($i = 0; $i -lt $count; $i++) {
            $n = $rng.Next($least, $most + 1)
            (1..$n | ForEach-Object { $pool[$rng.Next($pool.Count)] }) -join ' '
        }
    }

    $sets = [ordered]@{
        "words1" = { Pick $trainLines $Words 1 1 }; "words3" = { Pick $trainLines $Words 2 3 }; "lines" = { $trainLines };
        "held-words1" = { Pick $heldLines $Held 1 1 }; "held-words3" = { Pick $heldLines $Held 2 3 }; "held-lines" = { $heldLines }
    }
    foreach ($name in $sets.Keys) {
        $data = W "s-$name/pages.jsonl"
        if (Test-Path $data) { Write-Host "$data is there"; continue }
        WriteDocs (W "$name.txt") @(& $sets[$name])
        Remove-Item -Recurse -Force (W "s-$name") -ErrorAction SilentlyContinue
        $seed = if ($name.StartsWith("held")) { "1" } else { "0" }
        Idrak data synth (W "$name.txt") -o $data --fonts $Fonts -n 1 --scan none --grey 0 --layout plain --page $Page --zip --seed $seed
    }

    Step "6. Three stages: single words, two or three words, full lines"
    $stages = @("words1", "words3", "lines")
    $StageSteps = @($Steps1, $Steps2, $Steps3)
    $StageRates = @($Rate1, $Rate2, $Rate3)
    $from = $vlm
    $results = @()
    for ($s = 0; $s -lt 3; $s++) {
        $name = $stages[$s]
        $out = W "c$($s + 1)"
        Write-Host "stage $($s + 1): $name, $($StageSteps[$s]) steps at $($StageRates[$s])"
        Train $from $out $StageSteps[$s] @("--data", (W "s-$name/pages.jsonl"), "--images", (W "s-$name/pages.zip"), "--trained", "all",
            "--lr", $StageRates[$s], "-P", $plugin)
        $from = Last $out
        Idrak runs show $out
        $heldSet = W "s-held-$name"
        $line = (& dotnet $idrak tune evaluate $from (Join-Path $heldSet "pages.jsonl") --images (Join-Path $heldSet "pages.zip") --metric cer -d $Device -P $plugin) |
            ForEach-Object { Write-Host $_; $_ } | Where-Object { $_ -match "cer " } | Select-Object -Last 1
        if ($LASTEXITCODE -ne 0) { throw "evaluating $from failed" }
        $results += "stage $($s + 1) ($name): $line"
    }

    Step "Results (held-out images)"
    $results | ForEach-Object { Write-Host $_ }
    Write-Host "log: $(W 'b6.log')"
}
finally {
    Stop-Transcript | Out-Null
}
