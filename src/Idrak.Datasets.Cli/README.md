# idrak-data

Inspect, download and assemble datasets from the command line. The output is JSON Lines (one JSON object per line):
conversations as `{"messages": [...], "tools": [...]}` and plain text as `{"text": ...}`, the layout Idrak's
fine-tuning, Hugging Face's `datasets` and most training tools read.

## Install

From nuget.org (needs the .NET SDK): `dotnet tool install --global Idrak.Datasets.Cli`, then `idrak-data --help`.

From this repository, as a .NET tool:

```
dotnet pack src/Idrak.Datasets.Cli -c Release -o artifacts
dotnet tool install --global --prerelease --add-source artifacts Idrak.Datasets.Cli
idrak-data --help
```

Update after a `git pull`: pack again, then `dotnet tool update --global --prerelease --add-source artifacts Idrak.Datasets.Cli`
(each pack is a new prerelease, 0.1.0-dev.yyMMddHHmm, so the update always installs it).

As one executable that runs without .NET (Native AOT; `win-x64`, `linux-x64` or `osx-arm64`):

```
dotnet publish src/Idrak.Datasets.Cli -c Release -r win-x64 -p:PublishAot=true -p:PackAsTool=false -o artifacts/idrak-data
```

Or without installing: `dotnet run --project src/Idrak.Datasets.Cli -- <command>`.

## Examples

```
idrak-data show "hf:openai/gsm8k?config=main"
idrak-data count "hf:openai/gsm8k?config=main" "hf:openai/gsm8k?config=main&split=test"
idrak-data build "hf:openai/gsm8k?config=main&user={question}&assistant={answer}" --out gsm8k.jsonl --eval-fraction 0.02
idrak-data show "github:owner/repo?files=src/**/*.cs"
idrak-data build recipe.json --out train.jsonl
```

A recipe mixes sources:

```json
{
  "sources": [
    {"source": "hf:openai/gsm8k", "config": "main", "user": "{question}", "assistant": "{answer}", "weight": 1, "take": 2000},
    {"source": "hf:yahma/alpaca-cleaned", "weight": 2, "take": 4000},
    "my-examples.jsonl?weight=0.5"
  ],
  "system": "You are a helpful assistant.",
  "seed": 1, "min_chars": 20, "eval_fraction": 0.02
}
```

Credentials come from the environment: `HF_TOKEN` (or `huggingface-cli login`), `GITHUB_TOKEN`, `KAGGLE_USERNAME` and
`KAGGLE_KEY` (or `~/.kaggle/kaggle.json`), `ZENODO_TOKEN`. Downloads are cached under `IDRAK_CACHE` or
`~/.cache/idrak`, in `downloads/` laid out by source:

```
downloads/huggingface/datasets/<owner>/<name>/<commit>/<path in the repository>
downloads/huggingface/datasets/<owner>/<name>/parquet/<config>/<split>/00000.parquet   (the Hub's Parquet copy)
downloads/github/<owner>/<repo>/<commit>.tar.gz                                     (repository snapshots)
downloads/github/<owner>/<repo>/<commit>/<path>                                     (single files)
downloads/github/<owner>/<repo>/releases/<tag>/<asset>
downloads/kaggle/<owner>/<dataset>/<latest | vN>/<dataset>.zip
downloads/zenodo/<record>/<file>
downloads/urls/<host>/<path>
```

Branches and tags are resolved to their commit first, so new commits are downloaded again rather than read stale.
`idrak-data cache` shows the size per source; `idrak-data cache --clear` empties it. `idrak-data --help` lists
every option.

The same features are available from code in the `Idrak.Datasets` library (`Dataset`, `HuggingFace`, `GitHub`,
`Kaggle`, `Zenodo`, `ChatRows`, `DatasetSpec`, `DatasetRecipe`).
