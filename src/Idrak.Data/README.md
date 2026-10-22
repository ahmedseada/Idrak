# Idrak.Data

Datasets for Idrak (no dependencies beyond Idrak): read JSON Lines, JSON, CSV, text and Parquet files, archives and compressed files; download and cache from URLs, Hugging Face, GitHub, Kaggle and Zenodo; filter, map, shuffle, deduplicate, split, mix and turn rows into chat transcripts for fine-tuning, or columns into training samples (`TableSamples`).

## Contracts

The plug-in points only this package uses are in `Idrak.Data.Abstractions` (add the `using` line to name them):
`DataFileFormats` and `IDataFileFormat` (file formats), `DatasetSources` and `IDatasetSource` (with `DatasetSpec`,
`ReadOptions`, `IDatasetRows`), `ParquetCodecs` and `IParquetCodec`, `ITextNormalizer`. The registries register the
built-ins themselves, as the library defaults of their slots: an app's registration under a built-in name overrides it
and `Unregister` brings it back. A Parquet codec or a dataset source an app overrides throws to the caller when it fails,
reported with a hint (`SetPolicy` opts it into `FallBack` to the library's, or `Shadow`); a file format is used as registered (its rows are read lazily). Shared
contracts (sample sources, `IDownloader`, model sources) are in Idrak.Abstraction.

## Folders skipped

Reading a folder or an archive skips the folders `ReadOptions.SkippedFolders` names: by default
(`ReadOptions.DefaultSkippedFolders`) only `.git` and the editors' state (`.vs`, `.idea`). No language's build output or
dependency folders are skipped unless named (`bin`, `obj`, `node_modules`, `dist`, `__pycache__`, …): a caller that knows
them adds them. The `idrak` tool's data commands add every registered coding language's (Idrak.Text's `CodingLanguages`;
the tool registers Idrak.Text.Plugins' at startup), so they read code folders as before. A recipe takes them as
`"skip_folders": [...]`, `DatasetSpec.Open` and `GitHub.Repository` as an argument. The table from file extensions to
language names that labels the `code` format's rows (`DataFiles.LanguageOf`) is data only.

## Install

```bash
dotnet add package Idrak.Data
```

This is a library (for your code). The command-line tool `idrak` inspects, downloads and builds datasets with no code needed (`idrak data`); it is a separate package:

```bash
dotnet tool install -g Idrak.Cli
idrak help data
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
