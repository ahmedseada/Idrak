# Idrak.Data

Datasets for Idrak (no dependencies beyond Idrak): read JSON Lines, JSON, CSV, text and Parquet files, archives and compressed files; download and cache from URLs, Hugging Face, GitHub, Kaggle and Zenodo; filter, map, shuffle, deduplicate, split, mix and turn rows into chat transcripts for fine-tuning, or columns into training samples (`TableSamples`).

## Contracts

The plug-in points only this package uses are in `Idrak.Data.Abstractions` (add the `using` line to name them):
`DataFileFormats` and `IDataFileFormat` (file formats), `DatasetSources` and `IDatasetSource` (with `DatasetSpec`,
`ReadOptions`, `IDatasetRows`), `ParquetCodecs` and `IParquetCodec`, `ITextNormalizer`. The registries register the
built-ins themselves. Shared contracts (sample sources, `IDownloader`, model sources) are in Idrak.Abstraction.

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
