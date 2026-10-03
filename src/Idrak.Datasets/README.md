# Idrak.Datasets

Datasets for Idrak (no dependencies beyond Idrak): read JSON Lines, JSON, CSV, text and Parquet files, archives and compressed files; download and cache from URLs, Hugging Face, GitHub, Kaggle and Zenodo; filter, map, shuffle, deduplicate, split, mix and turn rows into chat transcripts for fine-tuning, or columns into training samples (`TableSamples`).

## Install

```bash
dotnet add package Idrak.Datasets
```

This is a library (for your code). The command-line tool `idrak-data` (inspect, download and build datasets, no code needed) is a separate package:

```bash
dotnet tool install -g Idrak.Datasets.Cli
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
