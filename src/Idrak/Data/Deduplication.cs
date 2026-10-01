// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data;

/// <summary>What makes two rows the same, for <see cref="Dataset.Deduplicate"/>.</summary>
public enum DuplicateMatch
{
    /// <summary>Rows with the same features are duplicates, whatever their targets (see <see cref="DuplicateConflicts"/>).</summary>
    Features,

    /// <summary>Rows are duplicates only when both their features and their targets are the same.</summary>
    FeaturesAndTargets,
}

/// <summary>What to keep when rows with the same features have different targets (with <see cref="DuplicateMatch.Features"/>).</summary>
public enum DuplicateConflicts
{
    /// <summary>The first row, with its targets.</summary>
    KeepFirst,

    /// <summary>The first row of the most frequent targets (the earliest of them on a tie): a majority vote over noisy labels.</summary>
    KeepMostFrequent,

    /// <summary>None of them: the features do not decide the targets, so the rows are dropped.</summary>
    DropAll,
}

/// <summary>Settings of <see cref="Dataset.Deduplicate"/>. The defaults remove exact duplicate rows only.</summary>
public sealed record DeduplicationOptions
{
    /// <summary>What makes two rows the same. <see cref="DuplicateMatch.Features"/> by default.</summary>
    public DuplicateMatch Match { get; init; } = DuplicateMatch.Features;

    /// <summary>With <see cref="DuplicateMatch.Features"/>: which of the rows to keep when their targets differ.</summary>
    public DuplicateConflicts Conflicts { get; init; } = DuplicateConflicts.KeepFirst;

    /// <summary>
    /// Near duplicates: the embedding of a row (its index in the dataset), computed by the caller (for example a
    /// sentence encoder over the texts the rows were made from). Rows whose embeddings have a cosine similarity of at
    /// least <see cref="SimilarityThreshold"/> with a row kept before them are dropped. Null (the default): exact
    /// duplicates only. The comparison is against every kept row, so its cost grows with rows × kept rows.
    /// </summary>
    public Func<int, float[]>? Embedding { get; init; }

    /// <summary>Cosine similarity at or above which two rows are near duplicates (with <see cref="Embedding"/>). 0.95 by default.</summary>
    public float SimilarityThreshold { get; init; } = 0.95f;
}

/// <summary>
/// What <see cref="Dataset.DeduplicateWithReport"/> did: the deduplicated rows, the indices of the kept rows in the
/// original dataset (to filter data kept alongside it, such as the texts), and how many rows went for each reason.
/// </summary>
public sealed record DeduplicationResult(Dataset Data, int[] Kept, int ExactDuplicates, int Conflicting, int NearDuplicates)
{
    /// <summary>Rows removed in all.</summary>
    public int Removed => ExactDuplicates + Conflicting + NearDuplicates;
}
