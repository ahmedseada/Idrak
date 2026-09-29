namespace Idrak.Data;

/// <summary>
/// A train/test split together with the scalers fitted on its training part. Produced by the extension methods in
/// <see cref="DataExtensions"/>; <see cref="Train"/> and <see cref="Test"/> are ordinary <see cref="Dataset"/>s.
/// </summary>
/// <param name="Train">The training part (scaled by the scalers set so far).</param>
/// <param name="Test">The test part (scaled by the same scalers).</param>
/// <param name="FeatureScaler">The feature scaler fitted on the training part, or null.</param>
/// <param name="TargetScaler">The target scaler fitted on the training part, or null.</param>
public sealed record DataSplit(Dataset Train, Dataset Test, IScaler? FeatureScaler = null, IScaler? TargetScaler = null);

/// <summary>
/// Shorter ways to write the usual data steps. Each method is exactly the calls you would otherwise write by hand,
/// with the same parameters and defaults.
/// </summary>
public static class DataExtensions
{
    /// <summary>
    /// <c>StandardScaler.FitFeatures(train)</c>, then <c>Scale(features: scaler)</c> on both parts.
    /// </summary>
    public static DataSplit StandardizeFeatures(this (Dataset Train, Dataset Test) split) =>
        new DataSplit(split.Train, split.Test).StandardizeFeatures();

    /// <summary>The same as the tuple overload, for a split that already has scalers of the other kind.</summary>
    public static DataSplit StandardizeFeatures(this DataSplit split) =>
        WithFeatures(split, StandardScaler.FitFeatures(split.Train));

    /// <summary><c>StandardScaler.FitTargets(train)</c>, then <c>Scale(targets: scaler)</c> on both parts.</summary>
    public static DataSplit StandardizeTargets(this (Dataset Train, Dataset Test) split) =>
        new DataSplit(split.Train, split.Test).StandardizeTargets();

    /// <summary>The same as the tuple overload, for a split that already has scalers of the other kind.</summary>
    public static DataSplit StandardizeTargets(this DataSplit split) =>
        WithTargets(split, StandardScaler.FitTargets(split.Train));

    /// <summary><c>MinMaxScaler.Fit(train.Features, train.FeatureCount)</c>, then <c>Scale(features: scaler)</c> on both parts.</summary>
    public static DataSplit NormalizeFeatures(this (Dataset Train, Dataset Test) split) =>
        new DataSplit(split.Train, split.Test).NormalizeFeatures();

    /// <summary>The same as the tuple overload, for a split that already has scalers of the other kind.</summary>
    public static DataSplit NormalizeFeatures(this DataSplit split) =>
        WithFeatures(split, MinMaxScaler.Fit(split.Train.Features, split.Train.FeatureCount));

    /// <summary><c>MinMaxScaler.Fit(train.Targets, train.TargetCount)</c>, then <c>Scale(targets: scaler)</c> on both parts.</summary>
    public static DataSplit NormalizeTargets(this (Dataset Train, Dataset Test) split) =>
        new DataSplit(split.Train, split.Test).NormalizeTargets();

    /// <summary>The same as the tuple overload, for a split that already has scalers of the other kind.</summary>
    public static DataSplit NormalizeTargets(this DataSplit split) =>
        WithTargets(split, MinMaxScaler.Fit(split.Train.Targets, split.Train.TargetCount));

    /// <summary><c>new DataLoader(dataset, batchSize, shuffle, dropLast, device, seed)</c>, with the constructor's defaults.</summary>
    public static DataLoader Batches(this Dataset dataset, int batchSize = 32, bool shuffle = false, bool dropLast = false,
        Device? device = null, int? seed = null) =>
        new(dataset, batchSize, shuffle, dropLast, device, seed);

    private static DataSplit WithFeatures(DataSplit split, IScaler scaler)
    {
        if (split.FeatureScaler is not null)
        {
            throw new InvalidOperationException("The features of this split are already scaled.");
        }

        return split with { Train = split.Train.Scale(features: scaler), Test = split.Test.Scale(features: scaler), FeatureScaler = scaler };
    }

    private static DataSplit WithTargets(DataSplit split, IScaler scaler)
    {
        if (split.TargetScaler is not null)
        {
            throw new InvalidOperationException("The targets of this split are already scaled.");
        }

        return split with { Train = split.Train.Scale(targets: scaler), Test = split.Test.Scale(targets: scaler), TargetScaler = scaler };
    }
}
