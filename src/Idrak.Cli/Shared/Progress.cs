// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Diagnostics;
using Idrak.Training;
using Idrak.Training.Abstractions;

namespace Idrak.Cli.Shared;

/// <summary>
/// A progress line for downloads, loading, training and long benchmarks (plans/done/idrak-cli.md, "Helpers every command
/// shares"): the label, the share done, the amount, the rate and the time left, shown by the library's progress
/// reporters (<see cref="ProgressReporters"/>) on the error output (so the standard output stays clean for pipes). The
/// reporter: none with <c>--quiet</c>; the one <c>IDRAK_PROGRESS</c> names when set ("console", "plain", "json",
/// "quiet" or a plug-in's); none with <c>--json</c> or <c>--plain</c> or when the error output is not a terminal; else
/// "console", the line redrawn in place.
/// </summary>
/// <example><code>
/// using var progress = new ProgressLine(context, "pull model.gguf", total: length, unit: ProgressUnit.Bytes);
/// progress.Report(done);   // as often as convenient: redrawn at most 10 times a second
/// </code></example>
internal sealed class ProgressLine : IProgressReporter
{
    private readonly IProgressReporter _reporter;

    /// <param name="context">The command (its <c>--quiet</c>, <c>--json</c> and error output).</param>
    /// <param name="label">What is in progress ("pull Qwen/Qwen3-0.6B").</param>
    /// <param name="total">The amount when known; null shows the amount and rate without a share or time left.</param>
    /// <param name="unit">How amounts are shown.</param>
    /// <param name="enabled">Tests: draw in place (true) or not (false) whatever the terminal; null decides as described above.</param>
    /// <exception cref="UsageException"><c>IDRAK_PROGRESS</c> names no registered reporter.</exception>
    public ProgressLine(CommandContext context, string label, long? total = null, ProgressUnit unit = ProgressUnit.Items, bool? enabled = null)
    {
        var writer = context.ErrorOutput;
        string reporter = enabled is { } on ? on ? ProgressReporters.ConsoleReporter : ProgressReporters.QuietReporter
            : context.Quiet ? ProgressReporters.QuietReporter
            : Chosen() ?? (!context.Json && !context.Plain && Terminal.IsTerminal(writer) ? ProgressReporters.ConsoleReporter : ProgressReporters.QuietReporter);
        _reporter = ProgressReporters.Get(reporter)(new ProgressRequest(label, total, unit, writer, null));
    }

    /// <summary>The reporter <c>IDRAK_PROGRESS</c> names, or null when it is not set.</summary>
    /// <exception cref="UsageException">It names no registered reporter.</exception>
    public static string? Chosen()
    {
        try
        {
            return ProgressReporters.FromEnvironment();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }
    }

    public string Label
    {
        get => _reporter.Label;
        set => _reporter.Label = value;
    }

    public long? Total
    {
        get => _reporter.Total;
        set => _reporter.Total = value;
    }

    /// <summary>What the counts are (a stage may change it: bytes read, then items made).</summary>
    public ProgressUnit Unit
    {
        get => _reporter.Unit;
        set => _reporter.Unit = value;
    }

    /// <summary>Whether the line is shown.</summary>
    public bool Enabled => _reporter.Enabled;

    /// <summary>The amount done so far.</summary>
    public long Done => _reporter.Done;

    /// <inheritdoc />
    public Func<string>? Detail
    {
        get => _reporter.Detail;
        set => _reporter.Detail = value;
    }

    /// <summary>Sets the amount done and redraws when the last drawing is more than a tenth of a second old.</summary>
    public void Report(long value) => _reporter.Report(value);

    /// <summary>Adds <paramref name="amount"/> to the amount done.</summary>
    public void Advance(long amount = 1) => _reporter.Advance(amount);

    /// <inheritdoc />
    public void Show() => _reporter.Show();

    /// <summary>
    /// Ends the current stage (its line stays as it stands, or <paramref name="message"/> in its place) and starts the next
    /// on a new line, with its own label, amount, rate and time left.
    /// </summary>
    public void NextStage(string label, long? total, ProgressUnit unit, string? message = null, bool keepLine = true) =>
        _reporter.NextStage(label, total, unit, message, keepLine);

    /// <summary>Draws the final state and ends the line (with <paramref name="message"/> in place of the progress, when given).</summary>
    public void Finish(string? message = null) => _reporter.Finish(message);

    /// <summary>Removes the line (nothing stays on the screen), for work whose result is reported elsewhere.</summary>
    public void Clear() => _reporter.Clear();

    /// <summary>
    /// Takes the line off the screen for a moment, so a line of output can be written; the next report draws it again
    /// below that output.
    /// </summary>
    public void Erase() => _reporter.Erase();

    public void Dispose() => _reporter.Dispose();

    /// <summary>
    /// Takes every progress line drawn on <paramref name="writer"/> off the screen for a moment (<see cref="Erase"/>), so
    /// a line another part of the command writes (a suggestion the library published) does not land inside one.
    /// </summary>
    public static void EraseOn(TextWriter writer) => ProgressReporters.EraseOn(writer);

    /// <summary>The line's text (<see cref="ProgressText.Format"/>).</summary>
    public static string Format(string label, long done, long? total, TimeSpan elapsed, ProgressUnit unit) => ProgressText.Format(label, done, total, elapsed, unit);

    /// <summary>A rate as the unit shows it (<see cref="ProgressText.Rate"/>).</summary>
    public static string Rate(double perSecond, ProgressUnit unit) => ProgressText.Rate(perSecond, unit);

    /// <summary>An amount as the unit shows it (<see cref="ProgressText.Amount"/>).</summary>
    public static string Amount(long value, ProgressUnit unit) => ProgressText.Amount(value, unit);
}

/// <summary>
/// A training run's progress line, as a trainer callback: "epoch 3/40", the batches done of the run, the rate and the
/// time left. Lines the command writes meanwhile call <see cref="ProgressLine.Erase"/> first.
/// </summary>
/// <param name="line">The line (enabled on a terminal only).</param>
/// <param name="epochs">Epochs in this run.</param>
/// <param name="before">Epochs done before it (a resumed run), for the numbers shown.</param>
internal sealed class TrainingProgress(ProgressLine line, int epochs, int before = 0) : ITrainerCallback
{
    private int _labelled = int.MinValue;                                        // the epoch the label was made for

    public void OnBatchEnd(TrainerContext context, BatchCompleted batch)
    {
        line.Total = (long)epochs * batch.BatchesPerEpoch;
        if (batch.Epoch != _labelled)
        {
            line.Label = $"epoch {batch.Epoch + before}/{epochs + before}";   // once per epoch, not a string per step
            _labelled = batch.Epoch;
        }

        line.Report((long)(batch.Epoch - 1) * batch.BatchesPerEpoch + batch.Batch);
    }

    public void OnTrainEnd(TrainerContext context, TrainingHistory history) => line.Clear();
}
