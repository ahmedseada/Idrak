// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Vision.Abstractions;

/// <summary>
/// A detection network's outputs read as candidate boxes for training (<see cref="DetectionHead"/>): differentiable tensors
/// of the network's outputs, so a loss on them trains the network. Dispose it after the step (a training step's tensor
/// scope does that too).
/// </summary>
/// <param name="Boxes">Each candidate's box, [N, P, 4] corners (x1, y1, x2, y2) in the network input's pixels.</param>
/// <param name="ClassLogits">Each candidate's class scores as logits (a sigmoid gives the probabilities), [N, P, classes].</param>
/// <param name="Objectness">Each candidate's objectness logit, [N, P], or null when the head has none (the class scores alone say whether a candidate is an object).</param>
public sealed record DetectionCandidates(Tensor Boxes, Tensor ClassLogits, Tensor? Objectness) : IDisposable
{
    /// <summary>The images.</summary>
    public int Images => Boxes.Shape[0];

    /// <summary>The candidates per image.</summary>
    public int Candidates => Boxes.Shape[1];

    /// <summary>The classes.</summary>
    public int Classes => ClassLogits.Shape[^1];

    /// <inheritdoc />
    public void Dispose()
    {
        Boxes.Dispose();
        ClassLogits.Dispose();
        Objectness?.Dispose();
    }
}

/// <summary>
/// The training side of a detection decoder: reads the network's outputs for a batch, [N, ...], as the
/// <see cref="DetectionCandidates"/> the decoder would turn into detections, but as tensors that carry gradients.
/// </summary>
public delegate DetectionCandidates DetectionHead(Tensor outputs);

/// <summary>Makes the <see cref="DetectionHead"/> for one detector (its input size, classes and settings, as its decoder gets them).</summary>
public delegate DetectionHead DetectionHeadFactory(DetectionDecoderContext context);

/// <summary>
/// The detection heads, by the name of the decoder whose outputs they read (<see cref="DetectionDecoders"/>): how a
/// detector is trained. A head is part of a model family as its decoder is (its grid or anchors, its box encoding), so it
/// is registered by the plug-in or app that brings the family, under the decoder's name; the library registers none. Idrak.Vision's
/// <c>DetectionObjective</c> matches the candidates to the true boxes and scores them with the registered losses. An
/// unregistered name is refused, naming this registry.
/// </summary>
public static class DetectionHeads
{
    private static readonly SlotTable<string, DetectionHeadFactory> Registry = new(nameof(DetectionHeads), comparer: StringComparer.Ordinal,
        unguarded: "a head is the training side of its decoder: the two must read the network's outputs the same way, so two registrations cannot be mixed");

    /// <summary>Registers the head of the decoder <paramref name="name"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, DetectionHeadFactory factory)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(factory);
        Registry.Register(name, factory, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's head <paramref name="name"/>; false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered head names (the decoders that can be trained).</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The head registered as <paramref name="name"/>, or null.</summary>
    public static DetectionHeadFactory? Find(string name) => Registry.Find(name);

    /// <summary>The head registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names <see cref="Register"/>.</exception>
    public static DetectionHeadFactory Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"Detection head '{name}' is not registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "register it with DetectionHeads.Register under its decoder's name (the plug-in or app that brings the model family; the library registers none).");

    /// <summary>The library's head <paramref name="name"/>, whatever an app registered over it; null when the library has none (it has none).</summary>
    public static DetectionHeadFactory? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the head <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>The head <paramref name="name"/> made for <paramref name="context"/>.</summary>
    public static DetectionHead Create(string name, DetectionDecoderContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        return Get(name)(context) ?? throw new InvalidOperationException($"Detection head '{name}' made no head.");
    }
}
