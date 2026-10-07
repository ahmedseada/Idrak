// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Modules;

/// <summary>
/// What a network step (<see cref="NetworkOp"/>) sees of the network builder (<c>Idrak.Layers.NetworkBuilder</c>,
/// started with <c>Network.Input</c>, <c>Network.Image</c>, ...): the shape of one sample so far, and the ways to add
/// layers to it. The builder supplies its device and random to the layers a step creates, and writes the step to JSON as
/// <c>{"op": name, ...arguments}</c> however the step made its layers, so <c>Network.FromJson</c> replays it through
/// the same registration in <see cref="NetworkOps"/>.
/// </summary>
/// <example>
/// <code>
/// NetworkOps.Register("scale", (b, a) =>
/// {
///     float factor = a.Float("factor");
///     return b.Lambda(x => x * factor, "Scale", [.. b.CurrentShape]);
/// });
/// NetworkOps.Register("dense", (b, a) =>
/// {
///     int inputs = b.CurrentShape[^1], outputs = a.Int("out");
///     return b.Add((device, random) => new Linear(inputs, outputs, true, device, random), [.. b.CurrentShape.SkipLast(1), outputs]);
/// });
/// NetworkOps.Register("block", (b, a) => b.Op("linear", new JsonObject { ["out"] = a.Int("width"), ["bias"] = true }).Op("relu"));
/// </code>
/// </example>
public interface INetworkBuilder
{
    /// <summary>The shape of one sample after the steps added so far (without the batch dimension).</summary>
    IReadOnlyList<int> CurrentShape { get; }

    /// <summary>
    /// Appends a layer applying <paramref name="function"/> (a lambda layer named <paramref name="name"/>). The builder
    /// cannot know what the function does to the shape, so the shape of one output sample is required.
    /// </summary>
    INetworkBuilder Lambda(Func<Tensor, Tensor> function, string name, int[] outputShape);

    /// <summary>
    /// Appends a layer that <paramref name="create"/> makes anew every time the network is built, from the builder's
    /// device and random (as the built-in steps make theirs). Its output shape is required.
    /// </summary>
    INetworkBuilder Add(Func<Device?, Random?, Module> create, int[] outputShape);

    /// <summary>
    /// Adds the step registered as <paramref name="name"/> in <see cref="NetworkOps"/> (a built-in layer such as
    /// "linear" or "relu", or a step of your own), with <paramref name="arguments"/> (the keys of its JSON besides "op").
    /// </summary>
    INetworkBuilder Op(string name, JsonObject? arguments = null);
}
