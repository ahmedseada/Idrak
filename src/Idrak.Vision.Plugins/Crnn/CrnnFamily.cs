// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Imaging;
using Idrak.Vision.Abstractions;

namespace Idrak.Vision.Plugins.Crnn;

/// <summary>
/// The CRNN line recognizer as an image model family (<see cref="IImageModelFamily"/>, task
/// <see cref="ImageTask.LineRecognition"/>), registered in <see cref="ImageModelFamilies"/> under
/// <see cref="CrnnPlugin.Architecture"/> by <see cref="CrnnPlugin"/>: a model folder's config.json holds the settings
/// (<see cref="CrnnSettings"/>), its safetensors the network's tensors by layer (<see cref="Tensors"/>); the network is
/// made from the settings (<see cref="CrnnRecognizer.Describe"/>) and each tensor read into it, its shape checked. The
/// model reads lines through the library (<c>model.LineReader()</c>): one channel, the alphabet as labels after the blank,
/// the settings' line preparation and reading order.
/// </summary>
public sealed class CrnnFamily : IImageModelFamily
{
    /// <inheritdoc />
    public string Name => CrnnPlugin.Architecture;

    /// <summary>
    /// The network's tensors by name, in the order <see cref="Module.Parameters"/> and <see cref="Module.Buffers"/> give
    /// them for each layer: <c>layers.{layer}.parameters.{index}</c> and <c>layers.{layer}.buffers.{index}</c> (a layer of
    /// the network's sequence, a tensor of that layer).
    /// </summary>
    public static IEnumerable<(string Name, Tensor Tensor)> Tensors(Module network)
    {
        ArgumentNullException.ThrowIfNull(network);
        int layer = 0;
        foreach (var child in network.Children())
        {
            int index = 0;
            foreach (var parameter in child.Parameters())
            {
                yield return ($"layers.{layer}.parameters.{index++}", parameter);
            }

            index = 0;
            foreach (var buffer in child.Buffers())
            {
                yield return ($"layers.{layer}.buffers.{index++}", buffer);
            }

            layer++;
        }
    }

    /// <inheritdoc />
    public ImageModel Read(ImageCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var store = checkpoint.RequireTensors();
        var settings = CrnnSettings.FromJson(checkpoint.Config);
        var network = CrnnRecognizer.Describe(settings, checkpoint.Device, seed: 0).Build();
        try
        {
            var expected = Tensors(network).ToList();
            foreach (var (name, tensor) in expected)
            {
                if (!store.Contains(name))
                {
                    throw new InvalidDataException($"{checkpoint.Path}: the checkpoint has no {name}, which the settings' network holds.");
                }

                var shape = store.ShapeOf(name);
                if (!shape.AsSpan().SequenceEqual(tensor.Shape))
                {
                    throw new InvalidDataException($"{checkpoint.Path}: {name} is {Tensor.FormatShape(shape)}; the settings' network holds {Tensor.FormatShape(tensor.Shape)}.");
                }

                tensor.Load(store.Read(name));
            }

            int stored = store.Names.Count();
            if (stored != expected.Count)
            {
                checkpoint.Notes.Add($"The checkpoint holds {stored} tensors; the network read {expected.Count} of them.");
            }

            return new ImageModel
            {
                Architecture = checkpoint.Architecture,
                Task = ImageTask.LineRecognition,
                Network = network,
                // Lines are prepared by Lines (ModelLineReader), not by an image preprocessor; this one says the size only.
                Preprocessor = new ImagePreprocessor { Height = settings.Height, Width = settings.Height, Normalize = false },
                Channels = 1,
                Labels = settings.Labels,
                Lines = settings.Lines,
            };
        }
        catch
        {
            network.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}
