// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections;

namespace Idrak.Layers;

/// <summary>
/// Runs modules one after another, feeding each output into the next.
/// Supports collection-initializer syntax:
/// <code>
/// var model = new Sequential
/// {
///     new Linear(2, 8),
///     new Tanh(),
///     new Linear(8, 1),
///     new Sigmoid(),
/// };
/// </code>
/// </summary>
public sealed class Sequential : Module, IEnumerable<Module>, ICachedModule
{
    private readonly List<Module> _modules = [];

    /// <summary>Creates an empty container; add layers with <see cref="Add"/> or a collection initializer.</summary>
    public Sequential()
    {
    }

    /// <summary>Creates a container from the given layers.</summary>
    public Sequential(params IEnumerable<Module> modules) => _modules.AddRange(modules);

    /// <summary>The number of layers.</summary>
    public int Count => _modules.Count;

    /// <summary>The layer at <paramref name="index"/>.</summary>
    public Module this[int index] => _modules[index];

    /// <summary>Appends a layer.</summary>
    public void Add(Module module) => _modules.Add(module ?? throw new ArgumentNullException(nameof(module)));

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Run(input, _modules.Count);

    /// <summary>
    /// Runs only the first <paramref name="layers"/> layers (for example everything but the output layer, to get the
    /// hidden states), as <see cref="Module.Forward"/> runs them all.
    /// </summary>
    public Tensor ForwardFirst(Tensor input, int layers)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentOutOfRangeException.ThrowIfNegative(layers);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(layers, _modules.Count);
        return Run(input, layers);
    }

    private Tensor Run(Tensor input, int layers)
    {
        Tensor Layer(int i, Tensor x) => _modules[i].Forward(x);
        return Autograd.IsEnabled || ComputeGraph.IsCapturing ? RunLayers(input, layers, Layer) : RunFreeing(input, layers, Layer);
    }

    private static Tensor RunLayers(Tensor input, int layers, Func<int, Tensor, Tensor> layer)
    {
        var x = input;
        for (int i = 0; i < layers; i++)
        {
            x = layer(i, x);
        }

        return x;
    }

    private static Tensor RunFreeing(Tensor input, int layers, Func<int, Tensor, Tensor> layer)
    {
        var x = input;

        // Without autograd, nothing needs a layer's intermediate results once the next layer has its input: each layer
        // runs in a scope of its own that frees them, and the previous layer's output goes as soon as it is used. A pass
        // then holds one layer's worth of memory, not the whole network's until the caller's scope ends. (Not while a
        // graph is being recorded: its work keeps using those buffers.)
        // A normalization of a layer's output computed ahead with it (see RMSNorm.HandOff) is kept for the next layer
        // too, and freed once that layer has run (or passed on when the layer is the norm itself and returns it); freed
        // with the layer's scope, the norm would compute it again.
        bool ownsX = false;
        Tensor? handed = null;
        for (int i = 0; i < layers; i++)
        {
            Tensor next;
            Tensor? nextHanded;
            bool created;
            using (var scope = new TensorScope())
            {
                next = layer(i, x);
                created = scope.Owns(next);
                scope.Keep(next);
                nextHanded = RMSNorm.HandedOff(next);
                nextHanded = nextHanded is not null && !ReferenceEquals(nextHanded, next) && scope.Owns(nextHanded) ? scope.Keep(nextHanded) : null;
            }

            if (handed is not null)
            {
                if (ReferenceEquals(handed, next))
                {
                    created = true;                                     // the norm returned it: owned as the output now
                }
                else
                {
                    handed.Dispose();
                }
            }

            handed = nextHanded;
            if (!ReferenceEquals(next, x))
            {
                if (ownsX)
                {
                    x.Dispose();
                }

                ownsX = created;
                x = next;
            }
        }

        handed?.Dispose();
        return x;
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => _modules;

    /// <summary>
    /// Incremental forward pass: layers implementing <see cref="ICachedModule"/> use their cached path, others run
    /// normally on the new positions only. Brackets the pass with <see cref="DecodingContext.BeginStep"/> /
    /// <see cref="DecodingContext.EndStep"/> when called at the top level.
    /// </summary>
    public Tensor ForwardCached(Tensor input, DecodingContext context) => ForwardCached(input, context, _modules.Count);

    /// <summary>
    /// <see cref="ForwardCached(Tensor, DecodingContext)"/> through the first <paramref name="layers"/> modules only (for
    /// example all but the output head, to read hidden states rather than logits).
    /// </summary>
    public Tensor ForwardCached(Tensor input, DecodingContext context, int layers)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(layers);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(layers, _modules.Count);
        bool outermost = !context.InStep;
        int steps = input.Shape[1];
        if (outermost)
        {
            context.BeginStep(steps);
        }

        // With LastPositionOnly, the layers after the last cached one see only the last position.
        int lastCached = outermost && context.LastPositionOnly && steps > 1 ? _modules.FindLastIndex(m => m is ICachedModule) : -1;
        Tensor Layer(int i, Tensor x)
        {
            var module = _modules[i];
            x = module is ICachedModule cached ? cached.ForwardCached(x, context) : module.Forward(x);
            return i == lastCached && x.Rank == 3 ? x.Narrow(1, x.Shape[1] - 1, 1) : x;
        }

        var x = Autograd.IsEnabled || ComputeGraph.IsCapturing ? RunLayers(input, layers, Layer) : RunFreeing(input, layers, Layer);

        if (outermost)
        {
            context.EndStep(steps);
        }

        return x;
    }

    /// <inheritdoc />
    public IEnumerator<Module> GetEnumerator() => _modules.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    public override string ToString() => $"Sequential({_modules.Count} layers)";
}
