// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Cluster;

namespace Idrak.Optimizers;

/// <summary>
/// Runs another optimizer's step on the CPU, keeping its state (Adam's two moments: twice the parameters' size) in system
/// memory instead of on the GPU. Each step downloads the gradients, updates CPU copies of the parameters with the CPU's
/// SIMD kernels on all cores, and uploads the new values: only gradients and weights cross PCIe, never the state. On a GPU
/// the copies go through pinned buffers in the background, so one parameter downloads while the previous one is updated
/// and uploaded. Use it when the optimizer state does not fit on the GPU (full fine-tuning of a model that just fits).
/// Works with any optimizer that updates each parameter on its own (SGD, Adam, AdamW, AdamW8Bit):
/// <c>new HostOptimizer(model.Parameters(), ps =&gt; new AdamW(ps, 1e-4f))</c>. In data-parallel training (plan 23) the
/// gradient synchronisation reduces the gradients on the host copies this optimizer downloads (<see cref="IHostGradients"/>),
/// and the clipping and the update then read the reduced values there.
/// </summary>
public sealed class HostOptimizer : Optimizer, IHostGradients
{
    // Pinned buffers: four for gradients coming down, two for weights going up, 16 MiB each (a larger tensor moves in
    // pieces), so the pinned memory stays at 96 MiB whatever the model's size.
    private const int SlotFloats = 4 << 20, DownSlots = 4, UpSlots = 2;

    private readonly Tensor[] _copies;                                  // the parameters on the CPU
    private readonly Optimizer[] _cpu;                                  // one per parameter, so each updates as it arrives
    private readonly List<(int Parameter, int Offset, int Count)> _pieces = [];   // a step's downloads, reused every step
    private IHostStaging? _staging;
    private bool _stagingTried;
    private bool _downloaded;                                           // the gradients are on the host copies (DownloadGradients)

    /// <summary>Creates the CPU copies of <paramref name="parameters"/> and the optimizers that update them.</summary>
    /// <param name="parameters">The parameters to train (on any device).</param>
    /// <param name="create">Creates an optimizer from the CPU copies it is given (one parameter each), e.g. <c>ps =&gt; new AdamW(ps, 1e-4f)</c>.</param>
    public HostOptimizer(IEnumerable<Tensor> parameters, Func<IReadOnlyList<Tensor>, Optimizer> create) : base(parameters, 0f)
    {
        ArgumentNullException.ThrowIfNull(create);
        _copies = [.. Parameters.Select(p => Tensor.Persistent(p.ToArray(), p.Shape, Device.Cpu, requiresGrad: true))];
        _cpu = [.. _copies.Select(c => create([c]))];
        LearningRate = _cpu[0].LearningRate;
    }

    /// <summary>The optimizers that run on the CPU, one per parameter (their learning rate follows <see cref="Optimizer.LearningRate"/>).</summary>
    public IReadOnlyList<Optimizer> CpuOptimizers => _cpu;

    /// <inheritdoc />
    public override void Step()
    {
        if (_downloaded)
        {
            UpdateDownloaded(0f, release: false);
            return;
        }

        Update(release: false);
    }

    /// <summary>
    /// Clips the gradients' global norm on their device, then updates on the CPU (see <see cref="Optimizer.ClipAndStep"/>).
    /// The gradients are released on their device as they are downloaded: the next backward pass writes fresh ones
    /// without zero-filling them first.
    /// </summary>
    public override void ClipAndStep(float maxNorm)
    {
        if (_downloaded)
        {
            UpdateDownloaded(maxNorm, release: true);                     // reduced on the host: clipped and updated there
            GradientsZeroed = true;
            return;
        }

        if (maxNorm > 0f)
        {
            ClipGradientNormOnDevice(maxNorm);                             // on the GPU: no host wait, one pass
        }

        Update(release: true);
        GradientsZeroed = true;
    }

    private IHostStaging? Staging()
    {
        if (!_stagingTried)
        {
            _stagingTried = true;
            _staging = Parameters[0].Backend.CreateHostStaging(DownSlots + UpSlots, SlotFloats);   // null where the device has none (the CPU)
        }

        return _staging;
    }

    /// <summary>
    /// Brings every gradient down to the host copies (in pieces through the pinned slots, a few ahead, on a GPU) without
    /// updating: a gradient synchronisation then reduces them there (<see cref="IHostGradients"/>), and the next
    /// <see cref="Step()"/> or <see cref="ClipAndStep"/> clips and updates from the host values.
    /// </summary>
    public void DownloadGradients()
    {
        var staging = Staging();
        var pieces = _pieces;
        pieces.Clear();
        for (int i = 0; i < _copies.Length; i++)
        {
            if (Parameters[i].Grad is not { } g)
            {
                _copies[i].ReleaseGrad();
                continue;
            }

            _copies[i].GradientTarget(out _);
            if (staging is null)
            {
                g.Backend.Download(g.Storage, _copies[i].Grad!.Storage.HostMemory[.._copies[i].Size]);
                continue;
            }

            for (int offset = 0; offset < _copies[i].Size; offset += SlotFloats)
            {
                pieces.Add((i, offset, Math.Min(SlotFloats, _copies[i].Size - offset)));
            }
        }

        if (staging is { } slots)
        {
            int next = 0;
            void Issue(int slot)
            {
                if (next < pieces.Count)
                {
                    var (parameter, offset, count) = pieces[next++];
                    slots.Download(Parameters[parameter].Grad!.Storage, offset, slot, count);
                }
            }

            for (int slot = 0; slot < DownSlots; slot++)
            {
                Issue(slot);
            }

            for (int j = 0; j < pieces.Count; j++)
            {
                int slot = j % DownSlots;                                  // pieces are issued to the slots in turn
                var (i, offset, count) = pieces[j];
                slots.Wait(slot);
                slots.Slot(slot)[..count].CopyTo(_copies[i].Grad!.Storage.HostMemory.Slice(offset, count));
                Issue(slot);
            }
        }

        _downloaded = true;
    }

    /// <inheritdoc />
    public Span<float> HostGradient(int parameter) =>
        _downloaded && _copies[parameter].Grad is { } g ? g.Storage.HostMemory[.._copies[parameter].Size] : [];

    /// <inheritdoc />
    public Span<float> CreateHostGradient(int parameter) => _copies[parameter].GradStorage().HostMemory[.._copies[parameter].Size];

    /// <inheritdoc />
    public void ReloadParameters()
    {
        for (int i = 0; i < _copies.Length; i++)
        {
            Parameters[i].Backend.Download(Parameters[i].Storage, _copies[i].Storage.HostMemory[.._copies[i].Size]);
        }
    }

    // The update from gradients already on the host copies (reduced there): clipped on the CPU, each parameter updated and
    // sent up.
    private void UpdateDownloaded(float maxNorm, bool release)
    {
        _downloaded = false;
        if (maxNorm > 0f)
        {
            ClipCopies(maxNorm);
        }

        var staging = Staging();
        int up = 0;
        for (int i = 0; i < _copies.Length; i++)
        {
            if (release)
            {
                Parameters[i].ReleaseGrad();
            }

            if (_copies[i].Grad is null)
            {
                continue;                                                  // no gradient on any rank: its optimizer skips it
            }

            Step(i);
            var values = _copies[i].Storage.HostMemory;
            if (staging is null)
            {
                Parameters[i].Backend.Upload(values[.._copies[i].Size], Parameters[i].Storage);
                continue;
            }

            for (int start = 0; start < _copies[i].Size; start += SlotFloats)
            {
                int n = Math.Min(SlotFloats, _copies[i].Size - start);
                int target = DownSlots + up++ % UpSlots;
                staging.Wait(target);
                values.Slice(start, n).CopyTo(staging.Slot(target));
                staging.Upload(target, Parameters[i].Storage, start, n);
            }
        }
    }

    // The global norm clipped on the host copies' gradients, as ClipGradientNormOnDevice does on the device.
    private void ClipCopies(float maxNorm)
    {
        var withGrad = _copies.Where(c => c.Grad is not null).ToList();
        if (withGrad.Count == 0)
        {
            return;
        }

        var backend = withGrad[0].Backend;
        using var factor = Tensor.PersistentZeros([1], withGrad[0].Device);
        foreach (var c in withGrad)
        {
            backend.SumSquares(c.Grad!.Storage, factor.Storage, c.Size);
        }

        backend.ClipFactor(factor.Storage, factor.Storage, maxNorm);
        foreach (var c in withGrad)
        {
            backend.GroupScaleShift(c.Grad!.Storage, factor.Storage, null, c.Grad.Storage, c.Size, 1, c.Size, false);
        }
    }

    private void Update(bool release)
    {
        if (Staging() is { } staging)
        {
            Pipelined(staging, release);
            return;
        }

        for (int i = 0; i < _copies.Length; i++)
        {
            var (p, copy) = (Parameters[i], _copies[i]);
            if (p.Grad is not { } g)
            {
                copy.ReleaseGrad();
                continue;
            }

            copy.GradientTarget(out _);
            g.Backend.Download(g.Storage, copy.Grad!.Storage.HostMemory[..copy.Size]);
            if (release)
            {
                p.ReleaseGrad();
            }

            Step(i);
            p.Backend.Upload(copy.Storage.HostMemory[..copy.Size], p.Storage);
        }
    }

    // Gradients come down in pieces of at most a slot, a few ahead of the CPU; once a parameter's last piece has arrived it
    // is updated and its new values go up, while the next pieces download.
    private void Pipelined(IHostStaging staging, bool release)
    {
        var pieces = _pieces;
        pieces.Clear();
        for (int i = 0; i < _copies.Length; i++)
        {
            if (Parameters[i].Grad is null)
            {
                _copies[i].ReleaseGrad();                                  // no gradient: its optimizer skips it too
                continue;
            }

            _copies[i].GradientTarget(out _);
            for (int offset = 0; offset < _copies[i].Size; offset += SlotFloats)
            {
                pieces.Add((i, offset, Math.Min(SlotFloats, _copies[i].Size - offset)));
            }
        }

        int next = 0, up = 0;
        void Issue(int slot)
        {
            if (next < pieces.Count)
            {
                var (parameter, offset, count) = pieces[next++];
                staging.Download(Parameters[parameter].Grad!.Storage, offset, slot, count);
            }
        }

        for (int slot = 0; slot < DownSlots; slot++)
        {
            Issue(slot);
        }

        for (int j = 0; j < pieces.Count; j++)
        {
            int slot = j % DownSlots;                                      // pieces are issued to the slots in turn
            var (i, offset, count) = pieces[j];
            staging.Wait(slot);
            staging.Slot(slot)[..count].CopyTo(_copies[i].Grad!.Storage.HostMemory.Slice(offset, count));
            Issue(slot);
            if (j + 1 < pieces.Count && pieces[j + 1].Parameter == i)
            {
                continue;                                                  // more of this parameter to come
            }

            if (release)
            {
                Parameters[i].ReleaseGrad();
            }

            Step(i);
            var values = _copies[i].Storage.HostMemory;
            for (int start = 0; start < _copies[i].Size; start += SlotFloats)
            {
                int n = Math.Min(SlotFloats, _copies[i].Size - start);
                int target = DownSlots + up++ % UpSlots;
                staging.Wait(target);                                      // its previous upload has read it
                values.Slice(start, n).CopyTo(staging.Slot(target));
                staging.Upload(target, Parameters[i].Storage, start, n);
            }
        }
    }

    private void Step(int i)
    {
        _cpu[i].LearningRate = LearningRate;
        _cpu[i].ClipAndStep(0f);                                           // the fused update where the optimizer has one
    }

    /// <inheritdoc />
    public override IReadOnlyDictionary<string, OptimizerStateValue> SaveState()
    {
        var state = new Dictionary<string, OptimizerStateValue>(StringComparer.Ordinal);
        for (int i = 0; i < _cpu.Length; i++)
        {
            foreach (var (name, value) in _cpu[i].SaveState())
            {
                state[$"{i}.{name}"] = value;
            }
        }

        return state;
    }

    /// <inheritdoc />
    public override void LoadState(IReadOnlyDictionary<string, OptimizerStateValue> state)
    {
        ArgumentNullException.ThrowIfNull(state);
        for (int i = 0; i < _cpu.Length; i++)
        {
            string prefix = $"{i}.";
            _cpu[i].LoadState(state.Where(p => p.Key.StartsWith(prefix, StringComparison.Ordinal)).ToDictionary(p => p.Key[prefix.Length..], p => p.Value, StringComparer.Ordinal));
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        _staging?.Dispose();                                               // waits for the last uploads
        _staging = null;
        foreach (var optimizer in _cpu)
        {
            optimizer.Dispose();
        }

        foreach (var copy in _copies)
        {
            copy.Dispose();
        }

        base.Dispose();
    }
}
