// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Backends;

namespace Idrak.Optimizers;

/// <summary>
/// Runs another optimizer's step on the CPU, keeping its state (Adam's two moments: twice the parameters' size) in system
/// memory instead of on the GPU. Each step downloads the gradients, updates CPU copies of the parameters with the CPU's
/// SIMD kernels on all cores, and uploads the new values: only gradients and weights cross PCIe, never the state. On a GPU
/// the copies go through pinned buffers in the background, so one parameter downloads while the previous one is updated
/// and uploaded. Use it when the optimizer state does not fit on the GPU (full fine-tuning of a model that just fits).
/// Works with any optimizer that updates each parameter on its own (SGD, Adam, AdamW, AdamW8Bit):
/// <c>new HostOptimizer(model.Parameters(), ps =&gt; new AdamW(ps, 1e-4f))</c>.
/// </summary>
public sealed class HostOptimizer : Optimizer
{
    // Pinned buffers: four for gradients coming down, two for weights going up, 16 MiB each (a larger tensor moves in
    // pieces), so the pinned memory stays at 96 MiB whatever the model's size.
    private const int SlotFloats = 4 << 20, DownSlots = 4, UpSlots = 2;

    private readonly Tensor[] _copies;                                  // the parameters on the CPU
    private readonly Optimizer[] _cpu;                                  // one per parameter, so each updates as it arrives
    private IHostStaging? _staging;
    private bool _stagingTried;

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
    public override void Step() => Update(release: false);

    /// <summary>
    /// Clips the gradients' global norm on their device, then updates on the CPU (see <see cref="Optimizer.ClipAndStep"/>).
    /// The gradients are released on their device as they are downloaded: the next backward pass writes fresh ones
    /// without zero-filling them first.
    /// </summary>
    public override void ClipAndStep(float maxNorm)
    {
        if (maxNorm > 0f)
        {
            ClipGradientNormOnDevice(maxNorm);                             // on the GPU: no host wait, one pass
        }

        Update(release: true);
        GradientsZeroed = true;
    }

    private void Update(bool release)
    {
        if (!_stagingTried)
        {
            _stagingTried = true;
            _staging = Parameters[0].Backend.CreateHostStaging(DownSlots + UpSlots, SlotFloats);   // null where the device has none (the CPU)
        }

        if (_staging is { } staging)
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
        var pieces = new List<(int Parameter, int Offset, int Count)>();
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
