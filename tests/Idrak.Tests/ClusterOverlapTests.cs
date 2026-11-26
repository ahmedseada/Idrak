// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Layers;
using Idrak.Training;

// Plan 23, phase C4: the gradient-ready hook (a hook per parameter) that lets gradient buckets be reduced while the
// backward pass continues.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterOverlapGroup =
    [
        ("cluster: the gradient-ready hook names each parameter once its backward pass finished it, the last layers first; a checkpointed block's as not final; none outside the scope", ClusterGradientReadyHook),
    ];

    // A classifier of a few layers (several parameters to bucket), without batch norm or dropout.
    private static Sequential OverlapModel(Device device, int width = 32, int layers = 2)
    {
        var init = new Random(5);
        var model = new Sequential { new Linear(2, width, device: device, random: init), new ReLU() };
        for (int i = 1; i < layers; i++)
        {
            model.Add(new Linear(width, width, device: device, random: init));
            model.Add(new ReLU());
        }

        model.Add(new Linear(width, 3, device: device, random: init));
        return model;
    }

    private static void ClusterGradientReadyHook(Device device)
    {
        using var model = OverlapModel(device);
        var parameters = model.Parameters().ToList();
        var input = Tensor.From([.. Enumerable.Range(0, 16).Select(i => (i % 7) * 0.1f)], [8, 2], device);
        var targets = Tensor.From([.. Enumerable.Range(0, 24).Select(i => i % 3 == (i / 3) % 3 ? 1f : 0f)], [8, 3], device);   // one-hot rows
        var seen = new List<(int Parameter, bool Outermost)>();
        using (var scope = new TensorScope())
        {
            var loss = Losses.CrossEntropy(model.Forward(input), targets);
            using (Autograd.OnGradientReady((t, outer) => seen.Add((parameters.IndexOf(t), outer))))
            {
                loss.Backward();
            }
        }

        Check(seen.Count == parameters.Count && seen.Select(s => s.Parameter).Order().SequenceEqual(Enumerable.Range(0, parameters.Count)) && seen.All(s => s.Outermost),
            $"each parameter once, in the outermost pass: {string.Join(", ", seen)}");
        Check(seen.FindIndex(s => s.Parameter >= parameters.Count - 2) < seen.FindIndex(s => s.Parameter < 2),
            $"the last layer's parameters are finished before the first layer's: {string.Join(", ", seen.Select(s => s.Parameter))}");

        // A checkpointed first layer: its parameters are finished inside the checkpoint's backward step, a nested pass.
        var first = (Linear)model[0];
        seen.Clear();
        int outside = 0;
        using (var scope = new TensorScope())
        {
            var hidden = Checkpointing.Checkpoint(x => first.Forward(x), input);
            var rest = model[1].Forward(hidden);
            for (int i = 2; i < model.Count; i++)
            {
                rest = model[i].Forward(rest);
            }

            var loss = Losses.CrossEntropy(rest, targets);
            using (Autograd.OnGradientReady((_, _) => outside++))
            {
            }

            using (Autograd.OnGradientReady((t, outer) => seen.Add((parameters.IndexOf(t), outer))))
            {
                loss.Backward();
            }
        }

        Check(seen.Where(s => s.Parameter < 2).All(s => !s.Outermost) && seen.Count(s => s.Parameter < 2) == 2 && seen.Where(s => s.Parameter >= 2).All(s => s.Outermost) && outside == 0,
            $"a checkpointed block's parameters are not final in its nested pass; the others are; nothing is heard outside a scope: {string.Join(", ", seen)}");
        input.Dispose();
        targets.Dispose();
    }
}
