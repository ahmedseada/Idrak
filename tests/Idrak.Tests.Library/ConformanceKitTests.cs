// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Testing;
using Idrak.Generation;

// Plan 10, phases 5 and 6c: the testing kit (Idrak.Abstraction.Testing): the tokenizers' contract checks and stress runs, once.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ConformanceKitGroup =
    [
        ("conformance kit: the library's tokenizers pass their contract suite; one that drops characters fails an exact round trip", ContractsConformant),
        ("conformance kit: a tokenizer's contract stress run (threads) passes", ContractStressAndRegression),
    ];

    internal static void ContractsConformant(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ContractsConformant)))
        {
            return;
        }

        Conformance.Check(new CharTokenizer("abcdefghijklmnopqrstuvwxyz ,.!?"), new CharTokenizer("abcdefghijklmnopqrstuvwxyz ,.!?")).ThrowIfFailed();
        var words = WordTokenizer.FromTexts(["hello world, the model is learning.", "the data and the token"], ["<unk>", "<s>"]);
        Conformance.Check(words, words).ThrowIfFailed();
        var exact = Conformance.Check(new CharTokenizer("ab"), options: null);
        exact.ThrowIfFailed();
        Check(!Conformance.Check(new CharTokenizer("ab"), new TokenizerSuite { ExactRoundTrip = true }).Passed, "dropping characters breaks an exact round trip");
    }

    internal static void ContractStressAndRegression(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ContractStressAndRegression)))
        {
            return;
        }

        var options = new StressOptions { Iterations = 60, Threads = 4 };
        Stress.Run(new CharTokenizer("abcdefghijklmnopqrstuvwxyz "), new TokenizerSuite(), options).ThrowIfFailed();
    }
}
