// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// A voice converter under test (Idrak.Audio's <c>IVoiceConverter</c>, or one of your own), given as what the suite needs
/// to drive it without depending on the package: its rate, its embeddings' width, its embedding of mono clips and its
/// conversion of mono samples to a target embedding (both at <paramref name="SampleRate"/>).
/// </summary>
/// <param name="SampleRate">The rate of the samples the suite makes and the converter gives.</param>
/// <param name="EmbeddingSize">The embeddings' width.</param>
/// <param name="Embed">The embedding of mono clips.</param>
/// <param name="Convert">Mono samples in the voice of a target embedding.</param>
/// <param name="LengthSlack">How many samples shorter than its input a conversion may be (a frame's hop: the converter keeps whole frames).</param>
public sealed record VoiceConverterUnderTest(int SampleRate, int EmbeddingSize, Func<IReadOnlyList<float[]>, float[]> Embed, Func<float[], float[], float[]> Convert, int LengthSlack);

/// <summary>
/// The checks of a voice converter on sounds the suite makes (tones and seeded noise, as source and as clips), checks
/// that hold for any converter, trained or not:
/// <list type="bullet">
/// <item>an embedding is as wide as said and finite, the same twice;</item>
/// <item>a conversion is finite, no longer than its input and at most <see cref="VoiceConverterUnderTest.LengthSlack"/> shorter;</item>
/// <item>the same samples twice, and from two calls at once (a converter shared by threads);</item>
/// <item>a conversion to the source's own embedding works as any other.</item>
/// </list>
/// </summary>
public sealed class VoiceConverterSuite : ContractSuite<VoiceConverterUnderTest>
{
    /// <inheritdoc />
    public override string Name => "voice-converter";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("tone to noise", 0.5, 3, 7);
        yield return Case("noise to tone", 0.3, 11, 2);
        yield return Case("short", 0.1, 5, 6);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case("random", Math.Round(0.1 + random.NextDouble() * (large ? 20 : 1), 2), random.Next(1, 1000), random.Next(1, 1000));
    }

    /// <inheritdoc />
    public override void Run(VoiceConverterUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        double seconds = (double)@case.Data["seconds"]!;
        int source = (int)@case.Data["source"]!, target = (int)@case.Data["target"]!, rate = implementation.SampleRate;
        var speech = VoiceSounds.Make(source, seconds, rate);
        var clips = new[] { VoiceSounds.Make(target, 0.6, rate), VoiceSounds.Make(target + 1, 0.4, rate) };
        float[] embedding, converted;
        try
        {
            embedding = implementation.Embed(clips);
            converted = implementation.Convert(speech, embedding);
        }
        catch (Exception e)
        {
            checks.Check("embeds and converts without throwing", false, $"{e.GetType().Name}: {e.Message}");
            return;
        }

        checks.Check("the embedding is as wide as said and finite", embedding.Length == implementation.EmbeddingSize && embedding.All(float.IsFinite),
            $"{embedding.Length} values for {implementation.EmbeddingSize}, {embedding.Count(v => !float.IsFinite(v))} not finite");
        checks.Compare("the same embedding twice", Comparisons.Difference(embedding, implementation.Embed(clips), 0f));
        checks.Check("the conversion is finite", converted.All(float.IsFinite), $"{converted.Count(v => !float.IsFinite(v))} samples are not finite");
        checks.Check("the conversion keeps the length", converted.Length <= speech.Length && converted.Length >= speech.Length - implementation.LengthSlack,
            $"{converted.Length} samples for {speech.Length} (at most {implementation.LengthSlack} fewer)");
        checks.Compare("the same samples twice", Comparisons.Difference(converted, implementation.Convert(speech, embedding), 0f));
        var both = new float[2][];
        Parallel.For(0, 2, i => both[i] = implementation.Convert(speech, embedding));
        float[] twice = [.. converted, .. converted], together = [.. both[0], .. both[1]];
        checks.Compare("the same samples from two calls at once", Comparisons.Difference(twice, together, 0f));
        var own = implementation.Convert(speech, implementation.Embed([speech]));
        checks.Check("a conversion to the source's own embedding", own.Length == converted.Length && own.All(float.IsFinite), $"{own.Length} samples, {own.Count(v => !float.IsFinite(v))} not finite");
    }

    private static ContractCase Case(string name, double seconds, int source, int target) =>
        new($"{name}, {seconds} s, sounds {source} and {target}", new JsonObject { ["seconds"] = seconds, ["source"] = source, ["target"] = target });
}

/// <summary>
/// A voice store under test (Idrak.Audio's <c>IVoiceStore</c> with an encoder, or one of your own), given as what the
/// suite needs to drive it without depending on the package. The suite writes the clips (16-bit WAV files) itself.
/// </summary>
/// <param name="Add">Installs a voice: its name, the clips' files, and the consent statement (null: none, which must be refused by throwing).</param>
/// <param name="Embedding">The installed voice's embedding (looked up ignoring case), or null when there is no voice of that name.</param>
/// <param name="List">The installed voices' names.</param>
/// <param name="Export">Writes the voice to a zip.</param>
/// <param name="Import">Installs the voice of a zip; gives its name.</param>
/// <param name="Remove">Removes the voice; false when there is none.</param>
public sealed record VoiceStoreUnderTest(Action<string, IReadOnlyList<string>, string?> Add, Func<string, float[]?> Embedding, Func<IReadOnlyList<string>> List,
    Action<string, string> Export, Func<string, string> Import, Func<string, bool> Remove);

/// <summary>
/// The checks of a voice store, on voices of clips the suite writes:
/// <list type="bullet">
/// <item>an added voice is listed, found ignoring case, and has a finite embedding;</item>
/// <item>adding it again, or without consent, is refused;</item>
/// <item>exported, removed (once: the second removal finds nothing) and imported, it has the same embedding.</item>
/// </list>
/// </summary>
/// <param name="folder">Where the suite writes its clips and zips (a folder of its own is made inside).</param>
public sealed class VoiceStoreSuite(string folder) : ContractSuite<VoiceStoreUnderTest>
{
    /// <inheritdoc />
    public override string Name => "voice-store";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one clip", "solo", 1, 3);
        yield return Case("three clips", "Trio-3", 3, 8);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int seed = random.Next(1, 1000);
        return Case("random", $"voice_{seed}", 1 + random.Next(large ? 4 : 2), seed);
    }

    /// <inheritdoc />
    public override void Run(VoiceStoreUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        string name = (string)@case.Data["name"]!;
        int clips = (int)@case.Data["clips"]!, seed = (int)@case.Data["seed"]!;
        string work = Path.Combine(folder, $"voice-store-{Guid.NewGuid():N}");
        Directory.CreateDirectory(work);
        try
        {
            var files = new List<string>();
            for (int i = 0; i < clips; i++)
            {
                string file = Path.Combine(work, $"clip{i}.wav");
                VoiceSounds.WriteWav(file, VoiceSounds.Make(seed + i, 1.2, 16000), 16000);
                files.Add(file);
            }

            implementation.Remove(name);
            try
            {
                implementation.Add(name, files, null);
                checks.Check("a voice without consent is refused", false, "Add without consent did not throw");
            }
            catch (Exception)
            {
                checks.Check("a voice without consent is refused", implementation.Embedding(name) is null, "it threw, but a voice was installed");
            }

            implementation.Add(name, files, $"I agree that this test voice {name} is installed.");
            checks.Check("the added voice is listed", implementation.List().Contains(name, StringComparer.OrdinalIgnoreCase), $"'{name}' is not in [{string.Join(", ", implementation.List())}]");
            var embedding = implementation.Embedding(name.ToUpperInvariant());
            checks.Check("found ignoring case, with a finite embedding", embedding is { Length: > 0 } && embedding.All(float.IsFinite),
                embedding is null ? $"'{name.ToUpperInvariant()}' was not found" : $"{embedding.Length} values, {embedding.Count(v => !float.IsFinite(v))} not finite");
            bool again;
            try
            {
                implementation.Add(name, files, "again");
                again = false;
            }
            catch (Exception)
            {
                again = true;
            }

            checks.Check("adding a voice of an installed name is refused", again, "the second Add did not throw");
            string zip = Path.Combine(work, name + ".zip");
            implementation.Export(name, zip);
            bool removed = implementation.Remove(name), second = implementation.Remove(name);
            checks.Check("removed once", removed && !second && implementation.Embedding(name) is null, $"Remove gave {removed}, then {second}");
            string imported = implementation.Import(zip);
            checks.Check("imported under its name", string.Equals(imported, name, StringComparison.OrdinalIgnoreCase), $"imported as '{imported}'");
            var back = implementation.Embedding(name);
            checks.Compare("the same embedding after export and import", back is null ? "not found after the import" : Comparisons.Difference(embedding ?? [], back, 0f));
            implementation.Remove(name);
        }
        finally
        {
            Directory.Delete(work, recursive: true);
        }
    }

    private static ContractCase Case(string label, string name, int clips, int seed) =>
        new($"{label}: '{name}', {clips} clip{(clips == 1 ? "" : "s")}", new JsonObject { ["name"] = name, ["clips"] = clips, ["seed"] = seed });
}

// The sounds the voice suites make: a mix of two tones and seeded noise, different for each kind.
internal static class VoiceSounds
{
    public static float[] Make(int kind, double seconds, int rate)
    {
        var samples = new float[Math.Max(1, (int)Math.Round(seconds * rate))];
        var random = new Random(kind);
        double f1 = 110 + 37 * (kind % 11), f2 = 2.3 * f1, noise = 0.05 + 0.05 * (kind % 3);
        for (int n = 0; n < samples.Length; n++)
        {
            double t = (double)n / rate;
            samples[n] = (float)(0.4 * Math.Sin(2 * Math.PI * f1 * t) + 0.2 * Math.Sin(2 * Math.PI * f2 * t) + noise * (random.NextDouble() * 2 - 1));
        }

        return samples;
    }

    public static void WriteWav(string path, float[] samples, int rate)
    {
        var bytes = new byte[44 + 2 * samples.Length];
        var span = bytes.AsSpan();
        "RIFF"u8.CopyTo(span);
        BinaryPrimitives.WriteInt32LittleEndian(span[4..], 36 + 2 * samples.Length);
        "WAVEfmt "u8.CopyTo(span[8..]);
        BinaryPrimitives.WriteInt32LittleEndian(span[16..], 16);
        BinaryPrimitives.WriteInt16LittleEndian(span[20..], 1);                  // PCM
        BinaryPrimitives.WriteInt16LittleEndian(span[22..], 1);                  // one channel
        BinaryPrimitives.WriteInt32LittleEndian(span[24..], rate);
        BinaryPrimitives.WriteInt32LittleEndian(span[28..], 2 * rate);
        BinaryPrimitives.WriteInt16LittleEndian(span[32..], 2);
        BinaryPrimitives.WriteInt16LittleEndian(span[34..], 16);
        "data"u8.CopyTo(span[36..]);
        BinaryPrimitives.WriteInt32LittleEndian(span[40..], 2 * samples.Length);
        for (int i = 0; i < samples.Length; i++)
        {
            BinaryPrimitives.WriteInt16LittleEndian(span[(44 + 2 * i)..], (short)Math.Round(Math.Clamp(samples[i], -1f, 1f) * 32767));
        }

        File.WriteAllBytes(path, bytes);
    }
}
