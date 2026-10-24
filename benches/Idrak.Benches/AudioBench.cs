// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins;

namespace Idrak.Benches;

/// <summary>
/// Every registered audio processor (the library's and the plug-ins', <c>AudioProcessors.Names</c>, so a new one is
/// measured without a change here) run alone over 10 s of speech-like sound at 16 kHz mono and at 48 kHz stereo:
/// <c>AudioPipeline.Apply</c> cold (the first call in the process: compiling, a model read) and the warm median, with the
/// real-time factor (seconds of sound a second). IDRAK_BENCH_AUDIO, a pipeline text, chooses the steps instead (each
/// measured alone, with its value and options). A processor whose value has no default is given one from its range
/// (the other rate for a rate in Hz, 1 kHz for a frequency, 6 dB for a level, else the middle of its range), and so is
/// an option it needs. A step that cannot run (rnnoise without its model
/// file, a rate it refuses) is skipped with the reason. The processors run on the host: the section is measured once,
/// under the first device.
/// </summary>
internal static class AudioBench
{
    private const string Section = "audio-process";
    private const double Seconds = 10;

    public static void Run(Device device, Settings settings, Report report)
    {
        if (!device.Equals(settings.Devices[0]))
        {
            Console.WriteLine($"   the audio processors run on the host: measured once, under {settings.Devices[0]}");
            return;
        }

        AudioPlugins.RegisterAll();
        (string Name, AudioData Sound)[] sounds = [("16k mono", Speech(16000, 1)), ("48k stereo", Speech(48000, 2))];
        // IDRAK_BENCH_AUDIO: comma-separated steps, each measured alone; a bare name takes its defaults, as the list does.
        var steps = new List<(string Name, AudioProcessorStep? Given)>();
        if (Environment.GetEnvironmentVariable("IDRAK_BENCH_AUDIO") is { Length: > 0 } chosen)
        {
            foreach (string item in chosen.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            {
                try
                {
                    steps.Add(item.AsSpan().IndexOfAny("= \t") < 0 ? (AudioProcessors.Get(item).Name, null) : (AudioPipeline.Parse(item).Steps[0].Name, AudioPipeline.Parse(item).Steps[0]));
                }
                catch (ArgumentException e)
                {
                    Console.WriteLine($"   IDRAK_BENCH_AUDIO '{item}': skipped: {e.Message}");
                }
            }
        }
        else
        {
            steps.AddRange(AudioProcessors.Names.Select(n => (n, (AudioProcessorStep?)null)));
        }

        foreach (var (name, given) in steps)
        {
            var processor = AudioProcessors.Get(name);
            foreach (var (soundName, sound) in sounds)
            {
                var step = given ?? DefaultStep(processor, sound.SampleRate);
                string label = $"{step} {soundName}";
                AudioPipeline pipeline;
                double cold, warm;
                try
                {
                    pipeline = new AudioPipeline([step]);
                    (cold, warm) = Bench.Time(device, () => pipeline.Apply(sound), settings.Repeats, settings.Warmup);
                }
                catch (Exception e) when (e is IOException or ArgumentException or InvalidDataException or NotSupportedException)
                {
                    Console.WriteLine($"   {label}: skipped: {e.Message}");
                    continue;
                }

                double factor = Seconds * 1000 / warm;
                Console.WriteLine(Bench.Line(label, ("ms cold", Bench.F(cold, "0.0")), ("ms warm", Bench.F(warm, "0.00")), ("x real time", Bench.F(factor, "0"))));
                report.Add(Section, label, device, "cold", cold, "ms");
                report.Add(Section, label, device, "warm", warm, "ms");
                report.Add(Section, label, device, "real time", factor, "x real time");
            }
        }
    }

    // The processor's step at its defaults; a value or an option it needs but has no default for, from its range.
    private static AudioProcessorStep DefaultStep(IAudioProcessor processor, int rate)
    {
        string? value = null;
        var options = new List<KeyValuePair<string, string>>();
        foreach (var parameter in processor.Parameters.Where(p => !p.Optional && p.Default is null))
        {
            string chosen = Choose(parameter, rate);
            if (parameter.Key.Length == 0)
            {
                value = chosen;
            }
            else
            {
                options.Add(KeyValuePair.Create(parameter.Key, chosen));
            }
        }

        return new AudioProcessorStep(processor.Name, value, options);
    }

    private static string Choose(AudioProcessorParameter value, int rate)
    {
        if (value.Input == AudioProcessorInput.Choice && value.Choices.Count > 0)
        {
            return value.Choices[0];
        }

        if (value.Input == AudioProcessorInput.Switch)
        {
            return "true";
        }

        double chosen = value.Unit == "Hz" && value.Input == AudioProcessorInput.Integer ? (rate == 48000 ? 16000 : 48000)  // a rate: the other one
            : value.Unit == "Hz" ? 1000                                                                                     // a frequency: inside any rate's band
            : value.Unit == "dB" ? 6                                                                                        // a level: a usual boost
            : value.Min is { } min && value.Max is { } max ? (min + max) / 2
            : value.Min ?? value.Max ?? 1;
        if (value.Max is { } top && chosen > top)
        {
            chosen = top;
        }

        if (value.Min is { } bottom && (chosen < bottom || (value.MinExclusive && chosen == bottom)))
        {
            chosen = bottom + (value.MinExclusive ? value.Step ?? 1 : 0);
        }

        return value.Input == AudioProcessorInput.Integer
            ? ((int)Math.Round(chosen)).ToString(CultureInfo.InvariantCulture)
            : chosen.ToString(CultureInfo.InvariantCulture);
    }

    // 10 s of speech-like sound (seeded): syllables of harmonics of a gliding pitch with two formants, pauses, a noise
    // floor under them, a second channel later and quieter.
    private static AudioData Speech(int rate, int channels)
    {
        int length = (int)(Seconds * rate);
        var random = new Random(31);
        var one = new float[length];
        double phase = 0;
        for (int i = 0; i < length; i++)
        {
            double t = (double)i / rate, syllable = t % 0.5;
            bool voiced = syllable < 0.35;
            double f0 = 110 + 60 * syllable / 0.35, envelope = voiced ? Math.Sin(Math.PI * syllable / 0.35) : 0;
            phase += 2 * Math.PI * f0 / rate;
            double tone = 0;
            if (envelope > 0)
            {
                for (int k = 1; k <= 30 && k * f0 < rate / 2.0 - 500; k++)
                {
                    double fk = k * f0, shape = 1.0 / k + 1.5 * Math.Exp(-Math.Pow((fk - 700) / 250, 2)) + 0.8 * Math.Exp(-Math.Pow((fk - 1800) / 400, 2));
                    tone += shape * Math.Sin(k * phase);
                }
            }

            one[i] = (float)(0.08 * envelope * tone + 0.01 * (2 * random.NextDouble() - 1));
        }

        var samples = new float[channels * length];
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < length; i++)
            {
                samples[c * length + i] = one[(i + c * rate / 7) % length] / (1 + c);
            }
        }

        return new AudioData(samples, channels, rate);
    }
}
