// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 21, phase S2: resampling, the short-time Fourier transform and log-mel features against torchaudio's and
// transformers' (tests/Idrak.Tests/data/audio: resample/, features/).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioFeatureGroup =
    [
        ("audio features: torchaudio's windowed-sinc resampling of 44.1 and 8 kHz to 16 kHz within 1e-5 on the host and on the device, the two agreeing; a sound at the rate already is itself", AudioResampling),
        ("audio features: the STFT magnitude and the log-mel of pcm16.wav within 1e-4 of torch's on the host path and the composed device path, the two magnitudes within 1e-5; the frame counts, the mel filterbank within 1e-6, preprocessor_config.json to the options", AudioLogMel),
        ("audio features: fft: the library's real FFT (forward and inverse) against a direct DFT in double within 1e-9 of the largest bin at 16, 400, 480, 512, 1024 and 2048 points and at odd and prime-factor sizes; the windowed float frame gives the double frame's bits; the same bits twice; the plan shared per size; a transform allocates nothing once warm", AudioFft),
        ("audio features: a ten-minute signal extracted in 30 s segments allocates the same device bytes for every segment after the first; the registry names log-mel and the STFT suite passes on both paths; cold and warm times", AudioFeatureMemoryAndSuite),
    ];

    private static AudioFeatureOptions FixtureFeatureOptions()
    {
        var json = JsonNode.Parse(File.ReadAllText(AudioData("tiny-whisper/preprocessor_config.json")))!.AsObject();
        return AudioFeatureOptions.FromConfig(json);
    }

    private static void AudioResampling(Device device)
    {
        foreach (var (name, rate) in new[] { ("rate44k", 44100), ("rate8k", 8000) })
        {
            var source = AudioCodecs.Decode(AudioData($"wav/{name}.wav"));
            var expected = ReadNpyFloat32(AudioData($"resample/{name}-16k.npy"));
            Check(source.SampleRate == rate && Resampler.OutputLength(source.Length, rate, 16000) == expected.Length, $"{name}: {expected.Length} samples expected");
            if (device.Type == DeviceType.Cpu)
            {
                var host = Resampler.Resample(source, 16000);
                Check(host.SampleRate == 16000 && host.Channels == 1 && host.Length == expected.Length, $"{name}: host {host.Length} samples at {host.SampleRate} Hz");
                AssertClose(expected, host.Samples, 1e-5f, $"{name}: the host resampling against torchaudio");
                Check(ReferenceEquals(Resampler.Resample(host, 16000), host), "a sound at the rate already is itself");
            }

            using var onDevice = Resampler.ResampleOnDevice(source, 16000, device);
            Check(onDevice.Shape.SequenceEqual([1, expected.Length]), $"{name}: the device gives [{string.Join(", ", onDevice.Shape.ToArray())}]");
            AssertClose(expected, onDevice.ToArray(), 1e-5f, $"{name}: the device resampling on {device} against torchaudio");
        }
    }

    private static void AudioLogMel(Device device)
    {
        var options = FixtureFeatureOptions();
        var fixture = JsonNode.Parse(File.ReadAllText(AudioData("features/features.json")))!.AsObject();
        Check(options.SampleRate == 16000 && options.WindowSize == 400 && options.HopSize == 160 && options.Mels == 80 && options.PadSeconds == 30,
            $"preprocessor_config.json reads to the fixture's options: {options.ToJson()}");
        var audio = AudioCodecs.Decode(AudioData("wav/pcm16.wav"));
        int stftFrames = (int)fixture["stft_frames"]!, melFrames = (int)fixture["log_mel_frames"]!;
        Check(Stft.FramesFor(audio.Length, options) == stftFrames && Stft.FramesFor(audio.Length, options with { Center = false }) == (audio.Length - 400) / 160 + 1,
            "the frame counts with and without centring");

        // The filterbank, [bins, mels].
        var filterbank = ReadNpyFloat32(AudioData("features/mel-filterbank.npy"));
        AssertClose(filterbank, MelFilterbank.Build(options), 1e-6f, "the mel filterbank against torchaudio's");

        // The STFT magnitude: torch's is [bins, frames]; ours [frames, bins].
        var magnitude = ReadNpyFloat32(AudioData("features/stft-magnitude.npy"));
        int bins = options.Bins;
        var expected = new float[stftFrames * bins];
        for (int k = 0; k < bins; k++)
        {
            for (int f = 0; f < stftFrames; f++)
            {
                expected[f * bins + k] = magnitude[k * stftFrames + f];
            }
        }

        float largest = expected.Max();
        var power = new float[stftFrames * bins];
        Stft.PowerOnHost(audio.Samples, options, power);
        var hostMagnitude = power.Select(MathF.Sqrt).ToArray();
        AssertClose(expected, hostMagnitude, 1e-4f * largest, "the host STFT magnitude against torch's");
        using (var composed = Stft.Magnitude(audio.Samples, options, device))
        {
            var values = composed.ToArray();
            AssertClose(expected, values, 1e-4f * largest, $"the composed STFT magnitude on {device} against torch's");
            AssertClose(hostMagnitude, values, 1e-5f * largest, $"the host and composed magnitudes agree on {device}");
        }

        // The log-mel features: torch's [80, 3000]; ours [3000, 80].
        var logMel = ReadNpyFloat32(AudioData("features/log-mel.npy"));
        var expectedMel = new float[melFrames * options.Mels];
        for (int m = 0; m < options.Mels; m++)
        {
            for (int f = 0; f < melFrames; f++)
            {
                expectedMel[f * options.Mels + m] = logMel[m * melFrames + f];
            }
        }

        float melLargest = expectedMel.Max(MathF.Abs);
        var extractor = AudioFeatureExtractors.Create("log-mel", options);
        Check(extractor.FramesFor(audio.Length) == melFrames, $"FramesFor gives {extractor.FramesFor(audio.Length)}, the fixture {melFrames}");
        float[]? hostValues = null;
        foreach (string path in new[] { "host", "composed" })
        {
            var on = AudioFeatureExtractors.Create("log-mel", options with { Path = path });
            using var features = on.Extract(audio, device);
            Check(features.Shape.SequenceEqual([melFrames, options.Mels]), $"{path}: [{string.Join(", ", features.Shape.ToArray())}]");
            var values = features.ToArray();
            AssertClose(expectedMel, values, 1e-4f * melLargest, $"the {path} log-mel on {device} against transformers'");
            if (hostValues is null)
            {
                hostValues = values;
            }
            else
            {
                // The log magnifies the composed path's float rounding in quiet bins (its window sum is in float, the host's FFT
                // in double): the paths agree within the reference's bound, not the magnitudes' tighter one.
                AssertClose(hostValues, values, 1e-4f * melLargest, $"the host and composed log-mel agree within the reference's bound on {device}");
            }
        }

        Throws<ArgumentException>(() => new AudioFeatureOptions { MelScale = "mel" }.Check(), "an unknown mel scale");
        Throws<ArgumentException>(() => AudioFeatureExtractors.Create("log-mel", options with { Path = "gpu" }), "an unknown path");
        Throws<NotSupportedException>(() => AudioFeatureExtractors.Create("mfcc"), "an unregistered extractor");
    }

    private static void AudioFeatureMemoryAndSuite(Device device)
    {
        var options = FixtureFeatureOptions();
        Check(AudioFeatureExtractors.Names.Contains("log-mel") && AudioFeatureExtractors.Origin("log-mel") == Overrides.Library, $"the registry: {string.Join(", ", AudioFeatureExtractors.Names)}");

        // Ten minutes in 30 s segments: the device holds the same bytes after each segment (buffers reused, rule 77).
        const int Rate = 16000;
        var random = new Random(5);
        var tenMinutes = new float[Rate * 60 * 10];
        for (int i = 0; i < tenMinutes.Length; i++)
        {
            tenMinutes[i] = (float)(random.NextDouble() * 2 - 1) * 0.5f;
        }

        var audio = new AudioData(tenMinutes, 1, Rate);
        var extractor = AudioFeatureExtractors.Create("log-mel", options);
        int segment = Rate * 30;
        var times = new List<double>();
        var held = new List<long>();
        for (int from = 0; from < audio.Length; from += segment)
        {
            var slice = audio.Slice(from, Math.Min(segment, audio.Length - from));
            var watch = Stopwatch.StartNew();
            using (var features = extractor.Extract(slice, device))
            {
                Check(features.Shape[0] == extractor.FramesFor(slice.Length), "a segment's frames");
            }

            times.Add(watch.Elapsed.TotalMilliseconds);
            held.Add(device.Backend.GetMemoryUsage().InUse);
        }

        // The first segment makes the cached basis and filterbank; from the second on the device holds the same bytes.
        Check(held.Skip(1).All(b => b == held[1]), $"device bytes after each segment: {string.Join(", ", held)} (not the same after the first)");
        Console.WriteLine($"    log-mel of 30 s on {device}: cold {times[0]:F1} ms, warm {times.Skip(1).Average():F1} ms ({extractor.Options.Path}: {((LogMelFeatures)extractor).PathFor(device)})");

        foreach (string path in new[] { "host", "composed" })
        {
            var on = options with { Path = path, Center = true };
            var window = Stft.Window(on.Window, on.WindowSize);
            PowerSpectra Power(float[] signal)
            {
                int frames = Stft.FramesFor(signal.Length, on);
                if (path == "host")
                {
                    var values = new float[frames * on.Bins];
                    Stft.PowerOnHost(signal, on, values);
                    return new PowerSpectra(frames, on.Bins, values);
                }

                using var tensor = Stft.Power(signal, on, device);
                return new PowerSpectra(frames, on.Bins, tensor.ToArray());
            }

            var report = Conformance.CheckStft(new StftUnderTest(on.WindowSize, on.HopSize, on.Center, n => window[n], Power), options.SampleRate);
            Check(report.Passed, $"the STFT suite on the {path} path on {device}: {report}");
        }
    }

    private static void AudioFft(Device device)
    {
        _ = device;
        var random = new Random(21);
        foreach (int size in new[] { 16, 400, 480, 512, 1024, 2048, 2, 15, 49, 154, 802 })
        {
            var fft = Fft.For(size);
            Check(ReferenceEquals(Fft.For(size), fft) && fft.Size == size && fft.Bins == size / 2 + 1, $"{size}: one plan per size, {fft.Bins} bins");
            var cos = new double[size];
            var sin = new double[size];
            for (int e = 0; e < size; e++)
            {
                cos[e] = Math.Cos(2 * Math.PI * e / size);
                sin[e] = Math.Sin(2 * Math.PI * e / size);
            }

            var x = new double[size];
            for (int n = 0; n < size; n++)
            {
                x[n] = random.NextDouble() * 2 - 1;
            }

            var scratch = new double[fft.ScratchLength];
            var re = new double[fft.Bins];
            var im = new double[fft.Bins];
            fft.Forward(x, re, im, scratch);
            double error = 0, largest = 0;
            for (int k = 0; k < fft.Bins; k++)
            {
                double dr = 0, di = 0;
                for (int n = 0; n < size; n++)
                {
                    int e = (int)((long)n * k % size);
                    dr += x[n] * cos[e];
                    di -= x[n] * sin[e];
                }

                error = Math.Max(error, Math.Max(Math.Abs(dr - re[k]), Math.Abs(di - im[k])));
                largest = Math.Max(largest, Math.Sqrt(dr * dr + di * di));
            }

            Check(error <= 1e-9 * largest, $"{size}: the forward FFT is {error:E2} off the direct DFT (largest bin {largest:E2})");
            Check(im[0] == 0 && (size % 2 == 1 || im[size / 2] == 0), $"{size}: the imaginary parts at 0 Hz and N / 2 are 0");

            // The inverse of a random spectrum against the direct irfft (the imaginary parts at 0 Hz and N / 2 left out).
            var specRe = new double[fft.Bins];
            var specIm = new double[fft.Bins];
            for (int k = 0; k < fft.Bins; k++)
            {
                specRe[k] = random.NextDouble() * 2 - 1;
                specIm[k] = random.NextDouble() * 2 - 1;
            }

            var back = new double[size];
            fft.Inverse(specRe, specIm, back, scratch);
            double inverseError = 0, inverseLargest = 0;
            for (int n = 0; n < size; n++)
            {
                double sum = specRe[0];
                for (int k = 1; k < size; k++)
                {
                    int e = (int)((long)n * k % size);
                    int b = k < fft.Bins ? k : size - k;
                    double xr = specRe[b], xi = k < fft.Bins ? specIm[b] : -specIm[b];
                    if (size % 2 == 0 && b == size / 2)
                    {
                        xi = 0;
                    }

                    sum += xr * cos[e] - xi * sin[e];                           // Re(X[k] e^(+2πi n k / N))
                }

                sum /= size;
                inverseError = Math.Max(inverseError, Math.Abs(sum - back[n]));
                inverseLargest = Math.Max(inverseLargest, Math.Abs(sum));
            }

            Check(inverseError <= 1e-9 * inverseLargest, $"{size}: the inverse FFT is {inverseError:E2} off the direct inverse DFT (largest sample {inverseLargest:E2})");
            fft.Inverse(re, im, back, scratch);
            double roundTrip = 0;
            for (int n = 0; n < size; n++)
            {
                roundTrip = Math.Max(roundTrip, Math.Abs(back[n] - x[n]));
            }

            Check(roundTrip <= 1e-12, $"{size}: the inverse of the forward is {roundTrip:E2} off the samples");

            // A float frame windowed as it is read gives the bits of the double frame windowed first; twice the same bits.
            var frame = new float[size];
            var window = Stft.Window("hann", size);
            var windowed = new double[size];
            for (int n = 0; n < size; n++)
            {
                frame[n] = (float)x[n];
                windowed[n] = window[n] * frame[n];
            }

            var floatRe = new double[fft.Bins];
            var floatIm = new double[fft.Bins];
            fft.Forward(frame, window, floatRe, floatIm, scratch);
            fft.Forward(windowed, re, im, scratch);
            Check(floatRe.AsSpan().SequenceEqual(re) && floatIm.AsSpan().SequenceEqual(im), $"{size}: the windowed float frame gives the windowed double frame's bits");
            fft.Forward(windowed, specRe, specIm, new double[fft.ScratchLength]);
            Check(specRe.AsSpan().SequenceEqual(re) && specIm.AsSpan().SequenceEqual(im), $"{size}: the same frame twice gives the same bits");
        }

        foreach (int size in new[] { 400, 512 })
        {
            var fft = Fft.For(size);
            var scratch = new double[fft.ScratchLength];
            var x = new double[size];
            var re = new double[fft.Bins];
            var im = new double[fft.Bins];
            x[1] = 1;
            void Both()
            {
                fft.Forward(x, re, im, scratch);
                fft.Inverse(re, im, x, scratch);
            }

            for (int i = 0; i < 64; i++)
            {
                Both();
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 256; i++)
            {
                Both();
            }

            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Check(allocated == 0, $"{size}: 256 forward and inverse transforms allocated {allocated} bytes once warm{(allocated == 0 ? "" : $" ({AllocatedTypes(Both)})")}");
        }
    }
}
