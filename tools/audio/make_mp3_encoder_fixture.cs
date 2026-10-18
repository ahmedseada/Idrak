#:project ../../src/Idrak.Audio/Idrak.Audio.csproj
#:property PublishAot=false

// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
//
// make_mp3_encoder_fixture.cs: the MP3 encoder's fixture (tests/Idrak.Tests/data/audio/mp3enc), checked by ffmpeg.
//   dotnet run tools/audio/make_mp3_encoder_fixture.cs -c Release --no-cache
// (--no-cache: a file-based app keeps its last build when only a project it references changed).
// Encodes the WAV fixture's signals (wav/; their samples are exact everywhere) with the library's Mp3Encoder at a few
// rates, bit rates and channel modes (the other rates by taking rate44k's samples as 48, 32, 22.05 or 24 kHz, exactly),
// decodes each file with ffmpeg (which must write nothing to stderr at -v error), and writes the files and files.json:
// the encoding's options, its bytes' SHA-256, ffmpeg's length, its SNR against the signal after the encoder's delay and
// the RMS of each channel's 576-sample blocks as ffmpeg decodes them. The tests ("audio codecs: mp3 encode:") encode
// the same signals again (the same bytes) and decode them with the library (ffmpeg's length, SNR and block RMS); ffmpeg
// is not needed then. Needs ffmpeg on the PATH.
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Audio.Abstractions;
using Idrak.Audio;

string root = Directory.GetCurrentDirectory();
while (!File.Exists(Path.Combine(root, "Idrak.slnx")))
{
    root = Path.GetDirectoryName(root) ?? throw new InvalidOperationException("run it inside the repository (no Idrak.slnx above the current folder)");
}

string data = Path.Combine(root, "tests", "Idrak.Tests", "data", "audio");
string folder = Path.Combine(data, "mp3enc");
Directory.CreateDirectory(folder);

// name, source WAV, how its samples are taken (as is; "stereo": a second channel of the samples reversed and halved;
// a rate to label them with; the first samples only), the encoder's options.
(string Name, string Wav, bool Stereo, int Rate, int Take, Mp3EncodeOptions Options)[] cases =
[
    ("pcm16-64k", "pcm16", false, 0, 0, new Mp3EncodeOptions()),
    ("stereo16-96k-joint", "stereo16", false, 0, 0, new Mp3EncodeOptions()),
    ("stereo16-48k-lr", "stereo16", false, 0, 0, new Mp3EncodeOptions { Bitrate = 48, ChannelMode = Mp3ChannelMode.Stereo }),
    ("stereo16-32k-mono", "stereo16", false, 0, 0, new Mp3EncodeOptions { Bitrate = 32, ChannelMode = Mp3ChannelMode.Mono }),
    ("rate44k-64k", "rate44k", false, 0, 0, new Mp3EncodeOptions()),
    ("rate44k-stereo-128k", "rate44k", true, 0, 0, new Mp3EncodeOptions()),
    ("as48k-stereo-160k-lr", "rate44k", true, 48000, 0, new Mp3EncodeOptions { Bitrate = 160, ChannelMode = Mp3ChannelMode.Stereo, Quality = Mp3Quality.High }),
    ("as32k-320k", "rate44k", false, 32000, 16000, new Mp3EncodeOptions { Bitrate = 320 }),
    ("as22k-32k-fast", "rate44k", false, 22050, 0, new Mp3EncodeOptions { Bitrate = 32, Quality = Mp3Quality.Fast }),
    ("as24k-stereo-64k", "rate44k", true, 24000, 0, new Mp3EncodeOptions { Bitrate = 64 }),
];

var files = new JsonObject();
foreach (var (name, wav, stereo, rate, take, options) in cases)
{
    var audio = Signal(Path.Combine(data, "wav", wav + ".wav"), stereo, rate, take);
    byte[] bytes = new Mp3Encoder(options).Encode(audio);
    string path = Path.Combine(folder, name + ".mp3");
    File.WriteAllBytes(path, bytes);
    var (channels, outRate, decoded) = Ffmpeg(path);
    var expected = options.ChannelMode == Mp3ChannelMode.Mono && audio.Channels > 1 ? Mono(audio) : audio;
    var entry = new JsonObject
    {
        ["file"] = name + ".mp3",
        ["wav"] = wav,
        ["stereo"] = stereo,
        ["rate"] = audio.SampleRate,
        ["take"] = take,
        ["bitrate"] = options.Bitrate,
        ["channel_mode"] = options.ChannelMode.ToString(),
        ["quality"] = options.Quality.ToString(),
        ["bytes"] = bytes.Length,
        ["sha256"] = Convert.ToHexStringLower(SHA256.HashData(bytes)),
        ["ffmpeg_channels"] = channels,
        ["ffmpeg_rate"] = outRate,
        ["ffmpeg_samples"] = decoded.Length / channels,
        ["ffmpeg_snr_db"] = Math.Round(Snr(expected, decoded, channels), 3),
        ["ffmpeg_block_rms"] = BlockRms(decoded, channels),
    };
    files[name] = entry;
    Console.WriteLine($"{name}: {bytes.Length} bytes, ffmpeg {channels} x {decoded.Length / channels} at {outRate} Hz, SNR {(double)entry["ffmpeg_snr_db"]!:F2} dB");
}

File.WriteAllText(Path.Combine(folder, "files.json"), files.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"wrote {folder}");
return 0;

static AudioData Signal(string wav, bool stereo, int rate, int take)
{
    var audio = AudioCodecs.Decode(wav);
    if (take > 0)
    {
        audio = audio.Slice(0, take);
    }

    var samples = audio.Samples;
    int n = audio.Length;
    if (stereo)
    {
        var both = new float[2 * n];
        samples.AsSpan(0, n).CopyTo(both);
        for (int i = 0; i < n; i++)
        {
            both[n + i] = 0.5f * samples[n - 1 - i];
        }

        return new AudioData(both, 2, rate > 0 ? rate : audio.SampleRate);
    }

    return new AudioData([.. samples], audio.Channels, rate > 0 ? rate : audio.SampleRate);
}

static AudioData Mono(AudioData audio)
{
    var mono = new float[audio.Length];
    audio.ToMono(mono);
    return new AudioData(mono, 1, audio.SampleRate);
}

// ffmpeg's decoding as interleaved float32, with nothing on stderr at -v error.
static (int Channels, int Rate, float[] Samples) Ffmpeg(string path)
{
    var probe = Run("ffprobe", ["-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels,sample_rate", "-of", "csv=p=0", path], out _).Trim().Split(',');
    int rate = int.Parse(probe[0], CultureInfo.InvariantCulture), channels = int.Parse(probe[1], CultureInfo.InvariantCulture);
    string raw = Path.GetTempFileName();
    try
    {
        Run("ffmpeg", ["-v", "error", "-y", "-i", path, "-f", "f32le", raw], out string errors);
        if (errors.Length > 0)
        {
            throw new InvalidOperationException($"ffmpeg reports errors decoding {path}: {errors}");
        }

        byte[] bytes = File.ReadAllBytes(raw);
        var samples = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, samples, 0, samples.Length * 4);
        return (channels, rate, samples);
    }
    finally
    {
        File.Delete(raw);
    }
}

static string Run(string program, string[] arguments, out string errors)
{
    var start = new ProcessStartInfo(program) { RedirectStandardOutput = true, RedirectStandardError = true };
    foreach (string a in arguments)
    {
        start.ArgumentList.Add(a);
    }

    using var process = Process.Start(start) ?? throw new InvalidOperationException($"{program} did not start (is it on the PATH?)");
    var stderr = process.StandardError.ReadToEndAsync();
    string output = process.StandardOutput.ReadToEnd();
    process.WaitForExit();
    errors = stderr.Result.Trim();
    if (process.ExitCode != 0)
    {
        throw new InvalidOperationException($"{program} {string.Join(' ', arguments)} exited with {process.ExitCode}: {errors}");
    }

    return output;
}

// The signal against ffmpeg's decoding from the encoder's delay on (ffmpeg's samples clamped to [-1, 1] as AudioData's are).
static double Snr(AudioData expected, float[] decoded, int channels)
{
    int n = expected.Length, total = decoded.Length / channels;
    double signal = 0, error = 0;
    for (int c = 0; c < channels; c++)
    {
        for (int i = 0; i < n && i + Mp3Encoder.Delay < total; i++)
        {
            double x = expected.Samples[c * n + i], y = Math.Clamp(decoded[(i + Mp3Encoder.Delay) * channels + c], -1f, 1f);
            signal += x * x;
            error += (x - y) * (x - y);
        }
    }

    return 10 * Math.Log10(signal / error);
}

// Each channel's RMS over blocks of 576 decoded samples (clamped to [-1, 1]), to 7 digits.
static JsonArray BlockRms(float[] decoded, int channels)
{
    int total = decoded.Length / channels;
    var rows = new JsonArray();
    for (int c = 0; c < channels; c++)
    {
        var row = new JsonArray();
        for (int at = 0; at < total; at += 576)
        {
            double sum = 0;
            int end = Math.Min(total, at + 576);
            for (int i = at; i < end; i++)
            {
                double v = Math.Clamp(decoded[i * channels + c], -1f, 1f);
                sum += v * v;
            }

            row.Add(Math.Round(Math.Sqrt(sum / (end - at)), 7));
        }

        rows.Add(row);
    }

    return rows;
}
