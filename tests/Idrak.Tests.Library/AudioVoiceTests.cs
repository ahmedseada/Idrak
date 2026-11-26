// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio.Abstractions;

// Plan 21, phase S9 route B: the tiny OpenVoice fixture (tools/audio/make_tiny_audio.py --sections voices) and its sounds,
// and the relative closeness the audio tests check with.
internal abstract partial class Tests
{
    internal static string TinyOpenVoice => AudioData("tiny-openvoice");

    // The fixture's sounds: wav/pcm16 (the source) and the second channel of wav/stereo16 and its second half (the clips), all read as 16 kHz.
    internal static (AudioData Source, AudioData[] Clips) OpenVoiceInputs()
    {
        var source = new AudioData(ReadNpyFloat32(AudioData("wav/pcm16.npy")), 1, 16000);
        var stereo = ReadNpyFloat32(AudioData("wav/stereo16.npy"));
        var other = stereo[(stereo.Length / 2)..];
        return (source, [new AudioData(other, 1, 16000), new AudioData(other[(other.Length / 2)..], 1, 16000)]);
    }

    // The largest difference over the largest expected value, checked against `relative`.
    internal static void CloseRelative(float[] expected, float[] actual, float relative, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: {actual.Length} values for {expected.Length}");
        float peak = 0, worst = 0;
        int at = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            peak = Math.Max(peak, Math.Abs(expected[i]));
            float d = Math.Abs(expected[i] - actual[i]);
            if (!(d <= worst))
            {
                worst = d;
                at = i;
            }
        }

        Check(worst <= relative * Math.Max(peak, 1e-12f), $"{what}: differs by {worst:E3} at {at} ({expected[at]} against {actual[at]}), {worst / Math.Max(peak, 1e-12f):E3} of the peak {peak:E3}, over {relative:E1}");
    }
}
