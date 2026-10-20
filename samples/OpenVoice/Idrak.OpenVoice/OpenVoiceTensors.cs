// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Formats;

using Idrak.Models;

namespace Idrak.OpenVoice;

/// <summary>
/// An OpenVoice checkpoint's tensors under the names the library's layers read (transformers' VITS names), so the
/// WaveNet, flow and HiFi-GAN generator load them unchanged (<c>FromStored</c>): <c>posterior_encoder.conv_pre</c> is OpenVoice's
/// <c>enc_q.pre</c>, coupling <c>i</c> of <c>flow.flows</c> is <c>flow.flows.{2i}</c> (the flips between take the odd
/// indices), <c>decoder.upsampler</c> is <c>dec.ups</c>. A <c>.weight</c> stored weight-normed (<c>.weight_g</c> and
/// <c>.weight_v</c>) is read folded. Other names (the reference encoder's) pass through. The store is the library's
/// <see cref="RenamedTensorStore"/>; only the names are OpenVoice's.
/// </summary>
internal static class OpenVoiceTensors
{
    /// <summary>
    /// <paramref name="store"/> under the VITS names (<see cref="Map"/>), weight-normed weights folded; disposing it leaves
    /// <paramref name="store"/> open.
    /// </summary>
    public static ITensorStore Over(ITensorStore store) => new RenamedTensorStore(store, Map, foldWeightNorm: true, ownsInner: false);

    /// <summary>The OpenVoice name of a VITS layer's tensor name.</summary>
    public static string Map(string name)
    {
        if (name.StartsWith("posterior_encoder.", StringComparison.Ordinal))
        {
            return "enc_q." + Rest(name["posterior_encoder.".Length..], ("conv_pre", "pre"), ("wavenet", "enc"), ("conv_proj", "proj"));
        }

        if (name.StartsWith("flow.flows.", StringComparison.Ordinal))
        {
            string rest = name["flow.flows.".Length..];
            int dot = rest.IndexOf('.');
            int index = int.Parse(rest.AsSpan(0, dot), System.Globalization.CultureInfo.InvariantCulture);
            return $"flow.flows.{2 * index}." + Rest(rest[(dot + 1)..], ("conv_pre", "pre"), ("wavenet", "enc"), ("conv_post", "post"));
        }

        if (name.StartsWith("decoder.", StringComparison.Ordinal))
        {
            return "dec." + Rest(name["decoder.".Length..], ("upsampler", "ups"));
        }

        return name;
    }

    // The first part of `rest` renamed by the pairs (VITS name, OpenVoice name), the remainder as it is.
    private static string Rest(string rest, params (string From, string To)[] renames)
    {
        foreach (var (from, to) in renames)
        {
            if (rest.StartsWith(from, StringComparison.Ordinal) && (rest.Length == from.Length || rest[from.Length] == '.'))
            {
                return to + rest[from.Length..];
            }
        }

        return rest;
    }
}
