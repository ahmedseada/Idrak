// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// An encoder-decoder model described by its settings: the decoder as a <see cref="DecoderSpec"/> (its blocks take a
/// cross-attention over the encoder's output: Idrak's <c>DecoderBlock</c> with a <c>CrossAttention</c>), and the encoder
/// as the family's own description, which the library writes and reads back without interpreting (a speech family's
/// convolutional front end and mel count, a text encoder's vocabulary). Written to and read from a model's config.json by
/// its family (<see cref="ToJson"/>, <see cref="FromJson"/>).
/// </summary>
/// <param name="Decoder">The decoder's settings.</param>
/// <param name="Encoder">The encoder's settings, as the family describes them.</param>
public sealed record EncoderDecoderSpec(DecoderSpec Decoder, JsonObject Encoder)
{
    private const string Format = "idrak-encoder-decoder/1";

    /// <summary>The spec as JSON (format "idrak-encoder-decoder/1"): the decoder's JSON under "decoder", the encoder's under "encoder".</summary>
    public JsonObject ToJson()
    {
        ArgumentNullException.ThrowIfNull(Decoder);
        ArgumentNullException.ThrowIfNull(Encoder);
        return new JsonObject { ["format"] = Format, ["decoder"] = Decoder.ToJson(), ["encoder"] = Encoder.DeepClone() };
    }

    /// <summary>Reads a spec written by <see cref="ToJson"/>.</summary>
    /// <exception cref="InvalidDataException">Not an encoder-decoder description.</exception>
    public static EncoderDecoderSpec FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        if (!IsDescription(json))
        {
            throw new InvalidDataException($"Not an encoder-decoder description (format '{json["format"]}').");
        }

        var decoder = json["decoder"] as JsonObject ?? throw new InvalidDataException("An encoder-decoder description has a \"decoder\" object.");
        var encoder = json["encoder"] as JsonObject ?? throw new InvalidDataException("An encoder-decoder description has an \"encoder\" object.");
        return new EncoderDecoderSpec(DecoderSpec.FromJson(decoder), (JsonObject)encoder.DeepClone());
    }

    /// <summary>Whether <paramref name="json"/> is an encoder-decoder description.</summary>
    public static bool IsDescription(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        return (string?)json["format"] == Format;
    }
}
