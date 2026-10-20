// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Serialization;

namespace Idrak.Abstraction;

/// <summary>
/// Numbers that are not finite (NaN, infinity: a diverged loss, a rate of nothing over no time) in JSON, which has
/// neither: they are written as null, so one such value does not fail every write that carries it, and null reads back
/// as NaN. The converters (<see cref="FiniteSingleJsonConverter"/>, <see cref="FiniteDoubleJsonConverter"/>) serve
/// <see cref="JsonSerializer"/>; <see cref="WriteNumber(Utf8JsonWriter, string, double)"/> and
/// <see cref="WriteNumberValue(Utf8JsonWriter, double)"/> serve a hand-written <see cref="Utf8JsonWriter"/> (the profile
/// and telemetry writers). One fixed meaning, so a helper and not a registry.
/// </summary>
/// <example>
/// <code>
/// var options = FiniteJsonNumbers.Add(new JsonSerializerOptions());
/// JsonSerializer.Serialize(new { loss = double.NaN }, options);   // {"loss":null}
/// </code>
/// </example>
public static class FiniteJsonNumbers
{
    /// <summary>
    /// Adds the converters of <see cref="float"/> and <see cref="double"/> (their nullable forms too) to
    /// <paramref name="options"/>, unless it has them already; returns <paramref name="options"/>.
    /// </summary>
    public static JsonSerializerOptions Add(JsonSerializerOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!options.Converters.Any(c => c is FiniteSingleJsonConverter))
        {
            options.Converters.Add(new FiniteSingleJsonConverter());
        }

        if (!options.Converters.Any(c => c is FiniteDoubleJsonConverter))
        {
            options.Converters.Add(new FiniteDoubleJsonConverter());
        }

        return options;
    }

    /// <summary>Writes the property <paramref name="name"/> as <paramref name="value"/>, or null when it is not finite.</summary>
    public static void WriteNumber(Utf8JsonWriter writer, string name, double value)
    {
        ArgumentNullException.ThrowIfNull(writer);
        if (double.IsFinite(value))
        {
            writer.WriteNumber(name, value);
        }
        else
        {
            writer.WriteNull(name);
        }
    }

    /// <summary>Writes <paramref name="value"/> (an array element, say), or null when it is not finite.</summary>
    public static void WriteNumberValue(Utf8JsonWriter writer, double value)
    {
        ArgumentNullException.ThrowIfNull(writer);
        if (double.IsFinite(value))
        {
            writer.WriteNumberValue(value);
        }
        else
        {
            writer.WriteNullValue();
        }
    }
}

/// <summary>Writes a <see cref="float"/> that is not finite as null, and reads null as <see cref="float.NaN"/> (see <see cref="FiniteJsonNumbers"/>).</summary>
public sealed class FiniteSingleJsonConverter : JsonConverter<float>
{
    /// <inheritdoc />
    public override bool HandleNull => true;

    /// <inheritdoc />
    public override float Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
        reader.TokenType == JsonTokenType.Null ? float.NaN : reader.GetSingle();

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, float value, JsonSerializerOptions options)
    {
        ArgumentNullException.ThrowIfNull(writer);
        if (float.IsFinite(value))
        {
            writer.WriteNumberValue(value);
        }
        else
        {
            writer.WriteNullValue();
        }
    }
}

/// <summary>Writes a <see cref="double"/> that is not finite as null, and reads null as <see cref="double.NaN"/> (see <see cref="FiniteJsonNumbers"/>).</summary>
public sealed class FiniteDoubleJsonConverter : JsonConverter<double>
{
    /// <inheritdoc />
    public override bool HandleNull => true;

    /// <inheritdoc />
    public override double Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
        reader.TokenType == JsonTokenType.Null ? double.NaN : reader.GetDouble();

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, double value, JsonSerializerOptions options)
    {
        ArgumentNullException.ThrowIfNull(writer);
        FiniteJsonNumbers.WriteNumberValue(writer, value);
    }
}
