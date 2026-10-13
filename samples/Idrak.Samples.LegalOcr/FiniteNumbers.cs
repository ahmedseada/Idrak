using System.Text.Json;
using System.Text.Json.Serialization;

namespace Idrak.Samples.LegalOcr;

/// <summary>
/// Writes a number that is not finite (NaN or infinity, e.g. a diverged training loss) as null: JSON has neither, and one
/// such value would otherwise fail every write that carries it, the page's event streams and the adapter's record.
/// </summary>
internal static class FiniteNumbers
{
    /// <summary>Adds the converters of <see cref="float"/> and <see cref="double"/> (their nullable forms too) to <paramref name="options"/>.</summary>
    public static JsonSerializerOptions Add(JsonSerializerOptions options)
    {
        options.Converters.Add(new SingleConverter());
        options.Converters.Add(new DoubleConverter());
        return options;
    }

    private sealed class SingleConverter : JsonConverter<float>
    {
        public override bool HandleNull => true;

        public override float Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
            reader.TokenType == JsonTokenType.Null ? float.NaN : reader.GetSingle();

        public override void Write(Utf8JsonWriter writer, float value, JsonSerializerOptions options)
        {
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

    private sealed class DoubleConverter : JsonConverter<double>
    {
        public override bool HandleNull => true;

        public override double Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
            reader.TokenType == JsonTokenType.Null ? double.NaN : reader.GetDouble();

        public override void Write(Utf8JsonWriter writer, double value, JsonSerializerOptions options)
        {
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
}
