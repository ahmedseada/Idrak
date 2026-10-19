// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// Reads a Hugging Face tokenizer.json, of any model type: the model's vocabulary (an object of token to id, or a list of
/// [piece, score]) and merges straight from the document, the small rest as a JSON tree.
/// </summary>
internal sealed class TokenizerJsonReader : ITokenizerFileReader
{
    public string Name => TokenizerFiles.TokenizerJson;

    public string Summary => "Hugging Face tokenizer.json (BPE, Unigram, WordPiece or a registered model type)";

    public string? Find(string path)
    {
        if (Directory.Exists(path))
        {
            string file = Path.Combine(path, "tokenizer.json");
            return File.Exists(file) ? file : null;
        }

        return File.Exists(path) && path.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ? path : null;
    }

    public TokenizerDefinition Read(string file, TokenizerOptions options) => ReadFile(file);

    // The vocabulary and merges (most of the file) are read straight from the document; only the small rest becomes a
    // JSON object tree.
    internal static TokenizerDefinition ReadFile(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(path), new JsonDocumentOptions { MaxDepth = 256 });
        var root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"{path} is not a tokenizer.json (its root is not an object).");
        }

        var rest = new JsonObject();
        Dictionary<string, int>? vocabulary = null;
        List<(string, double)>? pieces = null;
        (string, string)[]? merges = null;
        foreach (var property in root.EnumerateObject())
        {
            if (property.NameEquals("model") && property.Value.ValueKind == JsonValueKind.Object)
            {
                var model = new JsonObject();
                foreach (var field in property.Value.EnumerateObject())
                {
                    if (field.NameEquals("vocab") && field.Value.ValueKind == JsonValueKind.Object)
                    {
                        vocabulary = new Dictionary<string, int>(CountProperties(field.Value));
                        foreach (var entry in field.Value.EnumerateObject())
                        {
                            vocabulary[entry.Name] = entry.Value.GetInt32();
                        }
                    }
                    else if (field.NameEquals("vocab") && field.Value.ValueKind == JsonValueKind.Array)
                    {
                        pieces = new List<(string, double)>(field.Value.GetArrayLength());
                        foreach (var entry in field.Value.EnumerateArray())
                        {
                            if (entry.ValueKind != JsonValueKind.Array || entry.GetArrayLength() != 2)
                            {
                                throw new InvalidDataException($"{path}: model.vocab entry {pieces.Count} is not a [piece, score] pair.");
                            }

                            pieces.Add((entry[0].GetString()!, entry[1].GetDouble()));
                        }
                    }
                    else if (field.NameEquals("merges") && field.Value.ValueKind == JsonValueKind.Array)
                    {
                        merges = new (string, string)[field.Value.GetArrayLength()];
                        int i = 0;
                        foreach (var merge in field.Value.EnumerateArray())
                        {
                            merges[i++] = merge.ValueKind == JsonValueKind.Array
                                ? (merge[0].GetString()!, merge[1].GetString()!)
                                : BpeTokenizer.Split(merge.GetString()!);
                        }
                    }
                    else
                    {
                        model[field.Name] = JsonNode.Parse(System.Runtime.InteropServices.JsonMarshal.GetRawUtf8Value(field.Value));
                    }
                }

                rest[property.Name] = model;
            }
            else
            {
                rest[property.Name] = JsonNode.Parse(System.Runtime.InteropServices.JsonMarshal.GetRawUtf8Value(property.Value));
            }
        }

        return new TokenizerDefinition { Source = path, Json = rest, Vocabulary = vocabulary, Pieces = pieces, Merges = merges };
    }

    // Writes a definition as a tokenizer.json (UTF-8, indented): its JSON, with the model's vocabulary (an object of token to
    // id in id order, or a list of [piece, score]) and merges ([left, right] pairs in rank order) written back in.
    internal static void Write(TokenizerDefinition definition, string path)
    {
        ArgumentNullException.ThrowIfNull(definition);
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        var options = new JsonWriterOptions { Indented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
        using var writer = new Utf8JsonWriter(stream, options);
        writer.WriteStartObject();
        foreach (var (name, value) in definition.Json)
        {
            writer.WritePropertyName(name);
            if (name != "model" || value is not JsonObject model)
            {
                if (value is null)
                {
                    writer.WriteNullValue();
                }
                else
                {
                    value.WriteTo(writer);
                }

                continue;
            }

            writer.WriteStartObject();
            foreach (var (field, fieldValue) in model)
            {
                writer.WritePropertyName(field);
                if (fieldValue is null)
                {
                    writer.WriteNullValue();
                }
                else
                {
                    fieldValue.WriteTo(writer);
                }
            }

            if (definition.Vocabulary is { } vocabulary)
            {
                writer.WriteStartObject("vocab");
                foreach (var (token, id) in vocabulary.OrderBy(p => p.Value))
                {
                    writer.WriteNumber(token, id);
                }

                writer.WriteEndObject();
            }
            else if (definition.Pieces is { } pieces)
            {
                writer.WriteStartArray("vocab");
                foreach (var (piece, score) in pieces)
                {
                    writer.WriteStartArray();
                    writer.WriteStringValue(piece);
                    writer.WriteNumberValue(score);
                    writer.WriteEndArray();
                }

                writer.WriteEndArray();
            }

            if (definition.Merges is { } merges)
            {
                writer.WriteStartArray("merges");
                foreach (var (left, right) in merges)
                {
                    writer.WriteStartArray();
                    writer.WriteStringValue(left);
                    writer.WriteStringValue(right);
                    writer.WriteEndArray();
                }

                writer.WriteEndArray();
            }

            writer.WriteEndObject();
        }

        writer.WriteEndObject();
    }

    private static int CountProperties(JsonElement element)
    {
        int count = 0;
        foreach (var _ in element.EnumerateObject())
        {
            count++;
        }

        return count;
    }
}
