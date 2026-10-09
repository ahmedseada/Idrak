// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Testing.Devices;

/// <summary>
/// A recorded operation call as a saved case (<see cref="RegressionCase"/> of kind "operation") and back: the
/// operation by name, each argument with its type, and the values of each storage before and after (base64 of the
/// float32 bits, so NaNs and every last bit survive).
/// </summary>
internal static class OperationCases
{
    public static RegressionCase ToCase(RecordedCall call, string deviceKind, string problem)
    {
        var storages = new JsonArray();
        for (int i = 0; i < call.Before.Length; i++)
        {
            storages.Add((JsonNode)new JsonObject
            {
                ["parameter"] = DeviceConformance.ArgumentName(call, i),
                ["length"] = call.Before[i].Length,
                ["before"] = Floats(call.Before[i]),
                ["after"] = Floats(call.After[i]),
            });
        }

        var data = new JsonObject
        {
            ["operation"] = call.Operation.Name,
            ["case"] = call.Case,
            ["failedOn"] = deviceKind,
            ["problem"] = problem,
            ["arguments"] = new JsonArray([.. call.Arguments.Select(Encode)]),
            ["storages"] = storages,
            ["result"] = call.Result is bool result ? result : null,
        };
        return new RegressionCase(Regression.OperationKind, $"{call.Operation.Name} [{call.Case}]", data);
    }

    public static RecordedCall FromCase(RegressionCase @case)
    {
        var data = @case.Data;
        string name = (string?)data["operation"] ?? throw new InvalidDataException($"{@case.Name}: no \"operation\".");
        var operation = Ops.Find(name) ?? throw new InvalidDataException($"{@case.Name}: Idrak.Abstraction has no operation {name}.");
        var arguments = data["arguments"]!.AsArray().Select(Decode).ToArray();
        var storages = data["storages"]!.AsArray().Select(s => s!.AsObject()).ToList();
        return new RecordedCall((string?)data["case"] ?? @case.Name, operation, arguments, [.. storages.Select(s => Floats((string)s["before"]!))])
        {
            After = [.. storages.Select(s => Floats((string)s["after"]!))],
            Result = data["result"] is JsonValue result ? (bool)result : null,
        };
    }

    private static string Floats(float[] values) => Convert.ToBase64String(MemoryMarshal.AsBytes(values.AsSpan()));

    private static float[] Floats(string base64) => MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(base64)).ToArray();

    private static string Text(float value) => value.ToString("R", CultureInfo.InvariantCulture);

    private static float Single(JsonNode? node) => float.Parse((string)node!, CultureInfo.InvariantCulture);

    private static JsonNode? Encode(object? value) => value switch
    {
        null => null,
        Slot s => new JsonObject { ["storage"] = s.Index },
        int i => new JsonObject { ["int"] = i },
        uint u => new JsonObject { ["uint"] = u },
        long l => new JsonObject { ["long"] = l },
        bool b => new JsonObject { ["bool"] = b },
        float f => new JsonObject { ["float"] = Text(f) },
        UnaryOp op => new JsonObject { ["UnaryOp"] = op.ToString() },
        BinaryOp op => new JsonObject { ["BinaryOp"] = op.ToString() },
        PackedFormat format => new JsonObject { ["PackedFormat"] = format.ToString() },
        GemmEpilogue epilogue => new JsonObject { ["GemmEpilogue"] = epilogue.ToString() },
        ConvActivation activation => new JsonObject { ["ConvActivation"] = activation.ToString() },
        ConvGeometry g => new JsonObject
        {
            ["ConvGeometry"] = g.Dilated
                ? new JsonArray(g.N, g.C, g.H, g.W, g.KH, g.KW, g.SH, g.SW, g.PH, g.PW, g.DH, g.DW)
                : new JsonArray(g.N, g.C, g.H, g.W, g.KH, g.KW, g.SH, g.SW, g.PH, g.PW),
        },
        AttentionVariant v => new JsonObject { ["AttentionVariant"] = new JsonArray(v.Window, Text(v.Softcap)) },
        int[] values => new JsonObject { ["ints"] = new JsonArray([.. values.Select(v => (JsonNode)v)]) },
        object?[][] tuples => new JsonObject { ["tuples"] = new JsonArray([.. tuples.Select(t => (JsonNode)new JsonArray([.. t.Select(Encode)]))]) },
        _ => throw new NotSupportedException($"An argument of type {value.GetType().Name} cannot be saved."),
    };

    private static object? Decode(JsonNode? node)
    {
        if (node is not JsonObject json || json.Count != 1)
        {
            return node is null ? null : throw new InvalidDataException($"Unreadable argument {node.ToJsonString()}.");
        }

        var (type, value) = json.First();
        return type switch
        {
            "storage" => new Slot((int)value!),
            "int" => (int)value!,
            "uint" => (uint)value!,
            "long" => (long)value!,
            "bool" => (bool)value!,
            "float" => Single(value),
            "UnaryOp" => Enum.Parse<UnaryOp>((string)value!),
            "BinaryOp" => Enum.Parse<BinaryOp>((string)value!),
            "PackedFormat" => Enum.Parse<PackedFormat>((string)value!),
            "GemmEpilogue" => Enum.Parse<GemmEpilogue>((string)value!),
            "ConvActivation" => Enum.Parse<ConvActivation>((string)value!),
            "ConvGeometry" => value!.AsArray().Select(v => (int)v!).ToArray() is var g
                ? new ConvGeometry(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9]) { DH = g.Length > 10 ? g[10] : 1, DW = g.Length > 11 ? g[11] : 1 } : default,
            "AttentionVariant" => new AttentionVariant((int)value![0]!, Single(value[1])),
            "ints" => value!.AsArray().Select(v => (int)v!).ToArray(),
            "tuples" => value!.AsArray().Select(t => t!.AsArray().Select(Decode).ToArray()).ToArray(),
            _ => throw new InvalidDataException($"Unknown argument type \"{type}\"."),
        };
    }
}
