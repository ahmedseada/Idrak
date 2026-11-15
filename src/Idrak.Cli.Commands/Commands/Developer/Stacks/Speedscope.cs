// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;

namespace Idrak.Cli.Commands.Developer.Stacks;

/// <summary>
/// A profile's moments as a speedscope file (https://www.speedscope.app, or any viewer of its format): a sampled profile a
/// thread, each moment its stack from the root with its length in milliseconds; a moment not running .NET code ends in a
/// frame naming its kind ([outside .NET code], [spinning], [stopped by the runtime], [waiting]).
/// </summary>
internal static class Speedscope
{
    private static readonly string[] Markers = ["", "[outside .NET code]", "[spinning]", "[stopped by the runtime]", "[waiting]"];

    public static void Write(StackProfile p, string path)
    {
        double ms = 1000.0 / p.Trace.Frequency;
        int markers = p.Display.Length - 1;                                         // the kinds' frames follow the methods'
        using var stream = File.Create(path);
        using var w = new Utf8JsonWriter(stream);
        w.WriteStartObject();
        w.WriteString("$schema", "https://www.speedscope.app/file-format-schema.json");
        w.WriteString("exporter", "idrak trace --stacks");
        w.WriteString("name", System.IO.Path.GetFileName(p.Trace.Path));
        w.WriteNumber("activeProfileIndex", 0);
        w.WriteStartObject("shared");
        w.WriteStartArray("frames");
        foreach (string name in p.Display)
        {
            w.WriteStartObject();
            w.WriteString("name", name);
            w.WriteEndObject();
        }

        foreach (string marker in Markers[1..])
        {
            w.WriteStartObject();
            w.WriteString("name", marker);
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteEndObject();
        w.WriteStartArray("profiles");
        foreach (var thread in p.Moments.Where(m => m.Length > 0).GroupBy(m => m.Thread).OrderByDescending(g => g.Sum(m => m.Kind is TimeKind.Waiting ? 0 : m.Length)))
        {
            var moments = thread.OrderBy(m => m.Time).ToList();
            w.WriteStartObject();
            w.WriteString("type", "sampled");
            w.WriteString("name", $"Thread {thread.Key}");
            w.WriteString("unit", "milliseconds");
            w.WriteNumber("startValue", moments[0].Time * ms);
            w.WriteNumber("endValue", (moments[^1].Time + moments[^1].Length) * ms);
            w.WriteStartArray("samples");
            foreach (var m in moments)
            {
                w.WriteStartArray();
                for (int i = m.Frames.Length - 1; i >= 0; i--)
                {
                    w.WriteNumberValue(m.Frames[i]);
                }

                if (m.Kind != TimeKind.Running)
                {
                    w.WriteNumberValue(markers + (int)m.Kind);
                }

                w.WriteEndArray();
            }

            w.WriteEndArray();
            w.WriteStartArray("weights");
            foreach (var m in moments)
            {
                w.WriteNumberValue(m.Length * ms);
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteEndObject();
    }
}
