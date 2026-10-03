// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Design;

/// <summary><c>idrak viz network.json</c>: the network as a text diagram, a Mermaid flowchart or an SVG picture.</summary>
internal sealed class VizCommand : Command
{
    public override string Name => "viz";

    public override string Summary => "Draw a network.json as text, Mermaid or SVG";

    public override string Usage =>
        "NETWORK.json [--format text|mermaid|svg]\n\n" +
        "Prints the layers top to bottom with their output shapes and parameters: as a text diagram (default), a Mermaid\n" +
        "flowchart (paste into Markdown) or an SVG picture (redirect to a file).\n\n" +
        "Options:\n" +
        "      --format NAME   text (default), mermaid or svg\n\n" +
        "Examples:\n" +
        "  idrak viz ./run/network.json\n" +
        "  idrak viz network.json --format svg > network.svg\n\n" +
        "Environment: IDRAK_CONFIG, IDRAK_TRACE.";

    public override IReadOnlyCollection<string> ValueOptions => ["--format"];

    public override int Run(CommandContext context)
    {
        var (path, description) = ExplainCommand.Read(context.Argument(0, "NETWORK.json"));
        var analysis = NetworkAnalysis.Of(description);
        string format = context.Option("--format")?.ToLowerInvariant() ?? "text";
        var nodes = new List<(string Title, string Detail)> { ($"input ({analysis.Kind})", NetworkAnalysis.Shape(analysis.Input)) };
        nodes.AddRange(analysis.Layers.Select(l => (l.Description, NetworkAnalysis.Shape(l.Output) + (l.Parameters > 0 ? $", {DeviceMemory.Count(l.Parameters)} parameters" : ""))));
        string diagram = format switch
        {
            "text" => Text(nodes, analysis),
            "mermaid" => Mermaid(nodes),
            "svg" => Svg(nodes, (string?)description["name"] ?? Path.GetFileNameWithoutExtension(path)),
            _ => throw new UsageException($"--format {format}: use text, mermaid or svg."),
        };
        if (!context.Quiet && !context.Json)
        {
            context.Output.Write(diagram);
        }

        context.WriteJson(new JsonObject
        {
            ["path"] = path,
            ["format"] = format,
            ["nodes"] = new JsonArray([.. nodes.Select(n => (JsonNode)new JsonObject { ["title"] = n.Title, ["detail"] = n.Detail })]),
            ["diagram"] = diagram,
        });
        return ExitCodes.Ok;
    }

    private static string Text(List<(string Title, string Detail)> nodes, NetworkAnalysis analysis)
    {
        int width = nodes.Max(n => n.Title.Length);
        var sb = new StringBuilder();
        for (int i = 0; i < nodes.Count; i++)
        {
            if (i > 0)
            {
                sb.Append("  |\n");
            }

            sb.Append(nodes[i].Title.PadRight(width)).Append("  -> ").Append(nodes[i].Detail).Append('\n');
        }

        return sb.Append($"\n{analysis.Parameters:N0} parameters, {DeviceMemory.Count(analysis.Flops)} FLOPs per sample\n").ToString();
    }

    private static string Mermaid(List<(string Title, string Detail)> nodes)
    {
        var sb = new StringBuilder("flowchart TD\n");
        for (int i = 0; i < nodes.Count; i++)
        {
            string label = $"{nodes[i].Title}<br/>{nodes[i].Detail}".Replace("\"", "'", StringComparison.Ordinal);
            sb.Append($"  n{i}[\"{label}\"]\n");
        }

        for (int i = 1; i < nodes.Count; i++)
        {
            sb.Append($"  n{i - 1} --> n{i}\n");
        }

        return sb.ToString();
    }

    // Boxes in a column joined by arrows; plain SVG 1.1 with a white background, readable in any viewer.
    private static string Svg(List<(string Title, string Detail)> nodes, string title)
    {
        const int BoxHeight = 44, Gap = 22, Top = 40;
        int width = Math.Max(260, 16 + 7 * nodes.Max(n => Math.Max(n.Title.Length, n.Detail.Length)));
        int height = Top + nodes.Count * (BoxHeight + Gap);
        var sb = new StringBuilder();
        sb.Append($"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width + 40}\" height=\"{height}\" viewBox=\"0 0 {width + 40} {height}\" font-family=\"sans-serif\">\n");
        sb.Append($"  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n");
        sb.Append($"  <text x=\"20\" y=\"24\" font-size=\"14\" font-weight=\"bold\" fill=\"#222222\">{WebUtility.HtmlEncode(title)}</text>\n");
        sb.Append("  <defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"4\" refY=\"4\" orient=\"auto\"><path d=\"M0,0 L8,4 L0,8 z\" fill=\"#555555\"/></marker></defs>\n");
        for (int i = 0; i < nodes.Count; i++)
        {
            int y = Top + i * (BoxHeight + Gap);
            sb.Append($"  <rect x=\"20\" y=\"{y}\" width=\"{width}\" height=\"{BoxHeight}\" rx=\"6\" fill=\"#eef3fb\" stroke=\"#4a6fa5\"/>\n");
            sb.Append($"  <text x=\"30\" y=\"{y + 18}\" font-size=\"13\" fill=\"#222222\">{WebUtility.HtmlEncode(nodes[i].Title)}</text>\n");
            sb.Append($"  <text x=\"30\" y=\"{y + 35}\" font-size=\"11\" fill=\"#555555\">{WebUtility.HtmlEncode(nodes[i].Detail)}</text>\n");
            if (i + 1 < nodes.Count)
            {
                int x = 20 + width / 2;
                sb.Append($"  <line x1=\"{x}\" y1=\"{y + BoxHeight}\" x2=\"{x}\" y2=\"{y + BoxHeight + Gap - 4}\" stroke=\"#555555\" marker-end=\"url(#arrow)\"/>\n");
            }
        }

        return sb.Append("</svg>\n").ToString();
    }
}
