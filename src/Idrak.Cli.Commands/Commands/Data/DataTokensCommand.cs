// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Datasets;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data tokens FILE... --tokenizer tokenizer.json -o tokens.bin</c>: text encoded into a token file for pretraining
/// (<see cref="TokenFileWriter"/>; read by the "tokens" source of <c>idrak pretrain</c>).
/// </summary>
internal sealed class DataTokensCommand : Command
{
    public override string Name => "data tokens";

    public override string Summary => "Encode text files into a token file for pretraining (each document's ids, then a separator)";

    public override string Usage => """
        FILE... --tokenizer FILE -o TOKENS.bin [options]

        Encodes text files with a tokenizer into one token file: each document's ids, then the separator, little-endian,
        in the files' order, and a .json beside it with the count, the id type and the tokenizer (readers take the type
        from it). idrak pretrain reads it as a "tokens" source, cut into windows of its maximum length.

        Arguments:
          FILE  text files (UTF-8), folders of .txt files, or patterns (corpus/*.txt)

        Options:
              --tokenizer FILE   the tokenizer.json (or a model folder holding one)
              --documents D      blocks (documents between lines of ---, as idrak data text writes, the default), lines
                                 (each line a document) or files (each file a document)
              --separator TOKEN  the token written after every document (</s> when the tokenizer has it; none to write none)
              --type T           the integer type ids are stored in: auto (IDRAK_ID_DTYPE, else the smallest that holds
                                 the tokenizer's ids: uint8 up to 256, uint16 up to 65,536, ...; the default) or any
                                 registered integer type of whole bytes (uint16, int32, uint32, ...; u16 for short)
              --workers N        threads encoding documents (one per processor)
          -o, --out FILE         the token file to write

        Examples:
          idrak data tokens corpus.txt --tokenizer tokenizer.json -o tokens.bin
          idrak data tokens books --documents files --tokenizer tokenizer.json --type u32 -o books.bin
          idrak pretrain model -o run --tokens tokens.bin --steps 2000
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--tokenizer", "--documents", "--separator", "--type", "--workers", "--out"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out FILE for the token file to write.");
        string tokenizerPath = context.Option("--tokenizer") ?? throw new UsageException("Give --tokenizer FILE (a tokenizer.json or a model folder).");
        var files = TextFiles.Expand(context.Positional);
        if (files.Count == 0)
        {
            throw new UsageException("Give the text files to encode.");
        }

        var tokenizer = PretrainedTokenizers.TryLoad(tokenizerPath) ?? throw new UsageException($"No tokenizer at {tokenizerPath} (readers: {string.Join(", ", TokenizerFiles.Names)}).");
        var type = ParseType("--type", context.Option("--type"));
        var documents = (context.Option("--documents") ?? "blocks").ToLowerInvariant() switch
        {
            "blocks" => TokenDocuments.Blocks,
            "lines" => TokenDocuments.Lines,
            "files" => TokenDocuments.Files,
            var d => throw new UsageException($"--documents is blocks, lines or files, not '{d}'."),
        };
        int? separator = context.Option("--separator") switch
        {
            "none" => null,
            null => tokenizer.IdOf("</s>"),
            var token => tokenizer.IdOf(token) ?? throw new UsageException($"The tokenizer has no token '{token}'."),
        };
        var options = new TokenFileWriterOptions { Type = type, Documents = documents, Separator = separator, Workers = context.IntOption("--workers", 0) };
        long total = files.Sum(f => new FileInfo(f).Length);
        var watch = Stopwatch.StartNew();
        TokenFileResult result;
        using (var line = new ProgressLine(context, "encoding", total: total, unit: ProgressUnit.Bytes))
        {
            try
            {
                result = TokenFileWriter.Write(output, files, tokenizer, options, line);
            }
            catch (ArgumentException e)
            {
                throw new UsageException(e.Message);
            }

            line.Finish();
        }

        // The library's record (the format, the id type, the counts) with what the command knows: the tokenizer and the files.
        var described = JsonNode.Parse(File.ReadAllText(result.Record)) as JsonObject ?? throw new InvalidDataException($"{result.Record} is not a JSON object.");
        described["tokenizer"] = Path.GetFullPath(tokenizerPath);
        described["vocabulary_fingerprint"] = Idrak.Text.Abstractions.DistillationTeacher.VocabularyFingerprint(tokenizer);
        described["files"] = new JsonArray([.. files.Select(f => (JsonNode)Path.GetFullPath(f))]);
        File.WriteAllText(result.Record, described.ToJsonString(CommandContext.JsonOutput) + "\n");
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{result.Tokens:N0} tokens of {result.Documents:N0} documents ({total / 1048576.0:F1} MB of text) in {watch.Elapsed.TotalSeconds:F1} s, written to {result.Path}"));
        context.WriteJson(new JsonObject
        {
            ["path"] = result.Path, ["tokens"] = result.Tokens, ["documents"] = result.Documents, ["bytes"] = result.Bytes, ["type"] = result.Type.Name,
            ["separator"] = separator, ["seconds"] = Math.Round(watch.Elapsed.TotalSeconds, 3),
        });
        return ExitCodes.Ok;
    }

    /// <summary>A token type option: null (not given) or auto for the file's own choice, else a registered integer type (u16 for short).</summary>
    internal static Idrak.Abstraction.ElementType? ParseType(string option, string? text)
    {
        if (text is null || text.Trim().Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        try
        {
            return TokenFileRecord.Parse(text);
        }
        catch (ArgumentException e)
        {
            throw new UsageException($"{option}: {e.Message} Or auto.");
        }
    }
}
