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
        in the files' order (a .json beside it says the count, the width and the tokenizer). idrak pretrain reads it as a
        "tokens" source, cut into windows of its maximum length.

        Arguments:
          FILE  text files (UTF-8), folders of .txt files, or patterns (corpus/*.txt)

        Options:
              --tokenizer FILE   the tokenizer.json (or a model folder holding one)
              --documents D      blocks (documents between lines of ---, as idrak data text writes, the default), lines
                                 (each line a document) or files (each file a document)
              --separator TOKEN  the token written after every document (</s> when the tokenizer has it; none to write none)
              --type T           u16 (ids up to 65,535, the default), i32 or u32
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
        var type = (context.Option("--type") ?? "u16").ToLowerInvariant() switch
        {
            "u16" or "uint16" => TokenType.UInt16,
            "i32" or "int32" => TokenType.Int32,
            "u32" or "uint32" => TokenType.UInt32,
            var t => throw new UsageException($"--type is u16, i32 or u32, not '{t}'."),
        };
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

        var described = new JsonObject
        {
            ["format"] = "idrak-tokens/1", ["tokens"] = result.Tokens, ["documents"] = result.Documents, ["type"] = type.ToString(),
            ["separator"] = separator, ["tokenizer"] = Path.GetFullPath(tokenizerPath), ["vocabulary"] = tokenizer.VocabularySize,
            ["vocabulary_fingerprint"] = Idrak.Nlp.Abstractions.DistillationTeacher.VocabularyFingerprint(tokenizer),
            ["files"] = new JsonArray([.. files.Select(f => (JsonNode)Path.GetFullPath(f))]),
        };
        File.WriteAllText(output + ".json", described.ToJsonString(CommandContext.JsonOutput) + "\n");
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{result.Tokens:N0} tokens of {result.Documents:N0} documents ({total / 1048576.0:F1} MB of text) in {watch.Elapsed.TotalSeconds:F1} s, written to {result.Path}"));
        context.WriteJson(new JsonObject
        {
            ["path"] = result.Path, ["tokens"] = result.Tokens, ["documents"] = result.Documents, ["bytes"] = result.Bytes, ["type"] = type.ToString(),
            ["separator"] = separator, ["seconds"] = Math.Round(watch.Elapsed.TotalSeconds, 3),
        });
        return ExitCodes.Ok;
    }
}
