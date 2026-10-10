// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.RegularExpressions;

namespace Idrak.Cli.Commands.Serve;

/// <summary>
/// A response body that passes everything through and keeps the last few kilobytes, where both APIs put a reply's
/// token counts (the OpenAI-style usage, or the chat API's final line), for the request log.
/// </summary>
internal sealed partial class TailStream(Stream inner) : Stream
{
    private const int Keep = 8192;
    private readonly byte[] _tail = new byte[Keep];
    private long _written;

    public Stream Inner { get; } = inner;

    /// <summary>The kept end of the body as text.</summary>
    public string Text
    {
        get
        {
            if (_written <= Keep)
            {
                return Encoding.UTF8.GetString(_tail, 0, (int)_written);
            }

            int start = (int)(_written % Keep);
            var bytes = new byte[Keep];
            Array.Copy(_tail, start, bytes, 0, Keep - start);
            Array.Copy(_tail, 0, bytes, Keep - start, start);
            return Encoding.UTF8.GetString(bytes);
        }
    }

    /// <summary>The prompt and generated token counts in a reply's end, when it has them.</summary>
    public static (int? Prompt, int? Generated) Tokens(string text)
    {
        static int? Last(MatchCollection matches) => matches.Count == 0 ? null : int.Parse(matches[^1].Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        return (Last(PromptCount().Matches(text)), Last(GeneratedCount().Matches(text)));
    }

    public override bool CanRead => false;

    public override bool CanSeek => false;

    public override bool CanWrite => true;

    public override long Length => throw new NotSupportedException();

    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

    public override void Flush() => Inner.Flush();

    public override Task FlushAsync(CancellationToken cancellationToken) => Inner.FlushAsync(cancellationToken);

    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

    public override void SetLength(long value) => throw new NotSupportedException();

    public override void Write(byte[] buffer, int offset, int count)
    {
        Remember(buffer.AsSpan(offset, count));
        Inner.Write(buffer, offset, count);
    }

    // Without this override Stream copies a span into a rented array before calling the array overload.
    public override void Write(ReadOnlySpan<byte> buffer)
    {
        Remember(buffer);
        Inner.Write(buffer);
    }

    public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
    {
        Remember(buffer.Span);
        await Inner.WriteAsync(buffer, cancellationToken);
    }

    public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        WriteAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

    // The last Keep bytes in the ring: at most two block copies per write (a streamed token is one or two).
    private void Remember(ReadOnlySpan<byte> data)
    {
        if (data.Length > Keep)
        {
            data = data[^Keep..];
        }

        int at = (int)(_written % Keep);
        int first = Math.Min(data.Length, Keep - at);
        data[..first].CopyTo(_tail.AsSpan(at));
        data[first..].CopyTo(_tail);
        _written += data.Length;
    }

    [GeneratedRegex("\"(?:prompt_tokens|prompt_eval_count)\"\\s*:\\s*(\\d+)")]
    private static partial Regex PromptCount();

    [GeneratedRegex("\"(?:completion_tokens|eval_count)\"\\s*:\\s*(\\d+)")]
    private static partial Regex GeneratedCount();
}
