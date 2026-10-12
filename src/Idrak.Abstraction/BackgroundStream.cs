// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// Runs a synchronous stream on a thread-pool thread and hands its items to an asynchronous reader: the asynchronous
/// form of any generator that produces its output step by step (text, speech, frames).
/// </summary>
public static class BackgroundStream
{
    /// <summary>
    /// The items of <paramref name="source"/> (started on a thread-pool thread with a token cancelled when the reader stops
    /// or <paramref name="cancellationToken"/> is cancelled), as they come; the source's exception reaches the reader.
    /// </summary>
    public static async IAsyncEnumerable<T> Run<T>(Func<CancellationToken, IEnumerable<T>> source,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var channel = System.Threading.Channels.Channel.CreateUnbounded<T>(
            new System.Threading.Channels.UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        var producer = Task.Run(() =>
        {
            try
            {
                foreach (var item in source(stop.Token))
                {
                    channel.Writer.TryWrite(item);
                }

                channel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                channel.Writer.TryComplete(ex);
            }
        }, CancellationToken.None);

        try
        {
            await foreach (var item in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return item;
            }
        }
        finally
        {
            stop.Cancel();
            await producer.ConfigureAwait(false);
        }
    }
}
