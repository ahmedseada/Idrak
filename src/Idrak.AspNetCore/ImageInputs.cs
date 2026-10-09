// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net.Http.Headers;
using System.Text.Json.Nodes;
using Idrak.Data;

namespace Idrak.AspNetCore;

/// <summary>
/// How the chat endpoints take images (<see cref="CompletionsApiOptions.Images"/>, <see cref="ChatApiOptions.Images"/>):
/// how many one request may hold, and whether OpenAI-style <c>image_url</c> parts may name an http(s) address for the
/// server to download (off unless set: data URLs only). The size of a whole request is the host's limit (Kestrel's
/// <c>MaxRequestBodySize</c>, 30 MB unless set).
/// </summary>
public sealed class ImageInputOptions
{
    /// <summary>The most images one request may hold (8 unless set); more is a 400.</summary>
    public int MaxImages { get; set; } = 8;

    /// <summary>Whether <c>image_url</c> parts may give an http or https address the server downloads (false unless set).</summary>
    public bool AllowUrls { get; set; }

    /// <summary>The largest image the server downloads for an http(s) <c>image_url</c> (20 MB unless set); a larger one is a 400.</summary>
    public long MaxUrlBytes { get; set; } = 20L * 1024 * 1024;

    /// <summary>How long one download may take (30 seconds unless set); a slower one is a 400.</summary>
    public TimeSpan UrlTimeout { get; set; } = TimeSpan.FromSeconds(30);
}

/// <summary>The checks every chat endpoint applies to a request's images, and the downloads of http(s) image URLs.</summary>
internal static class ImageRequests
{
    private static readonly Lazy<HttpClient> Downloads = new(() => new HttpClient { Timeout = Timeout.InfiniteTimeSpan });

    /// <summary>
    /// Replaces each http(s) <c>image_url</c> of an OpenAI-style body with a data URL of the image downloaded, when
    /// <see cref="ImageInputOptions.AllowUrls"/>; otherwise refuses it. Throws <see cref="ArgumentException"/> (a 400).
    /// </summary>
    public static async Task ResolveUrls(JsonObject body, ImageInputOptions options, CancellationToken token)
    {
        if (body["messages"] is not JsonArray messages)
        {
            return;
        }

        foreach (var message in messages)
        {
            if (message?["content"] is not JsonArray parts)
            {
                continue;
            }

            foreach (var part in parts)
            {
                if (part is not JsonObject p || (string?)p["type"] != "image_url")
                {
                    continue;
                }

                string? url = CompletionsTranslation.ImageUrl(p);
                if (url is null || !(url.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || url.StartsWith("https://", StringComparison.OrdinalIgnoreCase)))
                {
                    continue;
                }

                if (!options.AllowUrls)
                {
                    throw new ArgumentException($"image_url: this server does not download images ({Shorten(url)}); send the image as a data URL "
                        + "(data:image/png;base64,...) or upload the file. The server's owner can allow http(s) image URLs.");
                }

                p["image_url"] = new JsonObject { ["url"] = (await Download(url, options, token)).ToDataUrl() };
            }
        }
    }

    private static async Task<ChatImage> Download(string url, ImageInputOptions options, CancellationToken token)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(token);
        timeout.CancelAfter(options.UrlTimeout);
        try
        {
            using var response = await Downloads.Value.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
            if (!response.IsSuccessStatusCode)
            {
                throw new ArgumentException($"image_url: {Shorten(url)} answered {(int)response.StatusCode} {response.ReasonPhrase}.");
            }

            if (response.Content.Headers.ContentLength > options.MaxUrlBytes)
            {
                throw new ArgumentException($"image_url: {Shorten(url)} holds {response.Content.Headers.ContentLength} bytes, more than the {options.MaxUrlBytes} this server downloads.");
            }

            await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token);
            using var data = new MemoryStream();
            var buffer = new byte[81920];
            int read;
            while ((read = await stream.ReadAsync(buffer, timeout.Token)) > 0)
            {
                if (data.Length + read > options.MaxUrlBytes)
                {
                    throw new ArgumentException($"image_url: {Shorten(url)} holds more than the {options.MaxUrlBytes} bytes this server downloads.");
                }

                data.Write(buffer, 0, read);
            }

            if (data.Length == 0)
            {
                throw new ArgumentException($"image_url: {Shorten(url)} answered no bytes.");
            }

            return ChatImage.FromBytes(data.ToArray(), MediaType(response.Content.Headers.ContentType));
        }
        catch (OperationCanceledException) when (!token.IsCancellationRequested)
        {
            throw new ArgumentException($"image_url: {Shorten(url)} did not answer within {options.UrlTimeout.TotalSeconds:0.#} s.");
        }
        catch (HttpRequestException ex)
        {
            throw new ArgumentException($"image_url: {Shorten(url)} could not be downloaded ({ex.Message}).");
        }
    }

    // The download's media type when it names an image; else the bytes' own signature decides.
    private static string? MediaType(MediaTypeHeaderValue? type) =>
        type?.MediaType is { } media && media.StartsWith("image/", StringComparison.OrdinalIgnoreCase) ? media : null;

    /// <summary>
    /// The request with its images checked: no more than <see cref="ImageInputOptions.MaxImages"/>, each in a format a
    /// registered codec reads (PNG, JPEG, BMP, PPM/PGM by default); with <paramref name="grayscale"/>, each turned upright
    /// and grey (<see cref="ChatImageDecoder.Grayscale"/>). Throws <see cref="ArgumentException"/> (a 400).
    /// </summary>
    public static ChatRequest Check(ChatRequest request, ImageInputOptions options, bool grayscale)
    {
        int count = 0;
        foreach (var message in request.Messages)
        {
            foreach (var part in message.Parts)
            {
                if (part is not ChatImage image)
                {
                    continue;
                }

                if (++count > options.MaxImages)
                {
                    throw new ArgumentException($"A request may hold {options.MaxImages} image{(options.MaxImages == 1 ? "" : "s")} at most on this server.");
                }

                if (ChatImageDecoder.FormatOf(image) is null)
                {
                    throw new ArgumentException($"Image {count} ({image.MediaType ?? "unknown type"}, {image.Data.Length} bytes) is not in a format this server reads; "
                        + "send PNG, JPEG, BMP or PPM/PGM (or a format an image codec the server registered reads).");
                }
            }
        }

        if (!grayscale || count == 0)
        {
            return request;
        }

        var messages = new List<ChatMessage>(request.Messages.Count);
        count = 0;
        foreach (var message in request.Messages)
        {
            if (!message.Parts.Any(p => p is ChatImage))
            {
                messages.Add(message);
                continue;
            }

            var parts = new List<ChatPart>(message.Parts.Count);
            foreach (var part in message.Parts)
            {
                if (part is ChatImage image)
                {
                    count++;
                    try
                    {
                        parts.Add(ChatImageDecoder.Grayscale(image));
                    }
                    catch (Exception ex) when (ex is InvalidDataException or NotSupportedException)
                    {
                        throw new ArgumentException($"Image {count} ({image.MediaType ?? "unknown type"}) does not decode: {ex.Message}");
                    }
                }
                else
                {
                    parts.Add(part);
                }
            }

            messages.Add(message with { Parts = parts });
        }

        return request with { Messages = messages };
    }

    private static string Shorten(string text) => text.Length <= 80 ? text : text[..80] + "…";
}
