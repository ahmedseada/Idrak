// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Text.Json.Nodes;
using Idrak.Imaging;

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

    /// <summary>
    /// Whether an http(s) <c>image_url</c> may reach a loopback, link-local, private (RFC 1918, carrier-grade NAT, unique
    /// local IPv6), multicast, unspecified or reserved address (false unless set): off, the server resolves each address
    /// (and each redirect's) and connects only to the public ones it checked, so a client cannot make it read its own
    /// network (a 400 naming the URL and the address).
    /// </summary>
    public bool AllowPrivateUrls { get; set; }
}

/// <summary>The checks every chat endpoint applies to a request's images, and the downloads of http(s) image URLs.</summary>
internal static class ImageRequests
{
    // One client for public addresses only and one for any (ImageInputOptions.AllowPrivateUrls). Redirects are followed
    // here, each hop's host checked; the connection itself goes to an address checked when it is made, so a name that
    // resolves to another address the second time (DNS rebinding) still cannot reach a private one.
    private static readonly Lazy<HttpClient> PublicDownloads = new(() => Client(allowPrivate: false));
    private static readonly Lazy<HttpClient> AnyDownloads = new(() => Client(allowPrivate: true));

    // Redirects one download follows at most.
    private const int MaxRedirects = 5;

    private static HttpClient Client(bool allowPrivate) => new(new SocketsHttpHandler
    {
        AllowAutoRedirect = false,
        ConnectCallback = (context, token) => Connect(context, allowPrivate, token),
    })
    { Timeout = Timeout.InfiniteTimeSpan };

    // Connects to the host's first reachable address; refuses a private one unless allowed. Through a proxy (the
    // endpoint is the proxy, not the request's host) the proxy is connected to as it is: the request's host was checked
    // before the request (CheckHost).
    private static async ValueTask<Stream> Connect(SocketsHttpConnectionContext context, bool allowPrivate, CancellationToken token)
    {
        var endpoint = context.DnsEndPoint;
        string? requested = context.InitialRequestMessage.RequestUri?.IdnHost.Trim('[', ']');
        bool proxied = requested is not null && !string.Equals(requested, endpoint.Host.Trim('[', ']'), StringComparison.OrdinalIgnoreCase);
        var addresses = await Resolve(endpoint.Host, token);
        if (!proxied && !allowPrivate && addresses.FirstOrDefault(a => PrivateKind(a) is not null) is { } refused)
        {
            throw new PrivateAddressException(refused);
        }

        var socket = new Socket(SocketType.Stream, ProtocolType.Tcp) { NoDelay = true };
        try
        {
            await socket.ConnectAsync(addresses, endpoint.Port, token);
            return new NetworkStream(socket, ownsSocket: true);
        }
        catch
        {
            socket.Dispose();
            throw;
        }
    }

    private static async ValueTask<IPAddress[]> Resolve(string host, CancellationToken token) =>
        IPAddress.TryParse(host.Trim('[', ']'), out var literal) ? [literal] : await Dns.GetHostAddressesAsync(host, token);

    /// <summary>
    /// What kind of address <paramref name="address"/> is when it is not a public one (loopback, link-local, private,
    /// carrier-grade NAT, multicast, unspecified, reserved, an IPv6 site-local or unique local one), or null when public.
    /// An IPv4 address mapped into IPv6 is judged as IPv4.
    /// </summary>
    internal static string? PrivateKind(IPAddress address)
    {
        if (address.IsIPv4MappedToIPv6)
        {
            address = address.MapToIPv4();
        }

        if (address.AddressFamily == AddressFamily.InterNetwork)
        {
            Span<byte> b = stackalloc byte[4];
            address.TryWriteBytes(b, out _);
            return b[0] switch
            {
                0 => "unspecified",
                10 => "private",
                127 => "loopback",
                100 when (b[1] & 0xC0) == 64 => "carrier-grade NAT",
                169 when b[1] == 254 => "link-local",
                172 when (b[1] & 0xF0) == 16 => "private",
                192 when b[1] == 168 => "private",
                192 when b[1] == 0 && b[2] == 0 => "reserved",
                198 when (b[1] & 0xFE) == 18 => "reserved",
                >= 224 and <= 239 => "multicast",
                >= 240 => "reserved",
                _ => null,
            };
        }

        if (address.Equals(IPAddress.IPv6Any) || address.Equals(IPAddress.IPv6None))
        {
            return "unspecified";
        }

        return IPAddress.IsLoopback(address) ? "loopback"
            : address.IsIPv6LinkLocal ? "link-local"
            : address.IsIPv6SiteLocal || address.IsIPv6UniqueLocal ? "private"
            : address.IsIPv6Multicast ? "multicast"
            : null;
    }

    // A connection refused because its address is not public.
    private sealed class PrivateAddressException(IPAddress address) : IOException($"{address} is a {PrivateKind(address)} address")
    {
        public IPAddress Address { get; } = address;
    }

    // The request's host (each hop's): an http(s) address whose every resolved address is public, unless allowed.
    private static async Task CheckHost(Uri uri, string url, ImageInputOptions options, CancellationToken token)
    {
        string hop = uri.OriginalString == url ? "" : $" (redirected to {Shorten(uri.OriginalString)})";
        if (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps)
        {
            throw new ArgumentException($"image_url: {Shorten(url)}{hop} is not an http(s) address.");
        }

        if (options.AllowPrivateUrls)
        {
            return;
        }

        IPAddress[] addresses;
        try
        {
            addresses = await Resolve(uri.IdnHost, token);
        }
        catch (SocketException e)
        {
            throw new ArgumentException($"image_url: {Shorten(url)}{hop}: the host {uri.IdnHost} does not resolve ({e.SocketErrorCode}).");
        }

        if (addresses.FirstOrDefault(a => PrivateKind(a) is not null) is { } refused)
        {
            throw new ArgumentException(Refusal(url, hop, refused));
        }
    }

    private static string Refusal(string url, string hop, IPAddress address) =>
        $"image_url: {Shorten(url)}{hop} reaches {address}, a {PrivateKind(address)} address, which this server does not download from "
        + "(loopback, link-local, private, multicast and reserved addresses are refused). The server's owner can allow them "
        + "(ImageInputOptions.AllowPrivateUrls; idrak serve --allow-private-image-urls).";

    // The answer of the URL after its redirects (at most MaxRedirects, each hop's host checked).
    private static async Task<HttpResponseMessage> Fetch(string url, ImageInputOptions options, CancellationToken token)
    {
        var client = (options.AllowPrivateUrls ? AnyDownloads : PublicDownloads).Value;
        var current = new Uri(url);
        for (int hop = 0; ; hop++)
        {
            await CheckHost(current, url, options, token);
            HttpResponseMessage response;
            try
            {
                response = await client.GetAsync(current, HttpCompletionOption.ResponseHeadersRead, token);
            }
            catch (HttpRequestException e) when (e.InnerException is PrivateAddressException refused)
            {
                throw new ArgumentException(Refusal(url, current.OriginalString == url ? "" : $" (redirected to {Shorten(current.OriginalString)})", refused.Address));
            }

            if ((int)response.StatusCode is 301 or 302 or 303 or 307 or 308 && response.Headers.Location is { } location)
            {
                response.Dispose();
                if (hop >= MaxRedirects)
                {
                    throw new ArgumentException($"image_url: {Shorten(url)} redirects more than {MaxRedirects} times.");
                }

                current = location.IsAbsoluteUri ? location : new Uri(current, location);
                continue;
            }

            return response;
        }
    }

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
            using var response = await Fetch(url, options, timeout.Token);
            if (!response.IsSuccessStatusCode)
            {
                // The status alone: the remote reason phrase and body are not repeated to the client.
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"image_url: {Shorten(url)} answered HTTP {(int)response.StatusCode}."));
            }

            if (response.Content.Headers.ContentLength > options.MaxUrlBytes)
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                    $"image_url: {Shorten(url)} holds {response.Content.Headers.ContentLength} bytes, more than the {options.MaxUrlBytes} this server downloads."));
            }

            await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token);

            // Sized by the announced length when there is one (its bytes are then not copied again at the end); the read
            // buffer is pooled and returned in finally.
            using var data = new MemoryStream(response.Content.Headers.ContentLength is long announced && announced > 0 ? (int)Math.Min(announced, Array.MaxLength) : 0);
            byte[] buffer = ArrayPool<byte>.Shared.Rent(81920);
            try
            {
                int read;
                while ((read = await stream.ReadAsync(buffer, timeout.Token)) > 0)
                {
                    if (data.Length + read > options.MaxUrlBytes)
                    {
                        throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                            $"image_url: {Shorten(url)} holds more than the {options.MaxUrlBytes} bytes this server downloads."));
                    }

                    data.Write(buffer, 0, read);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }

            if (data.Length == 0)
            {
                throw new ArgumentException($"image_url: {Shorten(url)} answered no bytes.");
            }

            byte[] bytes = data.Length == data.Capacity ? data.GetBuffer() : data.ToArray();
            return ChatImage.FromBytes(bytes, MediaType(response.Content.Headers.ContentType));
        }
        catch (OperationCanceledException) when (!token.IsCancellationRequested)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"image_url: {Shorten(url)} did not answer within {options.UrlTimeout.TotalSeconds:0.#} s."));
        }
        catch (HttpRequestException ex)
        {
            // The kind of failure, not the message (which may carry what the remote end sent).
            throw new ArgumentException($"image_url: {Shorten(url)} could not be downloaded ({ex.HttpRequestError}).");
        }
        catch (UriFormatException)
        {
            throw new ArgumentException($"image_url: {Shorten(url)} is not a valid address.");
        }
    }

    // The download's media type when it names an image; else the bytes' own signature decides.
    private static string? MediaType(MediaTypeHeaderValue? type) =>
        type?.MediaType is { } media && media.StartsWith("image/", StringComparison.OrdinalIgnoreCase) ? media : null;

    /// <summary>
    /// The request with the vision options a client sent (<c>"vision_options"</c>: a JSON object, or its text in a form
    /// field) for the model's vision family (<see cref="ChatRequest.VisionOptions"/>; the family checks the keys and
    /// values when it reads the images: an unknown key is a 400 naming those it takes). Absent or null: the request as it is.
    /// </summary>
    /// <exception cref="ArgumentException">The value is not a JSON object of strings, numbers and booleans (a 400).</exception>
    public static ChatRequest WithVisionOptions(ChatRequest request, JsonNode? options)
    {
        ArgumentNullException.ThrowIfNull(request);
        try
        {
            var parsed = options switch
            {
                null => null,
                JsonObject json => VisionOptions.FromJson(json),
                JsonValue text when text.TryGetValue(out string? s) => string.IsNullOrWhiteSpace(s) ? null
                    : JsonNode.Parse(s) as JsonObject is { } json ? VisionOptions.FromJson(json) : throw new FormatException("it is not a JSON object"),
                _ => throw new FormatException("it is not a JSON object"),
            };
            return parsed is null || parsed.Count == 0 ? request : request with { VisionOptions = parsed };
        }
        catch (Exception ex) when (ex is FormatException or System.Text.Json.JsonException)
        {
            throw new ArgumentException($"vision_options: {ex.Message} (give an object such as {{\"KEY\": true}}; the model's vision family names the keys it takes).");
        }
    }

    /// <summary>
    /// The request with the image transforms a client sent (<c>"image_transforms"</c>: a pipeline's text such as
    /// <c>"grayscale,max_width=1024,contrast=1.5"</c>, or a JSON array of steps; in a form field, either as text) for this
    /// request's images (<see cref="ChatRequest.ImageTransforms"/>, run in the order given, replacing the server's own;
    /// <c>"none"</c>: none). Absent, null or an empty field: the request as it is (the server's transforms).
    /// </summary>
    /// <exception cref="ArgumentException">An unknown transform or option, or a bad value: the message names the registered transforms (a 400).</exception>
    public static ChatRequest WithImageTransforms(ChatRequest request, JsonNode? transforms)
    {
        ArgumentNullException.ThrowIfNull(request);
        try
        {
            var parsed = transforms switch
            {
                null => null,
                JsonValue text when text.TryGetValue(out string? s) => string.IsNullOrWhiteSpace(s) ? null
                    : s.TrimStart().StartsWith('[') ? ImageTransformPipeline.FromJson(JsonNode.Parse(s)) : ImageTransformPipeline.Parse(s),
                _ => ImageTransformPipeline.FromJson(transforms),
            };
            return parsed is null ? request : request with { ImageTransforms = parsed };
        }
        catch (Exception ex) when (ex is ArgumentException or System.Text.Json.JsonException or FormatException)
        {
            throw new ArgumentException($"image_transforms: {ex.Message}");
        }
    }

    /// <summary>
    /// The request with its images checked: no more than <see cref="ImageInputOptions.MaxImages"/>, each in a format a
    /// registered codec reads (PNG, JPEG, BMP, PPM/PGM by default); with <paramref name="grayscale"/>, each turned upright
    /// and grey (<see cref="ChatImageDecoder.Grayscale(ChatImage)"/>). Throws <see cref="ArgumentException"/> (a 400).
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
