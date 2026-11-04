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
    /// (and each redirect's) once and connects only to a public one it checked, directly or through a tunnel it opens
    /// to that address across the proxy, so a client cannot make it read its own network (a 400 naming the URL and the
    /// address). Through a proxy only https addresses are downloaded then: a plain http request would let the proxy look
    /// the name up again.
    /// </summary>
    public bool AllowPrivateUrls { get; set; }

    /// <summary>
    /// Whether every image URL is refused while a proxy would carry it (false unless set): for a server that must never
    /// fetch through the proxy it is given, whatever the address. Off, an https download goes through a tunnel the server
    /// opens to the address it checked (<see cref="AllowPrivateUrls"/>).
    /// </summary>
    public bool RefuseThroughProxy { get; set; }

    /// <summary>The proxy downloads go through (null unless set: the system's, <see cref="HttpClient.DefaultProxy"/>).</summary>
    public IWebProxy? Proxy { get; set; }

    /// <summary>
    /// How a host name is looked up (null unless set: the system's DNS, <see cref="Dns.GetHostAddressesAsync(string, CancellationToken)"/>),
    /// for an app with its own resolver. Each download looks its host up once and connects to the address checked.
    /// </summary>
    public Func<string, CancellationToken, Task<IPAddress[]>>? Resolver { get; set; }
}

/// <summary>The checks every chat endpoint applies to a request's images, and the downloads of http(s) image URLs.</summary>
internal static class ImageRequests
{
    // Downloads held to the addresses checked (AllowPrivateUrls off): no proxy of the handler's own, each connection made by
    // Connect to an address CheckHost checked, directly or through a tunnel opened here across the proxy. Redirects are
    // followed here, each hop checked again.
    private static readonly Lazy<HttpClient> PinnedDownloads = new(() => new HttpClient(new SocketsHttpHandler
    {
        AllowAutoRedirect = false,
        UseProxy = false,
        ConnectCallback = Connect,
    })
    { Timeout = Timeout.InfiniteTimeSpan });

    // Downloads that may reach any address (AllowPrivateUrls): through the proxy the options name, else the system's.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<IWebProxy, HttpClient> ProxiedDownloads = [];
    private static readonly Lazy<HttpClient> AnyDownloads = new(() => new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = false })
    { Timeout = Timeout.InfiniteTimeSpan });

    // Redirects one download follows at most.
    private const int MaxRedirects = 5;

    // The addresses checked for a request, and the proxy whose tunnel reaches them (null: directly).
    private sealed record Pin(IPAddress[] Addresses, Uri? Proxy, ICredentials? Credentials);

    private static readonly HttpRequestOptionsKey<Pin> PinKey = new("idrak.image-url.pin");

    // Connects to an address the request's pin holds (checked when the host was looked up, never looked up again): directly,
    // or through a tunnel across the proxy (CONNECT to the checked address; TLS then runs over it with the URL's host name,
    // so the certificate is still checked against that name).
    private static async ValueTask<Stream> Connect(SocketsHttpConnectionContext context, CancellationToken token)
    {
        if (!context.InitialRequestMessage.Options.TryGetValue(PinKey, out var pin))
        {
            throw new InvalidOperationException("An image download without the address it was checked for.");
        }

        int port = context.DnsEndPoint.Port;
        if (pin.Proxy is not { } proxy)
        {
            return await Open(pin.Addresses, port, token);
        }

        var stream = await Open(await Dns.GetHostAddressesAsync(proxy.IdnHost, token), proxy.Port, token);
        try
        {
            await Tunnel(stream, proxy, pin.Credentials, pin.Addresses[0], port, token);
            return stream;
        }
        catch
        {
            await stream.DisposeAsync();
            throw;
        }
    }

    private static async Task<Stream> Open(IPAddress[] addresses, int port, CancellationToken token)
    {
        var socket = new Socket(SocketType.Stream, ProtocolType.Tcp) { NoDelay = true };
        try
        {
            await socket.ConnectAsync(addresses, port, token);
            return new NetworkStream(socket, ownsSocket: true);
        }
        catch
        {
            socket.Dispose();
            throw;
        }
    }

    // Asks the proxy for a tunnel to address:port (HTTP CONNECT) and reads its answer's head; anything but 2xx is refused.
    private static async Task Tunnel(Stream stream, Uri proxy, ICredentials? credentials, IPAddress address, int port, CancellationToken token)
    {
        string target = address.AddressFamily == AddressFamily.InterNetworkV6 ? $"[{address}]:{port.ToString(CultureInfo.InvariantCulture)}"
            : $"{address}:{port.ToString(CultureInfo.InvariantCulture)}";
        var head = new System.Text.StringBuilder().Append("CONNECT ").Append(target).Append(" HTTP/1.1\r\nHost: ").Append(target).Append("\r\n");
        if (Basic(proxy, credentials) is { } authorization)
        {
            head.Append("Proxy-Authorization: Basic ").Append(authorization).Append("\r\n");
        }

        byte[] request = System.Text.Encoding.ASCII.GetBytes(head.Append("\r\n").ToString());
        await stream.WriteAsync(request, token);

        // The answer's head, byte by byte up to its blank line: nothing after it is read, so TLS starts on a clean stream.
        var answer = new List<byte>(128);
        var one = new byte[1];
        while (answer.Count < 16 * 1024 && !(answer.Count >= 4 && answer[^4] == '\r' && answer[^3] == '\n' && answer[^2] == '\r' && answer[^1] == '\n'))
        {
            if (await stream.ReadAsync(one, token) == 0)
            {
                throw new ProxyTunnelException(proxy, target, "closed the connection");
            }

            answer.Add(one[0]);
        }

        string status = System.Text.Encoding.ASCII.GetString([.. answer]).Split("\r\n")[0];
        var parts = status.Split(' ', 3);
        if (parts.Length < 2 || !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int code))
        {
            throw new ProxyTunnelException(proxy, target, "did not answer as an HTTP proxy");
        }

        if (code is < 200 or > 299)
        {
            throw new ProxyTunnelException(proxy, target, string.Create(CultureInfo.InvariantCulture, $"answered HTTP {code}"));
        }
    }

    // The proxy's user and password (in its address, or the proxy's credentials) for Basic authentication, or null.
    private static string? Basic(Uri proxy, ICredentials? credentials)
    {
        string? user = null, password = null;
        if (proxy.UserInfo is { Length: > 0 } info)
        {
            int colon = info.IndexOf(':');
            (user, password) = colon < 0 ? (Uri.UnescapeDataString(info), "") : (Uri.UnescapeDataString(info[..colon]), Uri.UnescapeDataString(info[(colon + 1)..]));
        }
        else if (credentials?.GetCredential(proxy, "Basic") is { UserName.Length: > 0 } credential)
        {
            (user, password) = (credential.UserName, credential.Password);
        }

        return user is null ? null : Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes($"{user}:{password}"));
    }

    // A tunnel the proxy would not open (its status only: nothing it sent is repeated).
    private sealed class ProxyTunnelException(Uri proxy, string target, string what)
        : IOException($"the proxy {proxy.GetLeftPart(UriPartial.Authority)} {what} for a tunnel to {target}");

    private static async Task<IPAddress[]> Resolve(string host, ImageInputOptions options, CancellationToken token)
    {
        string name = host.Trim('[', ']');
        if (IPAddress.TryParse(name, out var literal))
        {
            return [literal];
        }

        var addresses = options.Resolver is { } resolve ? await resolve(name, token) : await Dns.GetHostAddressesAsync(name, token);
        return addresses.Length > 0 ? addresses : throw new SocketException((int)SocketError.HostNotFound);
    }

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

    // The proxy that would carry a request to `uri` (the options' or the system's), or null when it goes directly.
    private static Uri? ProxyFor(Uri uri, ImageInputOptions options)
    {
        var proxy = options.Proxy ?? HttpClient.DefaultProxy;
        if (proxy.IsBypassed(uri))
        {
            return null;
        }

        var through = proxy.GetProxy(uri);
        return through is null || through == uri ? null : through;
    }

    // The request's host (each hop's): an http(s) address, refused through a proxy when the options say so; then (private
    // addresses not allowed) looked up once, every address public, and the pin the connection is held to: those addresses,
    // directly or through the proxy's tunnel (https only: a plain http request through a proxy cannot be held to them).
    private static async Task<Pin?> CheckHost(Uri uri, string url, ImageInputOptions options, CancellationToken token)
    {
        string hop = Hop(uri, url);
        if (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps)
        {
            throw new ArgumentException($"image_url: {Shorten(url)}{hop} is not an http(s) address.");
        }

        var proxy = ProxyFor(uri, options);
        if (proxy is not null && options.RefuseThroughProxy)
        {
            throw new ArgumentException($"image_url: {Shorten(url)}{hop} would be downloaded through the proxy {proxy.GetLeftPart(UriPartial.Authority)}, "
                + "which this server does not do (ImageInputOptions.RefuseThroughProxy; idrak serve --refuse-proxied-image-urls).");
        }

        if (options.AllowPrivateUrls)
        {
            return null;
        }

        IPAddress[] addresses;
        try
        {
            addresses = await Resolve(uri.IdnHost, options, token);
        }
        catch (SocketException e)
        {
            throw new ArgumentException($"image_url: {Shorten(url)}{hop}: the host {uri.IdnHost} does not resolve ({e.SocketErrorCode}).");
        }

        if (addresses.FirstOrDefault(a => PrivateKind(a) is not null) is { } refused)
        {
            throw new ArgumentException(Refusal(url, hop, refused));
        }

        if (proxy is not null && uri.Scheme == Uri.UriSchemeHttp)
        {
            throw new ArgumentException($"image_url: {Shorten(url)}{hop} is a plain http address and a proxy ({proxy.GetLeftPart(UriPartial.Authority)}) "
                + "would carry it, looking the name up again, so the server could not hold it to the address it checked: give an https address.");
        }

        return new Pin(addresses, proxy, proxy is null ? null : (options.Proxy ?? HttpClient.DefaultProxy).Credentials);
    }

    private static string Hop(Uri uri, string url) => uri.OriginalString == url ? "" : $" (redirected to {Shorten(uri.OriginalString)})";

    private static string Refusal(string url, string hop, IPAddress address) =>
        $"image_url: {Shorten(url)}{hop} reaches {address}, a {PrivateKind(address)} address, which this server does not download from "
        + "(loopback, link-local, private, multicast and reserved addresses are refused). The server's owner can allow them "
        + "(ImageInputOptions.AllowPrivateUrls; idrak serve --allow-private-image-urls).";

    // The answer of the URL after its redirects (at most MaxRedirects, each hop's host checked and pinned).
    private static async Task<HttpResponseMessage> Fetch(string url, ImageInputOptions options, CancellationToken token)
    {
        var current = new Uri(url);
        for (int hop = 0; ; hop++)
        {
            var pin = await CheckHost(current, url, options, token);
            var client = pin is not null ? PinnedDownloads.Value
                : options.Proxy is { } proxy ? ProxiedDownloads.GetValue(proxy, p => new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = false, Proxy = p, UseProxy = true })
                { Timeout = Timeout.InfiniteTimeSpan })
                : AnyDownloads.Value;
            using var request = new HttpRequestMessage(HttpMethod.Get, current);
            if (pin is not null)
            {
                request.Options.Set(PinKey, pin);
            }

            HttpResponseMessage response;
            try
            {
                response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, token);
            }
            catch (HttpRequestException e) when (e.InnerException is ProxyTunnelException tunnel)
            {
                throw new ArgumentException($"image_url: {Shorten(url)}{Hop(current, url)} could not be downloaded: {tunnel.Message}.");
            }

            if ((int)response.StatusCode is 301 or 302 or 303 or 307 or 308 && response.Headers.Location is { } location)
            {
                response.Dispose();
                if (hop >= MaxRedirects)
                {
                    throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"image_url: {Shorten(url)} redirects more than {MaxRedirects} times."));
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
