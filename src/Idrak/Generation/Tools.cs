// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Idrak.Generation;

/// <summary>Built-in tools.</summary>
public static partial class WebTools
{
    /// <summary>
    /// <c>web_fetch</c>: downloads a page and returns its text (tags, scripts and styles removed; whitespace collapsed),
    /// cut to <paramref name="maxCharacters"/>. Only absolute http(s) URLs accepted by <paramref name="allow"/> are fetched.
    /// </summary>
    public static Tool Fetch(HttpClient http, Func<Uri, bool> allow, int maxCharacters)
    {
        ArgumentNullException.ThrowIfNull(allow);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxCharacters);
        var schema = new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject { ["url"] = new JsonObject { ["type"] = "string", ["description"] = "Absolute http(s) URL." } },
            ["required"] = new JsonArray("url"),
        };
        return Tool.Create("web_fetch", "Fetch a web page and return its text.", schema, async (args, token) =>
        {
            if (!Uri.TryCreate((string?)args["url"], UriKind.Absolute, out var url) || url.Scheme is not ("http" or "https"))
            {
                throw new ArgumentException("url must be an absolute http or https URL");
            }

            if (!allow(url))
            {
                throw new UnauthorizedAccessException($"{url} is not on the allowlist");
            }

            string html = await http.GetStringAsync(url, token).ConfigureAwait(false);
            string text = WebUtility.HtmlDecode(Tags().Replace(Blocks().Replace(html, " "), " "));
            return CollapseSpaces(text, maxCharacters);
        });
    }

    // Regex.Replace(text, @"\s+", " ").Trim() cut to max characters, in one pass that stops at the cut: \s is
    // char.IsWhiteSpace (as for Trim), so each run of white space becomes one space and none is kept at either end.
    internal static string CollapseSpaces(string text, int max)
    {
        char[] buffer = System.Buffers.ArrayPool<char>.Shared.Rent(Math.Min(text.Length, max));
        int length = 0;
        bool space = false;
        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                space = length > 0;
                continue;
            }

            if (space && length < max)
            {
                buffer[length++] = ' ';
            }

            if (length == max)
            {
                break;
            }

            buffer[length++] = c;
            space = false;
        }

        string result = new(buffer, 0, length);
        System.Buffers.ArrayPool<char>.Shared.Return(buffer);
        return result;
    }

    [GeneratedRegex(@"<(script|style)[^>]*>.*?</\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex Blocks();

    [GeneratedRegex("<[^>]+>")]
    private static partial Regex Tags();
}
