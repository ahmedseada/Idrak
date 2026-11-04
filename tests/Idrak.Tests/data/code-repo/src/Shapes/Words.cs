namespace Shapes;

/// <summary>Words in text, with strings that look like code to a reader that does not know C#.</summary>
public static class Words
{
    private const string Braces = "} { not a body {";
    private const string Verbatim = @"C:\path\ with ""quotes"" and { braces }";
    private const string Raw = """
        a raw string: void Fake() { return; } and "quotes"
        """;

    /// <summary>The number of words in <paramref name="text"/>: runs of letters or digits, any other character between them.</summary>
    public static int CountWords(string text)
    {
        int count = 0;
        bool inWord = false;
        foreach (char c in text)
        {
            bool part = char.IsLetterOrDigit(c) || c == '\'';
            if (part && !inWord)
            {
                count++;
            }

            inWord = part;
        }

        return count;
    }

    /// <summary>A label for the word count, in braces.</summary>
    public static string Label(string text) => $"{{{CountWords(text)}}} words in \"{text}\" ({Braces.Length + Verbatim.Length + Raw.Length})";

    /// <summary>Another Describe: two methods of one name are never a task (a call does not say which).</summary>
    public static string Describe(int count) => count == 1 ? "one word" : $"{count} words";

    /// <summary>Too small for a task.</summary>
    public static int Zero() => 0;
}
