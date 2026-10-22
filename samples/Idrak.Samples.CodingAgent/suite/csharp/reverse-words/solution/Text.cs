/// <summary>Helpers for sentences.</summary>
public static class Text
{
    /// <summary>The words of <paramref name="sentence"/> in reverse order, separated by single spaces.</summary>
    public static string ReverseWords(string sentence)
    {
        string[] words = sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return string.Join(' ', Enumerable.Reverse(words));
    }
}
