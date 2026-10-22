/// <summary>Checks on words and phrases.</summary>
public static class Words
{
    /// <summary>Whether <paramref name="text"/> reads the same both ways, ignoring case and what is not a letter or a digit.</summary>
    public static bool IsPalindrome(string text)
    {
        int i = 0, j = text.Length - 1;
        while (i < j)
        {
            if (!char.IsLetterOrDigit(text[i]))
            {
                i++;
            }
            else if (!char.IsLetterOrDigit(text[j]))
            {
                j--;
            }
            else if (char.ToLowerInvariant(text[i++]) != char.ToLowerInvariant(text[j--]))
            {
                return false;
            }
        }

        return true;
    }
}
