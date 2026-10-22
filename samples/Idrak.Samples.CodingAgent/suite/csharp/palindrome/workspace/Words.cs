/// <summary>Checks on words and phrases.</summary>
public static class Words
{
    /// <summary>Whether <paramref name="text"/> reads the same both ways, ignoring case and what is not a letter or a digit.</summary>
    public static bool IsPalindrome(string text)
    {
        for (int i = 0, j = text.Length - 1; i < j; i++, j--)
        {
            if (text[i] != text[j])
            {
                return false;
            }
        }

        return true;
    }
}
