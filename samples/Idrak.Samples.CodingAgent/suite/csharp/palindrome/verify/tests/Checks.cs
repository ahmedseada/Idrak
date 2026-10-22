// Hidden checks of this task: copied over tests/Checks.cs after the agent finishes; the program exits with 0 when
// every check passes.
int failed = 0;

void Check(bool ok, string what)
{
    if (!ok)
    {
        Console.WriteLine($"FAIL {what}");
        failed++;
    }
}

Check(Words.IsPalindrome("racecar"), "racecar is a palindrome");
Check(Words.IsPalindrome("Never odd or even"), "case and spaces are ignored");
Check(Words.IsPalindrome("A man, a plan, a canal: Panama"), "punctuation is ignored");
Check(!Words.IsPalindrome("hello"), "hello is not a palindrome");
Check(Words.IsPalindrome(""), "an empty text is a palindrome");
Check(!Words.IsPalindrome("ab, Ba c"), "ab, Ba c is not a palindrome");

Console.WriteLine(failed == 0 ? "all checks pass" : $"{failed} check(s) failed");
return failed == 0 ? 0 : 1;
