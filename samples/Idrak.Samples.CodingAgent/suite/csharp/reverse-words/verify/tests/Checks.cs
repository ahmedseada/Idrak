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

Check(Text.ReverseWords("hello big world") == "world big hello", "hello big world reverses to world big hello");
Check(Text.ReverseWords("single") == "single", "one word stays as it is");
Check(Text.ReverseWords("  one   two ") == "two one", "runs of spaces are not words");
Check(Text.ReverseWords("") == "", "an empty sentence gives an empty one");

Console.WriteLine(failed == 0 ? "all checks pass" : $"{failed} check(s) failed");
return failed == 0 ? 0 : 1;
