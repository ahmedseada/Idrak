using Shapes;

// The tests, named "group: what it checks", run by IDRAK_FILTER (texts separated by '|'; none runs them all).
string[] filters = Environment.GetEnvironmentVariable("IDRAK_FILTER")?.Split('|', StringSplitOptions.RemoveEmptyEntries) ?? [];
int failed = 0;
foreach (var (name, run) in Suite.All.Where(t => filters.Length == 0 || filters.Any(f => t.Name.Contains(f, StringComparison.OrdinalIgnoreCase))))
{
    try
    {
        run();
        Console.WriteLine($"PASS {name}");
    }
    catch (Exception e)
    {
        failed++;
        Console.WriteLine($"FAIL {name}: {e.Message}");
    }
}

return failed == 0 ? 0 : 1;

internal static class Suite
{
    internal static readonly (string Name, Action Run)[] All =
    [
        ("area: a rectangle's area is its width times its height", () => Check(Area.Rectangle(2, 3) == 6, "2 x 3")),
        ("words: runs of spaces and punctuation separate words", CountsWords),
    ];

    private static void CountsWords()
    {
        Check(Words.CountWords("one  two, three") == 3, "three words");
        Check(Words.CountWords("") == 0, "no words");
    }

    private static void Check(bool condition, string what)
    {
        if (!condition)
        {
            throw new InvalidOperationException(what);
        }
    }
}
