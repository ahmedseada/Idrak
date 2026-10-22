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

Check(MathTools.Sum([]) == 0, "Sum of no values is 0");
Check(MathTools.Sum([5]) == 5, "Sum of 5 is 5");
Check(MathTools.Sum([1, 2, 3]) == 6, "Sum of 1, 2, 3 is 6");
Check(MathTools.Sum([-4, 4, 10]) == 10, "Sum of -4, 4, 10 is 10");

Console.WriteLine(failed == 0 ? "all checks pass" : $"{failed} check(s) failed");
return failed == 0 ? 0 : 1;
