/// <summary>Arithmetic over lists of numbers.</summary>
public static class MathTools
{
    /// <summary>The sum of <paramref name="values"/>; 0 when there are none.</summary>
    public static int Sum(int[] values)
    {
        int total = 0;
        for (int i = 0; i < values.Length - 1; i++)
        {
            total += values[i];
        }

        return total;
    }
}
