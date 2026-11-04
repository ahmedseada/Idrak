namespace Shapes;

/// <summary>Areas of plane shapes.</summary>
public static class Area
{
    /// <summary>The area of a rectangle: its width times its height.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A side is negative.</exception>
    public static double Rectangle(double width, double height)
    {
        if (width < 0 || height < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "A side of a rectangle is not negative.");
        }

        return width * height;
    }

    /// <summary>The area of a circle of radius <paramref name="radius"/> (no test reaches it).</summary>
    public static double Circle(double radius) => Math.PI * radius * radius;

    /// <summary>Shapes described by name.</summary>
    public static string Describe(string name) => $"a shape called {name} with {{braces}}";

    // Not written yet: nothing to remove.
    public static double Triangle(double a, double b, double c) => throw new NotImplementedException();
}
