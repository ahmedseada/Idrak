// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Samples.HousePrices;

/// <summary>
/// Writes a synthetic but realistic house-price dataset. Prices depend non-linearly on the features
/// (location decays exponentially with distance, quality multiplies the value of floor area, age has
/// a renovation bump) plus ±6% noise, so a neural network clearly beats a straight line.
/// </summary>
internal static class HouseDataGenerator
{
    public const string Header = "id,area_sqft,bedrooms,bathrooms,age_years,distance_km,quality,garage_spaces,has_pool,lot_sqft,price";

    public static void Write(string path, int count = 2500, int seed = 7)
    {
        var random = new Random(seed);
        double Normal(double mean, double std) =>
            mean + std * Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());

        var sb = new StringBuilder(Header).AppendLine();
        for (int id = 1; id <= count; id++)
        {
            double area = Math.Clamp(Normal(1850, 650), 450, 5200);
            int bedrooms = (int)Math.Clamp(Math.Round(area / 620 + Normal(0, 0.6)), 1, 6);
            int bathrooms = (int)Math.Clamp(Math.Round(bedrooms * 0.6 + Normal(0.3, 0.5)), 1, 4);
            double age = Math.Round(random.NextDouble() * 85);
            double distance = Math.Round(0.5 + 38 * Math.Pow(random.NextDouble(), 1.6), 1);
            int quality = (int)Math.Clamp(Math.Round(Normal(6, 1.6)), 1, 10);
            int garage = random.NextDouble() switch { < 0.15 => 0, < 0.55 => 1, < 0.92 => 2, _ => 3 };
            int pool = random.NextDouble() < 0.12 ? 1 : 0;
            double lot = Math.Round(area * (1.3 + 2.5 * random.NextDouble()) + distance * 180);

            double value = 45_000
                + area * 115 * (0.72 + 0.065 * quality)
                + 14_000 * bathrooms + 5_000 * bedrooms
                + 9_500 * garage + 28_000 * pool
                + 6 * Math.Sqrt(lot) * 40
                - 1_500 * age + 11 * age * age;
            double location = 0.62 + 0.9 * Math.Exp(-distance / 11);
            double price = Math.Round(value * location * (1 + Normal(0, 0.06)) / 100) * 100;

            sb.AppendLine(string.Create(CultureInfo.InvariantCulture,
                $"{id},{area:F0},{bedrooms},{bathrooms},{age:F0},{distance:F1},{quality},{garage},{pool},{lot:F0},{Math.Max(price, 30_000):F0}"));
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, sb.ToString());
    }
}
