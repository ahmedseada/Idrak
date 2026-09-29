namespace Idrak.Samples.Summarizer;

/// <summary>A short report and its one-sentence reference summary.</summary>
public sealed record Report(string Kind, string Document, string Summary);

/// <summary>
/// Synthetic reports of four kinds. The first sentence sets the scene; the facts and some filler sentences follow
/// in random order. Writing the summary needs more than copying: the winner of a match comes from comparing two
/// scores, "rose" or "fell" from comparing two revenues, and "rain likely" from a percentage above 50.
/// </summary>
public static class Reports
{
    private static readonly string[] Teams = ["lions", "eagles", "sharks", "wolves", "tigers", "bears", "falcons", "hawks", "bulls", "rams", "foxes", "owls"];
    private static readonly string[] Cities = ["arden", "belmor", "castor", "dunmore", "elgin", "fairview", "granton", "halden", "irvale", "jasper", "kelso", "lindon"];
    private static readonly string[] Days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];
    private static readonly string[] People = ["ana", "ben", "carla", "dev", "eva", "farid", "gina", "hugo", "iris", "jonas"];
    private static readonly string[] Companies = ["acme", "borex", "cantor", "delton", "everon", "fulcrum", "gemini", "helix", "ionix", "juno"];
    private static readonly string[] Quarters = ["first", "second", "third", "fourth"];
    private static readonly string[] Buildings = ["warehouse", "school", "hotel", "factory", "library", "museum"];

    /// <summary>Generates <paramref name="count"/> reports, cycling through the four kinds.</summary>
    public static List<Report> Generate(int count, Random random)
    {
        var reports = new List<Report>(count);
        for (int i = 0; i < count; i++)
        {
            reports.Add((i % 4) switch
            {
                0 => Match(random),
                1 => Weather(random),
                2 => Company(random),
                _ => Fire(random),
            });
        }

        return reports;
    }

    private static T Pick<T>(Random r, T[] items) => items[r.Next(items.Length)];

    // Lead sentence first, then the facts and 2-3 fillers shuffled together.
    private static string Compose(Random r, string lead, string[] facts, string[] fillers)
    {
        var body = facts.Concat(fillers.OrderBy(_ => r.Next()).Take(r.Next(2, 4))).OrderBy(_ => r.Next());
        return string.Join(" ", [lead, .. body]);
    }

    private static Report Match(Random r)
    {
        string a = Pick(r, Teams), b;
        do { b = Pick(r, Teams); } while (b == a);
        int goalsA = r.Next(7), goalsB;
        do { goalsB = r.Next(7); } while (goalsB == goalsA);
        string city = Pick(r, Cities), day = Pick(r, Days);
        string Goals(int n) => n == 1 ? "1 goal" : $"{n} goals";
        string document = Compose(r, $"the {a} played the {b} in {city} on {day} .",
            [$"the {a} scored {Goals(goalsA)} .", $"the {b} scored {Goals(goalsB)} ."],
            [$"a crowd of {r.Next(2, 60)} thousand fans watched the game .", $"{Pick(r, People)} was named player of the match .",
             $"the weather was {Pick(r, new[] { "cold", "warm", "windy", "wet" })} .", "tickets sold out a week before the game .",
             $"the {a} coach praised the defence .", $"both teams play again next {Pick(r, Days)} ."]);
        var (winner, loser) = goalsA > goalsB ? (a, b) : (b, a);
        return new Report("match", document, $"the {winner} beat the {loser} {Math.Max(goalsA, goalsB)} to {Math.Min(goalsA, goalsB)} in {city} .");
    }

    private static Report Weather(Random r)
    {
        string city = Pick(r, Cities), day = Pick(r, Days);
        int high = r.Next(10, 36), low = high - r.Next(3, 13), rain = 5 * r.Next(0, 20);
        string document = Compose(r, $"here is the weather report for {city} on {day} .",
            [$"the high will be {high} degrees and the low will be {low} degrees .", $"there is a {rain} percent chance of rain ."],
            [$"winds will be {Pick(r, new[] { "light", "strong" })} from the {Pick(r, new[] { "north", "south", "east", "west" })} .",
             $"the sun rises at {r.Next(5, 8)} am .", "air quality will be good .", "a warmer week is expected .", "roads may be busy in the evening ."]);
        return new Report("weather", document, $"{city} will reach {high} degrees on {day} with {(rain >= 50 ? "rain likely" : "little chance of rain")} .");
    }

    private static Report Company(Random r)
    {
        string company = Pick(r, Companies), quarter = Pick(r, Quarters);
        int revenue = r.Next(10, 100), before;
        do { before = r.Next(10, 100); } while (before == revenue);
        string document = Compose(r, $"{company} reported results for the {quarter} quarter .",
            [$"revenue was {revenue} million dollars .", $"a year earlier revenue was {before} million dollars ."],
            [$"the company employs about {r.Next(2, 90)} hundred people .", $"chief executive {Pick(r, People)} thanked the staff .",
             $"shares closed at {r.Next(5, 99)} dollars .", $"the company plans to open {r.Next(2, 20)} new stores ."]);
        return new Report("company", document,
            $"{company} revenue {(revenue > before ? "rose" : "fell")} from {before} to {revenue} million dollars in the {quarter} quarter .");
    }

    private static Report Fire(Random r)
    {
        string building = Pick(r, Buildings), city = Pick(r, Cities);
        int rescued = r.Next(2, 41);
        string document = Compose(r, $"a fire broke out at a {building} in {city} on {Pick(r, Days)} .",
            [$"firefighters rescued {rescued} people from the building .", $"{r.Next(2, 12)} fire trucks were sent to the scene ."],
            ["the cause of the fire is not yet known .", $"nearby roads were closed for {r.Next(2, 9)} hours .",
             "two people were treated for smoke .", "the building will reopen next month ."]);
        return new Report("fire", document, $"{rescued} people were rescued from a {building} fire in {city} .");
    }
}
