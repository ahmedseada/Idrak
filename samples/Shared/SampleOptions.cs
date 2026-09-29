using System.Globalization;
using Idrak;
using Idrak.Diagnostics;

namespace Idrak.Samples;

/// <summary>Command-line options shared by the samples.</summary>
internal sealed record SampleOptions
{
    public Device Device { get; init; } = Device.Default;
    public int? Epochs { get; init; }
    public int? BatchSize { get; init; }
    public TelemetryLevel LogLevels { get; init; } = TelemetryLevel.Training;
    public string? LogFile { get; init; }
    public string? DataFile { get; init; }

    /// <summary>True for --mode predict: load the saved model and run inference only, no training.</summary>
    public bool PredictOnly { get; init; }

    /// <summary>--model: where the trained model is saved (train mode) or loaded from (predict mode).</summary>
    public string? ModelFile { get; init; }

    /// <summary>--input: sample-specific inference input (e.g. comma-separated features or a sentence).</summary>
    public string? Input { get; init; }

    /// <summary>The model path: --model if given, else models/<paramref name="fileName"/> next to the executable.</summary>
    public string ModelPath(string fileName)
    {
        string path = ModelFile ?? Path.Combine(AppContext.BaseDirectory, "models", fileName);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        return path;
    }

    /// <summary>In predict mode, fails with a helpful message when the model file is missing.</summary>
    public bool RequireModel(string path)
    {
        if (File.Exists(path))
        {
            return true;
        }

        Console.Error.WriteLine($"error: no trained model at {path}. Run without --predict first to train and save one (or pass --model <path>).");
        return false;
    }

    /// <summary>Values of sample-specific options (declared by the sample), keyed by name without dashes.</summary>
    public IReadOnlyDictionary<string, string> Extra { get; init; } = new Dictionary<string, string>();

    /// <summary>A sample-specific option's value, or <paramref name="fallback"/>.</summary>
    public string? Get(string name, string? fallback = null) => Extra.TryGetValue(name, out var v) ? v : fallback;

    public const string Usage = """
        Options:
          --device <auto|cpu|cuda|cuda:N>  where to run (default: auto = GPU if available, else CPU)
          --cpu / --cuda                   shortcuts for --device cpu / --device cuda
          --threads <n>                    CPU threads to use (default: all cores)
          --gpu-memory <MiB>               cap GPU tensor memory (default: unlimited)
          --cpu-memory <MiB>               cap CPU tensor memory (default: unlimited)
          --epochs <n>                     training epochs
          --batch-size <n>                 samples per batch
          --log <levels>                   comma list: training,batches,gradients,layers,operations,inference,all
          --log-file <path>                also write telemetry as JSON Lines to this file
          --data <path>                    dataset file (house-price sample)
          --mode <train|predict>           train and save the model (default), or load it and only run inference
          --predict                        shortcut for --mode predict
          --model <path>                   model file to save/load (default: models/<sample>.weights)
          --input <value>                  inference input (see the sample's header comment)
          --help                           show this help
        """;

    /// <summary>Parses the arguments and applies device and resource settings globally. Returns null after printing help.</summary>
    public static SampleOptions? Parse(string[] args, params (string Name, string Help)[] extra)
    {
        try
        {
            return ParseCore(args, extra);
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or FormatException)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            Environment.Exit(2);
            return null;
        }
    }

    private static SampleOptions? ParseCore(string[] args, (string Name, string Help)[] extra)
    {
        var options = new SampleOptions();
        var values = new Dictionary<string, string>();
        for (int i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
            int NextInt() => int.Parse(Next(), CultureInfo.InvariantCulture);

            switch (args[i].ToLowerInvariant())
            {
                case "--help" or "-h" or "-?":
                    Console.WriteLine(Usage);
                    foreach (var (name, help) in extra)
                    {
                        Console.WriteLine($"  --{name,-31}{help}");
                    }

                    return null;
                case var option when option.StartsWith("--", StringComparison.Ordinal) && extra.Any(e => e.Name == option[2..]):
                    values[option[2..]] = Next();
                    break;
                case "--cpu":
                    options = options with { Device = Device.Cpu };
                    break;
                case "--cuda" or "--gpu":
                    options = options with { Device = Device.Cuda() };
                    break;
                case "--device":
                    options = options with { Device = ParseDevice(Next()) };
                    break;
                case "--threads":
                    ComputeResources.MaxCpuThreads = NextInt();
                    break;
                case "--gpu-memory":
                    ComputeResources.GpuMemoryLimit = NextInt() * 1024L * 1024;
                    break;
                case "--cpu-memory":
                    ComputeResources.CpuMemoryLimit = NextInt() * 1024L * 1024;
                    break;
                case "--epochs":
                    options = options with { Epochs = NextInt() };
                    break;
                case "--batch-size":
                    options = options with { BatchSize = NextInt() };
                    break;
                case "--log":
                    options = options with { LogLevels = ParseLevels(Next()) };
                    break;
                case "--log-file":
                    options = options with { LogFile = Next() };
                    break;
                case "--data":
                    options = options with { DataFile = Next() };
                    break;
                case "--predict":
                    options = options with { PredictOnly = true };
                    break;
                case "--mode":
                    options = options with
                    {
                        PredictOnly = Next().ToLowerInvariant() switch
                        {
                            "predict" or "inference" => true,
                            "train" => false,
                            var m => throw new ArgumentException($"Unknown mode '{m}'. Use train or predict."),
                        },
                    };
                    break;
                case "--model":
                    options = options with { ModelFile = Next() };
                    break;
                case "--input":
                    options = options with { Input = Next() };
                    break;
                default:
                    throw new ArgumentException($"Unknown option '{args[i]}'.\n{Usage}");
            }
        }

        Device.Default = options.Device;
        return options with { Extra = values };
    }

    private static Device ParseDevice(string text) => text.ToLowerInvariant() switch
    {
        "auto" => Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu,
        "cpu" => Device.Cpu,
        "cuda" or "gpu" => Device.Cuda(),
        var s when s.StartsWith("cuda:", StringComparison.Ordinal) => Device.Cuda(int.Parse(s[5..], CultureInfo.InvariantCulture)),
        _ => throw new ArgumentException($"Unknown device '{text}'. Use auto, cpu, cuda or cuda:N."),
    };

    private static TelemetryLevel ParseLevels(string text) =>
        text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Aggregate(TelemetryLevel.None, (levels, name) => levels | Enum.Parse<TelemetryLevel>(name, ignoreCase: true));
}
