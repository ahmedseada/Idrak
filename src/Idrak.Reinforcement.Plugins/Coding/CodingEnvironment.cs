// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using Idrak.Abstraction.Reinforcement;
using Idrak.Reinforcement.Plugins.Coding.Abstractions;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;
using Idrak.Text.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// The <c>coding</c> environment kind (<see cref="EnvironmentKinds"/>, registered by <see cref="CodingPlugin"/>): an
/// agent with <see cref="CodingTools"/> works on coding tasks (<see cref="CodingTask"/>), each episode one task drawn
/// from its seed, rewarded by the task's verification. Observations are conversations (<see cref="ChatObservation"/>:
/// the system prompt, the task's prompt, the replies and the tools' results, the tools offered), actions the assistant's
/// replies.
/// <list type="bullet">
/// <item><b>Setup once per task:</b> the first episode that draws a task lays out its project and runs its setup commands
/// in a prepared copy, reused by every later episode (each starts from a clone of it); a task whose setup fails is left out
/// (episodes draw again from their seed) and reported (<see cref="CodingEnvironment.BrokenTasks"/>, the info's
/// <see cref="CodingInfo.BrokenTasks"/>).</item>
/// <item><b>Steps:</b> each reply's tool calls run through the tools as the agent's loop runs them
/// (<see cref="CodingAgent.RunToolCallsAsync"/>) and their results are the next observation; the episode ends when a reply
/// calls no tool (terminated), or when the rounds or the time run out (truncated).</item>
/// <item><b>The end:</b> the hidden tests are copied in (over any copy the agent made) and the verification commands run;
/// the reward is the option <c>reward</c>'s expression (<see cref="CodingRewards"/>, any <see cref="RewardFunctions"/>
/// name, "tests-passed" by default) over the episode's info (<see cref="CodingInfo"/>). Files the agent changed where the
/// hidden tests go are reported, and with <c>refuse-test-edits</c> the episode is refused: verification does not run and
/// counts as failed.</item>
/// <item><b>In parallel:</b> environments of one kind with the same options share the prepared copies; at most
/// <c>workspaces</c> of them run tools or verification at once (<see cref="WorkspacesVariable"/>; by default the CPU
/// threads, fewer when the free disk holds fewer copies of a prepared workspace), so the cores work on some episodes'
/// builds and tests while the device generates others' turns. An episode's workspace is deleted when it ends unless kept
/// (<c>keep</c>, <see cref="KeepWorkspacesVariable"/>).</item>
/// </list>
/// Options: <c>suite</c> (a suite folder, <see cref="CodingSuite"/>; unless the kind was made with tasks), <c>filter</c>,
/// <c>reward</c> (and each term's options as term.key), <c>rounds</c> (40), <c>time</c> (seconds, 1200), <c>system</c>
/// (the system prompt; <see cref="CodingAgent.SystemPrompt"/> of the languages), <c>languages</c> (the coding languages
/// the tools and the prompt work with, <see cref="CodingToolOptions.Languages"/>: all, detected in the tasks' folders,
/// none, or names; IDRAK_CODING_LANGUAGES, else all), <c>protect</c> (globs of files whose edits count as test edits
/// beside the hidden tests' paths, separated by commas; none by default), <c>think</c>, <c>sandbox</c>
/// (<see cref="CommandSandboxes"/>; IDRAK_SANDBOX, else process), <c>memory</c> (a command's memory: bytes, or with KB, MB,
/// GB), <c>processes</c>, <c>cpus</c>, <c>network</c> (true or false), <c>command-timeout</c> and <c>verify-timeout</c>
/// (seconds: a tool command's and a verification command's limit; by default measured per task: when a task is prepared
/// its verification runs once on its starting files, in a copy, and each limit is <c>timeout-scale</c> (10) times that
/// run's time, at least <c>timeout-least</c> seconds (30) and at most 300 and 600 seconds, so a program that never ends
/// costs seconds, not minutes), <c>timeout-scale</c>, <c>timeout-least</c>, <c>workspaces</c>, <c>keep</c>, <c>work</c> (the folder the copies are made in;
/// a new temporary folder by default), <c>refuse-test-edits</c>. A variant for one language
/// (<see cref="CodingEnvironmentVariant"/>, such as the C# plug-in's <c>csharp</c>) is this environment under its own
/// name, with its language's defaults, its own options and its own info.
/// </summary>
public sealed class CodingEnvironmentKind : IEnvironmentKind
{
    /// <summary>The kind's name: "coding".</summary>
    public const string KindName = "coding";

    /// <summary>The variable of the option <c>workspaces</c>: "IDRAK_RL_WORKSPACES".</summary>
    public const string WorkspacesVariable = "IDRAK_RL_WORKSPACES";

    /// <summary>The variable of the option <c>keep</c>: "IDRAK_RL_KEEP_WORKSPACES" (true or 1 keeps every episode's workspace).</summary>
    public const string KeepWorkspacesVariable = "IDRAK_RL_KEEP_WORKSPACES";

    private static readonly string[] OwnKeys =
    [
        "suite", "filter", "reward", "rounds", "time", "system", "languages", "protect", "think", "sandbox", "memory", "processes", "cpus", "network", "command-timeout",
        "verify-timeout", "timeout-scale", "timeout-least", "workspaces", "keep", "work", "refuse-test-edits",
    ];

    private readonly IReadOnlyList<CodingTask>? _tasks;
    private readonly CodingEnvironmentVariant? _variant;
    private readonly Dictionary<string, CodingSuiteRun> _runs = new(StringComparer.Ordinal);
    private readonly Lock _gate = new();

    /// <summary>The library's kind: its tasks come from the option <c>suite</c>.</summary>
    public CodingEnvironmentKind()
    {
        Name = KindName;
    }

    /// <summary>A kind of <paramref name="tasks"/> (the option <c>suite</c> is then refused).</summary>
    /// <param name="tasks">The tasks, at least one.</param>
    /// <param name="name">The kind's name (to register it beside the plug-in's).</param>
    public CodingEnvironmentKind(IReadOnlyList<CodingTask> tasks, string name = KindName)
    {
        ArgumentNullException.ThrowIfNull(tasks);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (tasks.Count == 0)
        {
            throw new ArgumentException("A coding environment has at least one task.", nameof(tasks));
        }

        _tasks = [.. tasks];
        Name = name;
    }

    /// <summary>The kind of <paramref name="variant"/>: the coding environment under the variant's name, with its defaults and options (its tasks from the option <c>suite</c>, or <paramref name="tasks"/>).</summary>
    /// <param name="variant">The variant.</param>
    /// <param name="tasks">The tasks (the option <c>suite</c> is then refused), or null.</param>
    public CodingEnvironmentKind(CodingEnvironmentVariant variant, IReadOnlyList<CodingTask>? tasks = null)
    {
        ArgumentNullException.ThrowIfNull(variant);
        ArgumentException.ThrowIfNullOrWhiteSpace(variant.Name);
        if (tasks is { Count: 0 })
        {
            throw new ArgumentException("A coding environment has at least one task.", nameof(tasks));
        }

        _variant = variant;
        _tasks = tasks is null ? null : [.. tasks];
        Name = variant.Name;
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <summary>The variant the kind runs (null for the plain coding environment).</summary>
    public CodingEnvironmentVariant? Variant => _variant;

    /// <inheritdoc />
    public string Summary => _variant?.Summary ?? "coding tasks (suite=DIR): the agent works with the coding tools; the hidden tests' verification is the reward (reward=tests-passed)";

    /// <inheritdoc />
    public IReadOnlyCollection<string> Keys => [.. OwnKeys.Concat(_variant?.Keys ?? []).Where(k => _tasks is null || k != "suite")];

    /// <inheritdoc />
    public Type ObservationType => typeof(ChatObservation);

    /// <inheritdoc />
    public Type ActionType => typeof(ChatMessage);

    /// <inheritdoc />
    public IEnvironment Create(IReadOnlyDictionary<string, string>? options = null, int seed = 0)
    {
        var settings = CodingSettings.Parse(this, options);
        lock (_gate)
        {
            if (!_runs.TryGetValue(settings.Key, out var run))
            {
                var tasks = _tasks ?? (settings.Suite is { } suite ? CodingSuite.Load(suite)
                    : throw new ArgumentException($"The {Name} environment reads its tasks from the option suite (a folder of task folders with task.json).", nameof(options)));
                tasks = [.. tasks.Where(t => settings.Filter is null || t.Id.Contains(settings.Filter, StringComparison.OrdinalIgnoreCase))];
                if (tasks.Count == 0)
                {
                    throw new ArgumentException($"The {Name} environment has no task{(settings.Filter is null ? "" : $" whose id holds '{settings.Filter}'")}.", nameof(options));
                }

                string key = settings.Key;
                run = new CodingSuiteRun(settings, tasks, _gate, () => _runs.Remove(key), _variant);
                _runs[settings.Key] = run;
            }

            return new CodingEnvironment(run, run.Acquire());
        }
    }
}

// The parsed options of one Create call; equal options share a run (prepared copies, the workspace limit).
internal sealed record CodingSettings(string? Suite, string? Filter, string Reward, Dictionary<string, string> RewardOptions, int Rounds, TimeSpan Time, string? SystemPrompt,
    bool? Think, CodingToolOptions Tools, TimeSpan VerifyTimeout, int Workspaces, bool Keep, string? Work, bool RefuseTestEdits, string Key, string Languages,
    IReadOnlyList<string> Protect, Dictionary<string, string> VariantOptions)
{
    /// <summary>The most a measured tool command's limit may be (the option command-timeout's old default).</summary>
    public static readonly TimeSpan MostCommand = TimeSpan.FromSeconds(300);

    /// <summary>The most a measured verification command's limit may be (the option verify-timeout's old default).</summary>
    public static readonly TimeSpan MostVerify = TimeSpan.FromSeconds(600);

    // command-timeout and verify-timeout as given (null: measured per task), and the measured limits' multiple and least.
    public TimeSpan? GivenCommandTimeout { get; init; }

    public TimeSpan? GivenVerifyTimeout { get; init; }

    public double TimeoutScale { get; init; } = 10;

    public TimeSpan TimeoutLeast { get; init; } = TimeSpan.FromSeconds(30);

    // A task's limits: as given, else `scale` times its measured verification, between the least and the most.
    public (TimeSpan Command, TimeSpan Verify) Limits(TimeSpan? measured)
    {
        TimeSpan Of(TimeSpan most) => measured is { } m ? TimeSpan.FromTicks(Math.Clamp((long)(m.Ticks * TimeoutScale), Math.Min(TimeoutLeast.Ticks, most.Ticks), most.Ticks)) : most;
        return (GivenCommandTimeout ?? Of(MostCommand), GivenVerifyTimeout ?? Of(MostVerify));
    }

    public static CodingSettings Parse(CodingEnvironmentKind kind, IReadOnlyDictionary<string, string>? given)
    {
        // A variant's defaults for the coding options, under what the caller gives.
        var options = new Dictionary<string, string>(kind.Variant?.Defaults() ?? new Dictionary<string, string>(), StringComparer.Ordinal);
        if (kind.Variant is { } defaultsOf && !options.ContainsKey("languages"))
        {
            options["languages"] = defaultsOf.Language;
        }

        foreach (var (key, value) in given ?? new Dictionary<string, string>())
        {
            options[key] = value;
        }

        string? suite = null, filter = null, work = null, think = null, system = null;
        string reward = CodingRewards.TestsPassed, sandbox = CommandSandboxes.DefaultName, languages = CodingLanguages.DefaultChoice;
        IReadOnlyList<string> protect = [];
        var rewardOptions = new Dictionary<string, string>(StringComparer.Ordinal);
        var variantOptions = new Dictionary<string, string>(StringComparer.Ordinal);
        int rounds = 40, processes = 0, workspaces = Positive(kind, CodingEnvironmentKind.WorkspacesVariable, Environment.GetEnvironmentVariable(CodingEnvironmentKind.WorkspacesVariable), allowEmpty: true);
        double time = 1200, cpus = 0, scale = 10, least = 30;
        double? commandTimeout = null, verifyTimeout = null;
        long memory = 0;
        bool network = true, refuse = false, keep = Flag(kind, CodingEnvironmentKind.KeepWorkspacesVariable, Environment.GetEnvironmentVariable(CodingEnvironmentKind.KeepWorkspacesVariable) ?? "false");
        foreach (var (key, value) in options)
        {
            switch (key)
            {
                case "suite" when kind.Keys.Contains("suite"):
                    suite = Path.GetFullPath(value);
                    break;
                case "filter":
                    filter = value;
                    break;
                case "reward":
                    reward = value;
                    break;
                case "rounds":
                    rounds = Positive(kind, key, value);
                    break;
                case "time":
                    time = Seconds(kind, key, value);
                    break;
                case "system":
                    system = value;
                    break;
                case "languages":
                    languages = value;
                    break;
                case "protect":
                    protect = [.. value.Split([',', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];
                    break;
                case "think":
                    think = Flag(kind, key, value) ? "true" : "false";
                    break;
                case "sandbox":
                    sandbox = value;
                    break;
                case "memory":
                    memory = Bytes(kind, key, value);
                    break;
                case "processes":
                    processes = Positive(kind, key, value);
                    break;
                case "cpus":
                    cpus = Seconds(kind, key, value);
                    break;
                case "network":
                    network = Flag(kind, key, value);
                    break;
                case "command-timeout":
                    commandTimeout = Seconds(kind, key, value);
                    break;
                case "verify-timeout":
                    verifyTimeout = Seconds(kind, key, value);
                    break;
                case "timeout-scale":
                    scale = Seconds(kind, key, value);
                    break;
                case "timeout-least":
                    least = Seconds(kind, key, value);
                    break;
                case "workspaces":
                    workspaces = Positive(kind, key, value);
                    break;
                case "keep":
                    keep = Flag(kind, key, value);
                    break;
                case "work":
                    work = Path.GetFullPath(value);
                    break;
                case "refuse-test-edits":
                    refuse = Flag(kind, key, value);
                    break;
                case var own when kind.Variant?.Keys.Contains(own, StringComparer.Ordinal) == true:
                    variantOptions[own] = value;
                    break;
                default:
                    if (!key.Contains('.', StringComparison.Ordinal))
                    {
                        throw new ArgumentException($"The {kind.Name} environment does not take the option '{key}'; it takes: {string.Join(", ", kind.Keys)}, and a reward term's options as term.key.",
                            nameof(options));
                    }

                    rewardOptions[key] = value;
                    break;
            }
        }

        kind.Variant?.Check(variantOptions);
        var tools = new CodingToolOptions
        {
            Sandbox = sandbox, CommandTimeout = commandTimeout is { } c ? TimeSpan.FromSeconds(c) : MostCommand, MaxMemoryBytes = memory, MaxProcesses = processes, MaxCpus = cpus, Network = network,
            Languages = languages, Environment = kind.Variant?.Variables(variantOptions),
        };
        string identity = string.Join("\n", options.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}={p.Value}"))
                          + $"\n#sandbox={sandbox}\n#workspaces={workspaces}\n#keep={keep}\n#languages={languages}";
        return new CodingSettings(suite, filter, reward, rewardOptions, rounds, TimeSpan.FromSeconds(time), system, think is null ? null : think == "true", tools,
            verifyTimeout is { } v ? TimeSpan.FromSeconds(v) : MostVerify, workspaces, keep, work, refuse, identity, languages, protect, variantOptions)
        {
            GivenCommandTimeout = commandTimeout is { } commandSeconds ? TimeSpan.FromSeconds(commandSeconds) : null,
            GivenVerifyTimeout = verifyTimeout is { } verifySeconds ? TimeSpan.FromSeconds(verifySeconds) : null,
            TimeoutScale = scale, TimeoutLeast = TimeSpan.FromSeconds(least),
        };
    }

    private static int Positive(CodingEnvironmentKind kind, string key, string? value, bool allowEmpty = false) =>
        allowEmpty && string.IsNullOrWhiteSpace(value) ? 0
        : int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n > 0 ? n
        : throw new ArgumentException($"The {kind.Name} environment's {key}={value} is not a positive whole number.");

    private static double Seconds(CodingEnvironmentKind kind, string key, string value) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double s) && s > 0 && double.IsFinite(s) ? s
        : throw new ArgumentException($"The {kind.Name} environment's {key}={value} is not a positive number.");

    private static bool Flag(CodingEnvironmentKind kind, string key, string value) => value.Trim().ToLowerInvariant() switch
    {
        "true" or "1" or "on" or "yes" => true,
        "false" or "0" or "off" or "no" or "" => false,
        _ => throw new ArgumentException($"The {kind.Name} environment's {key}={value} is true or false."),
    };

    // A size: bytes, or a number with B, KB, MB or GB (1024-based).
    private static long Bytes(CodingEnvironmentKind kind, string key, string value)
    {
        string text = value.Trim().ToUpperInvariant();
        long unit = 1;
        foreach (var (suffix, size) in new[] { ("GB", 1L << 30), ("MB", 1L << 20), ("KB", 1L << 10), ("B", 1L) })
        {
            if (text.EndsWith(suffix, StringComparison.Ordinal))
            {
                text = text[..^suffix.Length].Trim();
                unit = size;
                break;
            }
        }

        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double number) && number > 0 && double.IsFinite(number) ? (long)(number * unit)
            : throw new ArgumentException($"The {kind.Name} environment's {key}={value} is not a size (bytes, or a number with KB, MB or GB).");
    }
}

// What the environments of one kind with the same options share: the tasks, their prepared copies (setup once per task),
// the tasks whose setup failed, the tools' definitions and the limit on workspaces at work; released by the last one.
internal sealed class CodingSuiteRun
{
    private readonly Lock _kind;                             // the kind's lock: users come and go under it
    private readonly Action _forget;                         // removes the run from its kind (under the kind's lock)
    private readonly Lazy<Task<Prepared>>[] _prepared;
    private readonly string _work;
    private readonly bool _ownsWork;
    private readonly Lock _gate = new();
    private SemaphoreSlim? _workspaces;
    private int _users, _slots;
    private long _largest;

    public CodingSuiteRun(CodingSettings settings, IReadOnlyList<CodingTask> tasks, Lock kind, Action forget, CodingEnvironmentVariant? variant = null)
    {
        Settings = settings;
        Variant = variant;
        _kind = kind;
        _forget = forget;

        // The languages once for the run (detected: in any task's folders), named explicitly for every workspace's tools,
        // so the tools' definitions and the prompt are the same in every episode.
        Languages = settings.Languages.Trim().Equals(CodingLanguages.Detected, StringComparison.OrdinalIgnoreCase)
            ? [.. CodingLanguages.Registered.Where(l => tasks.Any(t => new[] { t.Base, t.Workspace }.Any(f => f is not null && l.Detect(f, settings.Tools.DetectionDepth))))]
            : CodingLanguages.Resolve(settings.Languages);
        Tools = settings.Tools with { Languages = Languages.Count == 0 ? CodingLanguages.None : string.Join(",", Languages.Select(l => l.Name)) };
        SystemPrompt = settings.SystemPrompt ?? CodingAgent.SystemPrompt(Languages);
        if (variant is not null)
        {
            var language = CodingLanguages.Get(variant.Language);
            tasks = [.. tasks.Select(t => variant.Task(t, language, settings.VariantOptions))];
        }

        Tasks = tasks;
        _ownsWork = settings.Work is null || !Directory.Exists(settings.Work);
        _work = settings.Work ?? Path.Combine(Path.GetTempPath(), "idrak-coding", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_work);
        Verifier = new CodingVerifier(new CodingVerifierOptions { Tools = Tools, Timeout = settings.VerifyTimeout });
        try
        {
            // The sandbox is named and checked now (a missing one, or limits it cannot keep, fail before any episode); the
            // tools' definitions are the same for every workspace, so every observation offers the same ones.
            var probe = new CodingTools(_work, Tools);
            Sandbox = probe.Sandbox;
            Definitions = [.. probe.All.Select(t => t.Definition)];
            _ = TextRewards.ScoreAsync(settings.Reward, [], settings.RewardOptions).AsTask().GetAwaiter().GetResult();   // a bad reward fails now, not at the first episode's end
        }
        catch
        {
            Delete(_work, _ownsWork);
            throw;
        }

        _prepared = [.. tasks.Select((task, i) => new Lazy<Task<Prepared>>(() => PrepareAsync(task, i), LazyThreadSafetyMode.ExecutionAndPublication))];
    }

    public CodingSettings Settings { get; }

    public CodingEnvironmentVariant? Variant { get; }

    // The coding languages of the run, the tools' options naming them, and the system prompt.
    public IReadOnlyList<ICodingLanguage> Languages { get; }

    public CodingToolOptions Tools { get; }

    public string SystemPrompt { get; }

    public IReadOnlyList<CodingTask> Tasks { get; }

    public CodingVerifier Verifier { get; }

    public ICommandSandbox Sandbox { get; }

    public IReadOnlyList<ToolDefinition> Definitions { get; }

    public ConcurrentDictionary<string, string> Broken { get; } = new(StringComparer.Ordinal);

    // The tasks left out so far, by id, in order (for the info).
    public string BrokenList => string.Join(",", Broken.Keys.Order(StringComparer.Ordinal));

    // A new environment of the run (called under the kind's lock): its slot, which names its episodes' folders.
    public int Acquire()
    {
        _users++;
        return _slots++;
    }

    // An environment is gone; the last one takes the run from its kind and deletes its folders (unless kept).
    public void Release()
    {
        lock (_kind)
        {
            if (--_users > 0)
            {
                return;
            }

            _forget();
        }

        if (!Settings.Keep)
        {
            Delete(Path.Combine(_work, "prepared"), true);
            Delete(Path.Combine(_work, "episodes"), true);
            Delete(_work, _ownsWork);
        }
    }

    // The task an episode of `seed` plays and its prepared copy: the tasks in an order drawn from the seed, the first whose
    // setup passed (so a seed always plays the same task).
    public (CodingTask Task, Prepared Copy) Choose(int seed)
    {
        var order = Enumerable.Range(0, Tasks.Count).ToArray();
        new Random(seed).Shuffle(order);
        foreach (int i in order)
        {
            var prepared = _prepared[i].Value.GetAwaiter().GetResult();
            if (!prepared.Broken)
            {
                return (Tasks[i], prepared);
            }
        }

        throw new InvalidOperationException($"Every task's setup failed, so no episode can start: {string.Join("; ", Broken.Select(b => $"{b.Key}: {b.Value.Trim()}"))}");
    }

    public string EpisodeFolder(int slot, int episode) =>
        Path.Combine(_work, "episodes", string.Create(CultureInfo.InvariantCulture, $"s{slot:000}-e{episode:000000}"));

    // At most `workspaces` episodes run tools or verification at once (made on first use, from the cores and the free disk).
    public SemaphoreSlim Workspaces
    {
        get
        {
            lock (_gate)
            {
                return _workspaces ??= new SemaphoreSlim(Settings.Workspaces > 0 ? Settings.Workspaces : MeasuredWorkspaces());
            }
        }
    }

    // The CPU threads, fewer when the free disk holds fewer than two copies per workspace of the largest prepared one.
    private int MeasuredWorkspaces()
    {
        int cores = Math.Max(1, ComputeResources.MaxCpuThreads);
        long largest = Interlocked.Read(ref _largest);
        try
        {
            long free = new DriveInfo(Path.GetPathRoot(_work)!).AvailableFreeSpace;
            return largest <= 0 ? cores : (int)Math.Clamp(free / (2 * largest), 1, cores);
        }
        catch (Exception e) when (e is IOException or ArgumentException or UnauthorizedAccessException)
        {
            return cores;
        }
    }

    private async Task<Prepared> PrepareAsync(CodingTask task, int index)
    {
        string folder = Path.Combine(_work, "prepared", index.ToString("000", CultureInfo.InvariantCulture));
        if (Directory.Exists(folder))
        {
            CodingWorkspace.Delete(folder);
        }

        CodingWorkspace.Prepare(task, folder, Languages);
        var setup = await Verifier.SetupAsync(task, Verifier.Checks(folder)).ConfigureAwait(false);
        if (!setup.Passed)
        {
            Broken[task.Id] = setup.Output;
            return new Prepared(folder, true, Settings.Tools.CommandTimeout, Settings.VerifyTimeout);
        }

        CodingWorkspace.CommitStart(folder, Languages);
        long bytes = new DirectoryInfo(folder).EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
        long seen;
        while (bytes > (seen = Interlocked.Read(ref _largest)) && Interlocked.CompareExchange(ref _largest, bytes, seen) != seen)
        {
        }

        var (command, verify) = Settings.Limits(Settings.GivenCommandTimeout is null || Settings.GivenVerifyTimeout is null ? await BaselineAsync(task, folder).ConfigureAwait(false) : null);
        return new Prepared(folder, false, command, verify);
    }

    // The time of the task's verification on its starting files, in a copy of the prepared folder (the hidden tests are
    // copied into the folder verified, which episodes clone), under the most a measured limit may be.
    private async Task<TimeSpan> BaselineAsync(CodingTask task, string prepared)
    {
        string copy = prepared + "-baseline";
        if (Directory.Exists(copy))
        {
            CodingWorkspace.Delete(copy);
        }

        CodingWorkspace.Clone(prepared, copy);
        try
        {
            var clock = Stopwatch.StartNew();
            await Verifier.VerifyAsync(task, new CodingTools(copy, Tools with { CommandTimeout = CodingSettings.MostVerify }, Sandbox)).ConfigureAwait(false);
            return clock.Elapsed;
        }
        finally
        {
            Delete(copy, true);
        }
    }

    private static void Delete(string folder, bool delete)
    {
        if (!delete)
        {
            return;
        }

        try
        {
            CodingWorkspace.Delete(folder);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }

    // A prepared task: its folder, whether its setup failed, and its tool commands' and verification commands' limits.
    public sealed record Prepared(string Folder, bool Broken, TimeSpan CommandLimit = default, TimeSpan VerifyLimit = default);
}

/// <summary>
/// An episode of the <c>coding</c> environment (<see cref="CodingEnvironmentKind"/>): a task drawn from the seed, a
/// workspace cloned from its prepared copy, the assistant's replies stepped through the coding tools, the hidden tests'
/// verification as the reward. One environment runs one episode at a time.
/// </summary>
public sealed class CodingEnvironment : IEnvironment<ChatObservation, ChatMessage>
{
    private readonly CodingSuiteRun _run;
    private readonly int _slot;
    private int _episodes;
    private CodingTask? _task;
    private string? _start;
    private CodingTools? _tools;
    private TimeSpan _verifyLimit;
    private IToolRegistry? _registry;
    private List<ChatMessage> _messages = [];
    private int _rounds, _calls, _errors;
    private string? _previous;
    private Stopwatch _clock = new();
    private bool _active, _disposed;

    internal CodingEnvironment(CodingSuiteRun run, int slot)
    {
        _run = run;
        _slot = slot;
    }

    /// <inheritdoc />
    public Space ObservationSpace => Space.Conversation;

    /// <inheritdoc />
    public Space ActionSpace => Space.Conversation;

    /// <summary>The tasks left out because their setup failed (id → the setup's output), shared by the kind's environments with the same options.</summary>
    public IReadOnlyDictionary<string, string> BrokenTasks => _run.Broken;

    /// <summary>The task of the episode under way (or the last one), or null before the first reset.</summary>
    public CodingTask? Task => _task;

    /// <summary>The episode's workspace (deleted when the episode ends unless the workspaces are kept), or null before the first reset.</summary>
    public string? Workspace => _tools?.Workspace;

    /// <inheritdoc />
    public ChatObservation Reset(int seed)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Abandon();
        var (task, prepared) = _run.Choose(seed);
        string folder = _run.EpisodeFolder(_slot, ++_episodes);
        if (Directory.Exists(folder))
        {
            CodingWorkspace.Delete(folder);
        }

        CodingWorkspace.Clone(prepared.Folder, folder);
        _task = task;
        _start = prepared.Folder;
        _verifyLimit = prepared.VerifyLimit;
        _tools = new CodingTools(folder, _run.Tools with { CommandTimeout = prepared.CommandLimit }, _run.Sandbox);
        _registry = _tools.Registry();
        _messages = [new ChatMessage("system", _run.SystemPrompt), new ChatMessage("user", task.Prompt)];
        (_rounds, _calls, _errors, _previous) = (0, 0, 0, null);
        _clock = Stopwatch.StartNew();
        _active = true;
        return Observation();
    }

    /// <inheritdoc />
    public async ValueTask<EnvironmentStep<ChatObservation>> StepAsync(ChatMessage action, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_active || _task is null || _registry is null || _tools is null)
        {
            throw new InvalidOperationException("No episode is under way: reset the coding environment first (also after an episode ended).");
        }

        ArgumentNullException.ThrowIfNull(action);
        var settings = _run.Settings;
        _rounds++;
        _messages.Add(action);
        if (action.ToolCalls is not { Count: > 0 })
        {
            return await EndAsync(action, truncated: false, cancellationToken).ConfigureAwait(false);
        }

        var left = settings.Time - _clock.Elapsed;
        if (left <= TimeSpan.Zero)
        {
            return await EndAsync(action, truncated: true, cancellationToken).ConfigureAwait(false);
        }

        using (var limit = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
        {
            limit.CancelAfter(left);
            var gate = _run.Workspaces;
            await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                var turn = await CodingAgent.RunToolCallsAsync(_registry, action, _previous, limit.Token).ConfigureAwait(false);
                _previous = turn.Key;
                _calls += turn.Results.Count;
                _errors += turn.Errors;
                _messages.AddRange(turn.Messages);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                gate.Release();
                return await EndAsync(action, truncated: true, cancellationToken).ConfigureAwait(false);
            }

            gate.Release();
        }

        if (_rounds >= settings.Rounds)
        {
            return await EndAsync(action, truncated: true, cancellationToken).ConfigureAwait(false);
        }

        return new EnvironmentStep<ChatObservation>(Observation(), 0f, false, false, Info());
    }

    // The episode's end: test edits looked for, the hidden tests copied in and verified (unless refused), the reward.
    private async ValueTask<EnvironmentStep<ChatObservation>> EndAsync(ChatMessage last, bool truncated, CancellationToken cancellationToken)
    {
        var settings = _run.Settings;
        var task = _task!;
        string workspace = _tools!.Workspace;
        IReadOnlyList<string> edits = [.. CodingWorkspace.TestEdits(task, _start!, workspace, _run.Languages)
            .Union(CodingWorkspace.ProtectedEdits(_start!, workspace, settings.Protect, _run.Languages), StringComparer.Ordinal).Order(StringComparer.Ordinal)];
        bool refused = settings.RefuseTestEdits && edits.Count > 0;
        _run.Variant?.BeforeVerify(workspace, settings.VariantOptions);
        CodingVerification verification;
        if (refused)
        {
            verification = new CodingVerification(false, task.Verify?.Count ?? 0, 0, $"refused: the agent changed the hidden tests' files {string.Join(", ", edits)}");
        }
        else
        {
            var gate = _run.Workspaces;
            await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                verification = await _run.Verifier.VerifyAsync(task, new CodingTools(workspace, _run.Tools with { CommandTimeout = _verifyLimit }, _run.Sandbox),
                    cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                gate.Release();
            }
        }

        _active = false;
        var info = Info();
        info[CodingInfo.Task] = task.Id;
        info[CodingInfo.VerifyPassed] = verification.Passed ? "true" : "false";
        info[CodingInfo.VerifyCommands] = verification.Commands.ToString(CultureInfo.InvariantCulture);
        info[CodingInfo.VerifyPassedCommands] = verification.PassedCommands.ToString(CultureInfo.InvariantCulture);
        info[CodingInfo.TestEdits] = string.Join(",", edits);
        info[CodingInfo.Outcome] = (verification.Passed ? CodingOutcome.Passed : truncated ? CodingOutcome.OutOfBudget : CodingOutcome.Failed).ToString();
        if (refused)
        {
            info[CodingInfo.Refused] = "test-edits";
        }

        if (!_run.Broken.IsEmpty)
        {
            info[CodingInfo.BrokenTasks] = _run.BrokenList;
        }

        _run.Variant?.AfterVerify(task, workspace, verification, info, settings.VariantOptions);

        Verification = verification;
        var scores = await TextRewards.ScoreAsync(settings.Reward, [new RewardInput(task.Prompt, last.Content, null, info)], settings.RewardOptions, cancellationToken)
            .ConfigureAwait(false);
        var observation = Observation();
        DeleteWorkspace();
        return new EnvironmentStep<ChatObservation>(observation, scores[0], !truncated, truncated, info);
    }

    /// <summary>The last episode's verification (its commands' output), or null before one ended.</summary>
    public CodingVerification? Verification { get; private set; }

    private ChatObservation Observation() => new([.. _messages], _run.Definitions, _run.Settings.Think);

    private Dictionary<string, string> Info() => new(StringComparer.Ordinal)
    {
        [CodingInfo.Rounds] = _rounds.ToString(CultureInfo.InvariantCulture),
        [CodingInfo.ToolCalls] = _calls.ToString(CultureInfo.InvariantCulture),
        [CodingInfo.ToolErrors] = _errors.ToString(CultureInfo.InvariantCulture),
    };

    // An episode under way is dropped (its workspace deleted unless kept).
    private void Abandon()
    {
        if (_active)
        {
            _active = false;
            DeleteWorkspace();
        }
    }

    private void DeleteWorkspace()
    {
        if (_tools is { } tools && !_run.Settings.Keep)
        {
            try
            {
                CodingWorkspace.Delete(tools.Workspace);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
            }
        }
    }

    /// <summary>Drops the episode under way (its workspace) and releases the shared copies when it is the last of its kind's run.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Abandon();
        _disposed = true;
        _run.Release();
    }
}
