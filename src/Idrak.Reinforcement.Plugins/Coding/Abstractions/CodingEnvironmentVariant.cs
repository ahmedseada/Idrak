// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;
using Idrak.Text.Abstractions;

namespace Idrak.Reinforcement.Plugins.Coding.Abstractions;

/// <summary>
/// A variant of the <c>coding</c> environment for one coding language (the C# plug-in's <c>csharp</c>): an environment
/// kind of its own <see cref="IKind.Name"/> (<see cref="CodingEnvironmentKind"/>, made with the variant) that is the coding
/// environment with the language's defaults (its setup and verification for a task that names none, its programs and
/// prompt lines) and defaults for the coding options (<see cref="Defaults"/>: the reward, the protected files). A variant
/// with options of its own implements <see cref="IVariantOptions"/>; one that reads the workspace around verification
/// into the episode's info (for its rewards) implements <see cref="IVerificationHooks"/>. It copies nothing of the coding
/// environment.
/// </summary>
public interface ICodingEnvironmentVariant : IKind
{
    /// <summary>The coding language (a <see cref="CodingLanguages"/> name) whose defaults the tasks get, and the coding option <c>languages</c> unless given.</summary>
    string Language { get; }

    /// <summary>Coding options it sets when the caller does not (for example <c>reward</c> and <c>protect</c>); none by default.</summary>
    IReadOnlyDictionary<string, string> Defaults() => CodingEnvironmentVariants.NoOptions;

    /// <summary>
    /// The task as the variant runs it: by default with <paramref name="language"/>'s setup and verification
    /// (<see cref="CodingEnvironmentVariants.WithLanguageDefaults"/>) for what it does not name.
    /// </summary>
    CodingTask Task(CodingTask task, ICodingLanguage language, IReadOnlyDictionary<string, string> options) => CodingEnvironmentVariants.WithLanguageDefaults(task, language);
}

/// <summary>
/// A coding variant with options of its own (<see cref="IConfigurableKind.Keys"/>, beside the coding environment's): its
/// keys, the check of their values, and the variables every command of a run gets from them, declared together so they
/// agree.
/// </summary>
public interface IVariantOptions : ICodingEnvironmentVariant, IConfigurableKind
{
    /// <summary>Checks its own options (<paramref name="options"/>: those of <see cref="IConfigurableKind.Keys"/> given), once per run; throws <see cref="ArgumentException"/> naming a bad one.</summary>
    void Check(IReadOnlyDictionary<string, string> options);

    /// <summary>Variables every command of a run gets (the agent's tools, setup and verification), over the languages' own.</summary>
    IReadOnlyDictionary<string, string> Variables(IReadOnlyDictionary<string, string> options);
}

/// <summary>A coding variant that works on the workspace around an episode's verification, and reads what it left into the episode's info.</summary>
public interface IVerificationHooks : ICodingEnvironmentVariant
{
    /// <summary>Called before an episode's verification (a refused one too), in its workspace: for example to remove result files the agent could have left.</summary>
    void BeforeVerify(string workspace, IReadOnlyDictionary<string, string> options);

    /// <summary>
    /// Called after an episode's verification (or its refusal: <paramref name="verification"/> then did not run), before
    /// the reward: adds what it reads from <paramref name="workspace"/> to <paramref name="info"/>, under keys its rewards read.
    /// </summary>
    void AfterVerify(CodingTask task, string workspace, CodingVerification verification, EpisodeInfo info, IReadOnlyDictionary<string, string> options);
}

/// <summary>What coding variants share.</summary>
public static class CodingEnvironmentVariants
{
    /// <summary>No options.</summary>
    public static IReadOnlyDictionary<string, string> NoOptions { get; } = new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary><paramref name="task"/> with <paramref name="language"/>'s setup and verification (<see cref="ICodingLanguage.DefaultSetup"/>, <see cref="ICodingLanguage.DefaultVerify"/>) for what it does not name.</summary>
    public static CodingTask WithLanguageDefaults(CodingTask task, ICodingLanguage language)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(language);
        return task with { Setup = task.Setup ?? language.DefaultSetup, Verify = task.Verify ?? language.DefaultVerify };
    }
}
