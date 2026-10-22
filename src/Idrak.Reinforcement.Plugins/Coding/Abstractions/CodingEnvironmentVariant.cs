// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Text.Abstractions;

namespace Idrak.Reinforcement.Plugins.Coding.Abstractions;

/// <summary>
/// A variant of the <c>coding</c> environment for one coding language (the C# plug-in's <c>csharp</c>): an environment
/// kind of its own name (<see cref="CodingEnvironmentKind"/>, made with the variant) that
/// is the coding environment with the language's defaults (its setup and verification for a task that names none, its
/// programs and prompt lines), defaults for the coding options (<see cref="Defaults"/>: the reward, the protected files),
/// options of its own (<see cref="Keys"/>), variables for every command, and what it reads from the workspace after
/// verification into the episode's info (for its rewards). It copies nothing of the coding environment.
/// </summary>
public abstract class CodingEnvironmentVariant
{
    /// <summary>The environment kind's name (in <c>EnvironmentKinds</c>).</summary>
    public abstract string Name { get; }

    /// <summary>One line on the kind (its <c>IEnvironmentKind.Summary</c>).</summary>
    public abstract string Summary { get; }

    /// <summary>The coding language (a <see cref="CodingLanguages"/> name) whose defaults the tasks get, and the coding option <c>languages</c> unless given.</summary>
    public abstract string Language { get; }

    /// <summary>Its own options, beside the coding environment's (none by default).</summary>
    public virtual IReadOnlyCollection<string> Keys => [];

    /// <summary>Coding options it sets when the caller does not (for example <c>reward</c> and <c>protect</c>); none by default.</summary>
    public virtual IReadOnlyDictionary<string, string> Defaults() => new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>Checks its own options (<paramref name="options"/>: those of <see cref="Keys"/> given), once per run; throws <see cref="ArgumentException"/> naming a bad one.</summary>
    public virtual void Check(IReadOnlyDictionary<string, string> options)
    {
    }

    /// <summary>Variables every command of a run gets (the agent's tools, setup and verification), over the languages' own; none by default.</summary>
    public virtual IReadOnlyDictionary<string, string> Variables(IReadOnlyDictionary<string, string> options) => new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>
    /// The task as the variant runs it: by default with <paramref name="language"/>'s setup and verification
    /// (<see cref="ICodingLanguage.DefaultSetup"/>, <see cref="ICodingLanguage.DefaultVerify"/>) for what it does not name.
    /// </summary>
    public virtual CodingTask Task(CodingTask task, ICodingLanguage language, IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(language);
        return task with { Setup = task.Setup ?? language.DefaultSetup, Verify = task.Verify ?? language.DefaultVerify };
    }

    /// <summary>Called before an episode's verification (a refused one too), in its workspace: for example to remove result files the agent could have left. Nothing by default.</summary>
    public virtual void BeforeVerify(string workspace, IReadOnlyDictionary<string, string> options)
    {
    }

    /// <summary>
    /// Called after an episode's verification (or its refusal: <paramref name="verification"/> then did not run), before
    /// the reward: adds what it reads from <paramref name="workspace"/> to <paramref name="info"/>. Nothing by default.
    /// </summary>
    public virtual void AfterVerify(CodingTask task, string workspace, CodingVerification verification, IDictionary<string, string> info, IReadOnlyDictionary<string, string> options)
    {
    }
}
