// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Plugins;

/// <summary>
/// The usual way to write an <see cref="IIdrakPlugin"/>: <see cref="RegisterParts"/> registers each part with
/// <see cref="Part"/> (how to register it and how to take it back), and the base keeps the rest: <see cref="Register"/>
/// runs once however often it is called (and from any thread), a part that fails takes back the parts before it, and
/// <see cref="Unregister"/> takes every part back in reverse order, so the registries are left as they were found.
/// </summary>
/// <example>
/// <code>
/// internal sealed class WhisperRegistration() : IdrakPlugin("whisper")
/// {
///     protected override void RegisterParts() =>
///         Part(() => SpeechFamilies.Register(new WhisperFamily()), () => SpeechFamilies.Unregister(WhisperFamily.Architecture));
/// }
/// </code>
/// </example>
public abstract class IdrakPlugin : IIdrakPlugin
{
    private readonly Lock _gate = new();

    // What takes each registered part back, in the order they were registered; null while not registered.
    private volatile List<Action>? _undo;

    // The parts RegisterParts has registered so far (only while it runs).
    private List<Action>? _registering;

    /// <summary>A plug-in named <paramref name="name"/>.</summary>
    protected IdrakPlugin(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public bool IsRegistered => _undo is not null;

    /// <inheritdoc/>
    public void Register()
    {
        lock (_gate)
        {
            if (_undo is not null || _registering is not null)
            {
                return;                                              // registered already, or registering (a part that asks for its own plug-in)
            }

            var parts = new List<Action>();
            _registering = parts;
            try
            {
                RegisterParts();
            }
            catch (Exception e)
            {
                _registering = null;
                Undo(parts, e);
                throw;
            }

            _registering = null;
            _undo = parts;
        }
    }

    /// <inheritdoc/>
    public void Unregister()
    {
        lock (_gate)
        {
            if (_undo is not { } parts)
            {
                return;
            }

            _undo = null;
            Undo(parts, null);
        }
    }

    /// <summary>Registers the plug-in's parts, each through <see cref="Part"/> (or <see cref="Requires"/>).</summary>
    protected abstract void RegisterParts();

    /// <summary>
    /// Registers one part (<paramref name="register"/>) and keeps <paramref name="unregister"/>, which takes it back on
    /// <see cref="Unregister"/> (or when a later part fails). Called from <see cref="RegisterParts"/> only.
    /// </summary>
    protected void Part(Action register, Action unregister)
    {
        ArgumentNullException.ThrowIfNull(register);
        ArgumentNullException.ThrowIfNull(unregister);
        var parts = _registering ?? throw new InvalidOperationException($"{GetType().Name}: Part is called from RegisterParts only.");
        register();
        parts.Add(unregister);
    }

    /// <summary>
    /// Registers the plug-in this one builds on (a vision family on its text decoder's family) through
    /// <see cref="IdrakPlugins"/>, unless it is registered already; <see cref="Unregister"/> removes it again only when
    /// this plug-in registered it. Called from <see cref="RegisterParts"/> only.
    /// </summary>
    protected void Requires(IIdrakPlugin plugin)
    {
        ArgumentNullException.ThrowIfNull(plugin);
        if (plugin.IsRegistered)
        {
            return;
        }

        Part(() => IdrakPlugins.Register(plugin), () => IdrakPlugins.Unregister(plugin));
    }

    // Takes the parts back, the last first; every one is tried, and their failures are thrown together (after the one
    // that made a registration fail, when there is one, which the caller rethrows).
    private static void Undo(List<Action> parts, Exception? cause)
    {
        List<Exception>? failures = null;
        for (int i = parts.Count - 1; i >= 0; i--)
        {
            try
            {
                parts[i]();
            }
            catch (Exception e)
            {
                (failures ??= []).Add(e);
            }
        }

        if (failures is not null && cause is null)
        {
            throw new AggregateException("A plug-in part could not be taken back.", failures);
        }
    }
}
