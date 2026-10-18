// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// A listener told as each layer's forward begins on this thread (plan 23, phase C6): a sharded run whose ranks hold only
/// their share of the weights (<c>ClusterOptions.Shard</c> = parameters) gathers a layer's weights just before its forward
/// and frees them after. <see cref="Module.Forward"/> enters every module; a layer whose forward reads other layers'
/// weights in one fused pass (a projection with its normalization) enters those too. With no listener each entry is one
/// thread-static read.
/// </summary>
public static class ForwardHooks
{
    [ThreadStatic]
    private static Func<Module, IDisposable?>? t_enter;

    /// <summary>
    /// Until the returned scope is disposed, <paramref name="enter"/> is called on this thread as each module's forward
    /// begins (<see cref="Enter"/>); what it returns, if anything, is disposed when that forward ends. The listener before
    /// the scope answers again after it.
    /// </summary>
    /// <param name="enter">Called with the module whose forward begins; returns a scope to end with it, or null.</param>
    public static IDisposable OnEnter(Func<Module, IDisposable?> enter)
    {
        ArgumentNullException.ThrowIfNull(enter);
        var scope = new EnterScope(t_enter);
        t_enter = enter;
        return scope;
    }

    /// <summary>Whether a listener is set on this thread.</summary>
    public static bool Listening => t_enter is not null;

    /// <summary>
    /// <paramref name="module"/>'s forward begins on this thread (or a fused pass reads its weights): the listener's scope
    /// to dispose when it ends, or null (no listener, or nothing to do).
    /// </summary>
    public static IDisposable? Enter(Module module) => t_enter?.Invoke(module);

    private sealed class EnterScope(Func<Module, IDisposable?>? previous) : IDisposable
    {
        private bool _ended;

        public void Dispose()
        {
            if (!_ended)
            {
                _ended = true;
                t_enter = previous;
            }
        }
    }
}
