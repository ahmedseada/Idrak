// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>Global switches for automatic differentiation.</summary>
public static class Autograd
{
    [ThreadStatic]
    private static int t_disabledDepth;

    /// <summary>Whether operations on the current thread record the graph needed for <see cref="Tensor.Backward()"/>.</summary>
    public static bool IsEnabled => t_disabledDepth == 0;

    /// <summary>
    /// Turns gradient recording off on this thread until the returned scope is disposed.
    /// Use it for inference and evaluation: it is faster and allocates less.
    /// </summary>
    /// <example><c>using (Autograd.NoGrad()) { var prediction = model.Forward(x); }</c></example>
    public static NoGradScope NoGrad()
    {
        t_disabledDepth++;
        return new NoGradScope(true);
    }

    [ThreadStatic]
    private static int t_firstPassDepth;

    [ThreadStatic]
    private static List<int>? t_recordedChoices;

    [ThreadStatic]
    private static Queue<int>? t_replayedChoices;

    /// <summary>
    /// Whether a pass the backward pass runs again with recording (an activation checkpoint's first pass,
    /// <see cref="FirstPass"/>) runs on this thread: operations then compute as they do while recording (the kernels and
    /// paths a gradient needs, no inference-only ones), so the kept output is the one the recompute gives, bit for bit.
    /// </summary>
    public static bool IsFirstPass => t_firstPassDepth > 0;

    /// <summary>
    /// Turns gradient recording off on this thread until disposed, for a pass the backward pass runs again with recording
    /// (activation checkpointing): operations compute as they do while recording (<see cref="IsFirstPass"/>), and each
    /// choice that hangs on the device's state at the time (a measured path, the memory free) goes into
    /// <paramref name="choices"/>, which <see cref="Replay"/> hands to the recompute in the same order.
    /// </summary>
    /// <param name="choices">Receives the pass's choices, in order.</param>
    public static ChoiceScope FirstPass(List<int> choices)
    {
        ArgumentNullException.ThrowIfNull(choices);
        var scope = new ChoiceScope(t_recordedChoices, t_replayedChoices, firstPass: true);
        t_disabledDepth++;
        t_firstPassDepth++;
        t_recordedChoices = choices;                                       // a recompute's queue goes on being read
        return scope;
    }

    /// <summary>
    /// Until disposed, the operations on this thread that make a choice <see cref="FirstPass"/> records take
    /// <paramref name="choices"/> in order instead of choosing again (the recompute of a checkpointed pass).
    /// </summary>
    /// <param name="choices">What the first pass recorded.</param>
    public static ChoiceScope Replay(IReadOnlyList<int> choices)
    {
        ArgumentNullException.ThrowIfNull(choices);
        var scope = new ChoiceScope(t_recordedChoices, t_replayedChoices, firstPass: false);
        t_recordedChoices = null;
        t_replayedChoices = new Queue<int>(choices);
        return scope;
    }

    // The choice a recompute replays, if one is left to replay on this thread.
    internal static bool TryReplayChoice(out int choice)
    {
        if (t_replayedChoices is { Count: > 0 } replayed)
        {
            choice = replayed.Dequeue();
            return true;
        }

        choice = 0;
        return false;
    }

    // Keeps a choice for the recompute while a first pass records on this thread.
    internal static void RecordChoice(int choice) => t_recordedChoices?.Add(choice);

    /// <summary>Restores what <see cref="FirstPass"/> or <see cref="Replay"/> changed when disposed.</summary>
    public readonly struct ChoiceScope : IDisposable
    {
        private readonly List<int>? _recorded;
        private readonly Queue<int>? _replayed;
        private readonly bool _firstPass;

        internal ChoiceScope(List<int>? recorded, Queue<int>? replayed, bool firstPass) => (_recorded, _replayed, _firstPass) = (recorded, replayed, firstPass);

        /// <inheritdoc />
        public void Dispose()
        {
            if (_firstPass)
            {
                t_disabledDepth--;
                t_firstPassDepth--;
            }

            t_recordedChoices = _recorded;
            t_replayedChoices = _replayed;
        }
    }

    /// <summary>
    /// Defines an operation with its own forward and backward steps, written with public tensor operations (see
    /// <see cref="DifferentiableFunction"/>). Call <see cref="DifferentiableFunction.Apply"/> to run it.
    /// </summary>
    /// <param name="name">The operation's name (for telemetry and error messages).</param>
    /// <param name="forward">Computes the result from the inputs. It runs without recording gradients.</param>
    /// <param name="backward">
    /// Given the inputs, the forward result and the gradient of the loss with respect to that result, returns the gradient
    /// with respect to each input, in order and of the input's shape (null for an input that needs none). It runs without
    /// recording gradients; the tensors it creates are disposed after the gradients are added to the inputs'.
    /// </param>
    /// <example>
    /// <code>
    /// var softplus = Autograd.Function("softplus", x => (x[0].Exp() + 1f).Log(), (x, y, g) => [g * x[0].Sigmoid()]);
    /// </code>
    /// </example>
    public static DifferentiableFunction Function(string name, Func<IReadOnlyList<Tensor>, Tensor> forward,
        Func<IReadOnlyList<Tensor>, Tensor, Tensor, IReadOnlyList<Tensor?>> backward) => new(name, forward, backward);

    [ThreadStatic]
    private static Action<Tensor, bool>? t_gradientReady;

    [ThreadStatic]
    private static int t_backwardDepth;

    /// <summary>
    /// Until the returned scope is disposed, <paramref name="ready"/> is called on this thread for each tensor that
    /// requires gradients and was created directly (a parameter) once a backward pass on this thread finished its
    /// gradient: every operation of that pass that read it has passed its gradient back. The second argument is true in
    /// the outermost backward pass and false in one run inside another's backward step (a checkpointed block's), whose
    /// parameters the outer pass may still add to. A data-parallel run reduces finished gradients while the backward pass
    /// goes on (plan 23, phase C4). The listener before the scope answers again after it.
    /// </summary>
    /// <param name="ready">Called with the parameter and whether the pass is the outermost.</param>
    public static IDisposable OnGradientReady(Action<Tensor, bool> ready)
    {
        ArgumentNullException.ThrowIfNull(ready);
        var scope = new ReadyScope(t_gradientReady);
        t_gradientReady = ready;
        return scope;
    }

    // A leaf's gradient is finished in the backward pass now running on this thread.
    internal static void GradientFinished(Tensor leaf)
    {
        if (t_gradientReady is { } ready)
        {
            ready(leaf, t_backwardDepth == 1);
        }
    }

    // Whether anyone listens for finished gradients on this thread (one static read in the backward loop).
    internal static bool ListensForGradients => t_gradientReady is not null;

    // A backward pass begins (true) or ends (false) on this thread: the depth tells an outer pass from one nested in it.
    internal static void Backward(bool begins) => t_backwardDepth += begins ? 1 : -1;

    private sealed class ReadyScope(Action<Tensor, bool>? previous) : IDisposable
    {
        private bool _ended;

        public void Dispose()
        {
            if (!_ended)
            {
                _ended = true;
                t_gradientReady = previous;
            }
        }
    }

    /// <summary>Restores gradient recording when disposed.</summary>
    public readonly struct NoGradScope : IDisposable
    {
        private readonly bool _active;

        internal NoGradScope(bool active) => _active = active;

        /// <inheritdoc />
        public void Dispose()
        {
            if (_active)
            {
                t_disabledDepth--;
            }
        }
    }
}
