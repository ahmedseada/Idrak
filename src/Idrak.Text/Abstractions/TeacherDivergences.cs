// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Text.Abstractions;

/// <summary>
/// A divergence between a student's next-token distribution p and a teacher's q, row by row, that distillation trains
/// on (<c>FineTuningLossInput.TeacherDivergence</c>, <c>FineTuningLosses.OnPolicyDistillation</c>), registered in
/// <see cref="TeacherDivergences"/> under <see cref="Name"/>. It runs a chunk of trained rows at a time, without
/// gradients: the trainer turns its gradient with respect to the student's softened logits into the network's with one
/// more pass of the output head over the rows, so the [tokens, vocabulary] logits are never stored. Thread-safe: it keeps
/// no state.
/// </summary>
public interface ITeacherDivergence
{
    /// <summary>The divergence's name in <see cref="TeacherDivergences"/> ("forward", "reverse", "jsd").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>Whether it reads a weight β in (0, 1) (the generalized Jensen-Shannon divergence's).</summary>
    bool UsesBeta { get; }

    /// <summary>
    /// Each row's divergence [rows] from the student's log-probabilities log p [rows, vocabulary] (log_softmax of the
    /// logits over the temperature, u = z / T) and the teacher's probabilities q [rows, vocabulary] at the same
    /// temperature, both on one device; and when <paramref name="gradient"/> is set, its gradient with respect to u
    /// [rows, vocabulary] (the trainer scales each row by its weight and 1 / T). Called without gradients; new tensors.
    /// </summary>
    /// <param name="studentLogProbabilities">log p of the rows [rows, vocabulary].</param>
    /// <param name="teacherProbabilities">q of the rows [rows, vocabulary] (zeros allowed).</param>
    /// <param name="beta">The weight when <see cref="UsesBeta"/>, in (0, 1); unused otherwise.</param>
    /// <param name="gradient">Whether the gradient is wanted.</param>
    (Tensor Values, Tensor? Gradient) Rows(Tensor studentLogProbabilities, Tensor teacherProbabilities, float beta, bool gradient);
}

/// <summary>
/// The divergences distillation trains on, by name (ignoring case). The library's:
/// <list type="bullet">
/// <item><see cref="Forward"/> ("forward"): KL(q ‖ p) = Σ q (log q − log p), the teacher's mass covered (Hinton et al.'s
/// distillation, <c>FineTuningLosses.Distillation</c>); gradient p − q.</item>
/// <item><see cref="Reverse"/> ("reverse"): KL(p ‖ q) = Σ p (log p − log q), mode-seeking (the usual on-policy choice:
/// the student is penalized where it puts mass the teacher does not); gradient p ⊙ (f − KL), f = log p − log q.</item>
/// <item><see cref="JensenShannon"/> ("jsd"): the generalized Jensen-Shannon divergence of Agarwal et al. 2024 (as TRL's
/// GKD computes it), β KL(q ‖ m) + (1 − β) KL(p ‖ m) with m = β q + (1 − β) p, β in (0, 1) (near 0 it behaves as β times
/// the forward KL, near 1 as (1 − β) times the reverse one); gradient p ⊙ (a − Σ p a), a = (1 − β) (log p − log m).</item>
/// </list>
/// A zero of q is read as 1e-30 inside a logarithm (q · log q is 0 there). Register another with <see cref="Register"/>;
/// one under a library name shadows the library's, which <see cref="Unregister"/> brings back.
/// </summary>
public static class TeacherDivergences
{
    /// <summary>The forward KL(q ‖ p).</summary>
    public const string Forward = "forward";

    /// <summary>The reverse KL(p ‖ q).</summary>
    public const string Reverse = "reverse";

    /// <summary>The generalized Jensen-Shannon divergence of weight β.</summary>
    public const string JensenShannon = "jsd";

    private static readonly ITeacherDivergence LibraryForward = new ForwardDivergence();

    private static readonly SlotTable<string, ITeacherDivergence> Table = BuiltIn();

    private static SlotTable<string, ITeacherDivergence> BuiltIn()
    {
        var table = new SlotTable<string, ITeacherDivergence>(nameof(TeacherDivergences), (slot, app, library) => new GuardedDivergence(slot, app, library),
            StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(Forward, LibraryForward);
        table.RegisterDefault(Reverse, new ReverseDivergence());
        table.RegisterDefault(JensenShannon, new JensenShannonDivergence());
        return table;
    }

    /// <summary>Registers <paramref name="divergence"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITeacherDivergence divergence)
    {
        ArgumentNullException.ThrowIfNull(divergence);
        ArgumentException.ThrowIfNullOrWhiteSpace(divergence.Name);
        Table.Register(divergence.Name, divergence, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's divergence of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The divergence of that name, or null.</summary>
    public static ITeacherDivergence? Find(string name) => Table.Find(name);

    /// <summary>The divergence of that name; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static ITeacherDivergence Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(TeacherDivergences), name, Table.Keys,
            $"Teacher divergence '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TeacherDivergences.Register.");

    /// <summary>The library's divergence of that name, whatever an app registered over it; null when the library has none.</summary>
    public static ITeacherDivergence? Default(string name) => Table.Default(name);

    /// <summary>Who registered the divergence of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's divergence of that name fails (<see cref="SlotPolicy.Throw"/> unless set).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered divergence as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(d => $"{d.Name}: {d.Summary}"));

    // Whether `kind` is the library's forward KL itself (not an app's over it): the chunked pass's own path, as the
    // distillation before the registry computed it.
    internal static bool IsLibraryForward(ITeacherDivergence kind) => ReferenceEquals(kind, LibraryForward);

    // The rule the chunked passes compute `kind` with (null: their own forward KL).
    internal static TokenDivergenceRule? Rule(ITeacherDivergence? kind, float beta) =>
        kind is null || IsLibraryForward(kind) ? null : (student, teacher, gradient) => kind.Rows(student, teacher, beta, gradient);

    // An app's divergence over the library's: each call falls back on its own, or is compared with the library's.
    private sealed class GuardedDivergence(Slot slot, ITeacherDivergence app, ITeacherDivergence library) : ITeacherDivergence
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public bool UsesBeta => app.UsesBeta;

        public (Tensor Values, Tensor? Gradient) Rows(Tensor studentLogProbabilities, Tensor teacherProbabilities, float beta, bool gradient) =>
            slot.Call(() => app.Rows(studentLogProbabilities, teacherProbabilities, beta, gradient), () => library.Rows(studentLogProbabilities, teacherProbabilities, beta, gradient),
                (a, b) => Comparisons.Difference(b.Values.ToArray(), a.Values.ToArray(), 1e-5f));
    }

    // Each row's sum over the vocabulary repeated along it [rows, vocabulary]: an outer product with a row of ones (no broadcasting).
    internal static Tensor AlongRows(Tensor perRow, int rows, int vocabulary) =>
        perRow.Reshape(rows, 1).MatMul(Tensor.Full([1, vocabulary], 1f, perRow.Device));

    private sealed class ForwardDivergence : ITeacherDivergence
    {
        public string Name => Forward;

        public string Summary => "KL(teacher ‖ student): covers the teacher's mass (Hinton et al. 2015's distillation); gradient p − q";

        public bool UsesBeta => false;

        public (Tensor Values, Tensor? Gradient) Rows(Tensor studentLogProbabilities, Tensor teacherProbabilities, float beta, bool gradient)
        {
            var (logp, q) = (studentLogProbabilities, teacherProbabilities);
            var values = (q * (q.Maximum(1e-30f).Log() - logp)).Sum(1);
            return (values, gradient ? logp.Exp() - q : null);
        }
    }

    private sealed class ReverseDivergence : ITeacherDivergence
    {
        public string Name => Reverse;

        public string Summary => "KL(student ‖ teacher): mode-seeking, the usual on-policy choice (Agarwal et al. 2024); gradient p ⊙ (log p − log q − KL)";

        public bool UsesBeta => false;

        public (Tensor Values, Tensor? Gradient) Rows(Tensor studentLogProbabilities, Tensor teacherProbabilities, float beta, bool gradient)
        {
            var (logp, q) = (studentLogProbabilities, teacherProbabilities);
            int rows = logp.Shape[0], vocabulary = logp.Shape[^1];
            var p = logp.Exp();
            var f = logp - q.Maximum(1e-30f).Log();
            var values = (p * f).Sum(1);
            return (values, gradient ? p * (f - AlongRows(values, rows, vocabulary)) : null);
        }
    }

    private sealed class JensenShannonDivergence : ITeacherDivergence
    {
        public string Name => JensenShannon;

        public string Summary => "the generalized Jensen-Shannon divergence β KL(q ‖ m) + (1 − β) KL(p ‖ m), m = β q + (1 − β) p (Agarwal et al. 2024, TRL's GKD)";

        public bool UsesBeta => true;

        public (Tensor Values, Tensor? Gradient) Rows(Tensor studentLogProbabilities, Tensor teacherProbabilities, float beta, bool gradient)
        {
            if (!(beta > 0f && beta < 1f))
            {
                throw new ArgumentOutOfRangeException(nameof(beta), beta, "The Jensen-Shannon divergence's beta is in (0, 1).");
            }

            var (logp, q) = (studentLogProbabilities, teacherProbabilities);
            int rows = logp.Shape[0], vocabulary = logp.Shape[^1];
            var p = logp.Exp();
            var logm = (q * beta + p * (1f - beta)).Maximum(1e-30f).Log();
            var studentPart = logp - logm;
            var values = (q * (q.Maximum(1e-30f).Log() - logm)).Sum(1) * beta + (p * studentPart).Sum(1) * (1f - beta);
            if (!gradient)
            {
                return (values, null);
            }

            // d/dp_v = (1 − β) log(p_v / m_v) (the q terms' derivatives cancel), then through the softmax.
            var a = studentPart * (1f - beta);
            var mean = (p * a).Sum(1);
            return (values, p * (a - AlongRows(mean, rows, vocabulary)));
        }
    }
}
