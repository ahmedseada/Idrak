// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// How one glyph is drawn off its ideal place, as a pen would: moved (<see cref="Dx"/>, <see cref="Dy"/>, pixels, y
/// down), scaled, sheared (a slant: x moves by <see cref="Shear"/> times the height above the baseline) and turned
/// (<see cref="Rotation"/>, radians, counterclockwise as seen), all about the glyph's pen origin on the baseline, and drawn
/// at <see cref="Ink"/> of full ink.
/// </summary>
/// <param name="Dx">The move to the right, in pixels.</param>
/// <param name="Dy">The move down, in pixels.</param>
/// <param name="Scale">The size, a ratio (1: as set).</param>
/// <param name="Shear">The slant: x moves by this times the height above the baseline (0: upright).</param>
/// <param name="Rotation">The turn, in radians, counterclockwise as seen (0: none).</param>
/// <param name="Ink">How much ink, 0 to 1 (1: full).</param>
public readonly record struct GlyphPose(float Dx, float Dy, float Scale, float Shear, float Rotation, float Ink)
{
    /// <summary>The glyph as set: no move, scale 1, no slant or turn, full ink.</summary>
    public static GlyphPose Identity => new(0, 0, 1, 0, 0, 1);

    /// <summary>Whether this is <see cref="Identity"/>.</summary>
    public bool IsIdentity => this == Identity;
}

/// <summary>
/// The most a glyph pose moves a glyph for given settings (<see cref="IGlyphPose.Bounds"/>), so a layout keeps posed text
/// inside its box and a check can hold the pose to it.
/// </summary>
/// <param name="Move">The most |dx| or |dy|, in ems of the text size.</param>
/// <param name="Scale">The most |scale - 1|.</param>
/// <param name="Shear">The most |shear|.</param>
/// <param name="Rotation">The most |rotation|, in radians.</param>
/// <param name="Ink">The least ink (1: always full).</param>
public readonly record struct GlyphPoseBounds(float Move, float Scale, float Shear, float Rotation, float Ink)
{
    /// <summary>No move at all: the bounds of the identity.</summary>
    public static GlyphPoseBounds None => new(0, 0, 0, 0, 1);

    /// <summary>
    /// The most any point of a glyph moves, in pixels, for text of <paramref name="size"/> pixels per em whose outline
    /// points lie within <paramref name="reach"/> pixels of the glyph's pen origin (a first-order bound, with room).
    /// </summary>
    public float Reach(float size, float reach) => Move * size + (Scale + Shear + Rotation) * reach * 1.1f;
}

/// <summary>What a glyph pose is asked for one glyph (one cluster: every glyph of a cluster is drawn with the pose of its first).</summary>
/// <param name="Line">The line's running number on the page (each line its own pose along it).</param>
/// <param name="X">How far along the line the cluster is, in pixels from the line's left (drift follows distance, not count).</param>
/// <param name="Cluster">The cluster: its first character's index in its paragraph.</param>
/// <param name="Size">The text's size, in pixels per em.</param>
/// <param name="Seed">The page's seed for poses.</param>
public readonly record struct GlyphPoseInput(int Line, float X, int Cluster, float Size, int Seed);

/// <summary>
/// A glyph pose's settings, read once per page from its options (<see cref="IGlyphPose.Read"/>): the options as given,
/// and whatever a pose parses them into (a pose's own settings derive from this class).
/// </summary>
public class GlyphPoseSettings
{
    /// <summary>Settings of <paramref name="options"/> (none when null).</summary>
    public GlyphPoseSettings(IReadOnlyDictionary<string, string>? options = null) => Options = options ?? PageOptions.None;

    /// <summary>No options: a pose's defaults.</summary>
    public static GlyphPoseSettings None { get; } = new();

    /// <summary>The options as given, by key.</summary>
    public IReadOnlyDictionary<string, string> Options { get; }
}

/// <summary>
/// The stage from set text to written text: each glyph drawn a little differently, as a hand writes (a slow drift along
/// the line, a slant, a little noise in size and ink), or exactly (<c>none</c>). Its settings are read once per page
/// (<see cref="Read"/>) and declare their bounds (<see cref="Bounds"/>); then each cluster of each line gets its pose
/// (<see cref="Pose"/>), the same for the same input, within the bounds. Joined script stays joined: between
/// neighbouring clusters the move, slant and turn change only smoothly with the distance along the line. Register one
/// with <see cref="GlyphPoses.Register"/>.
/// </summary>
public interface IGlyphPose : IConfigurableKind
{
    /// <summary>The pose's settings from <paramref name="options"/> (its own keys, <see cref="IConfigurableKind.Keys"/>; defaults for those not given).</summary>
    /// <exception cref="ArgumentException">A key it does not take (the message names its keys), or a bad value.</exception>
    GlyphPoseSettings Read(IReadOnlyDictionary<string, string> options);

    /// <summary>The most a pose drawn with <paramref name="settings"/> moves a glyph.</summary>
    GlyphPoseBounds Bounds(GlyphPoseSettings settings);

    /// <summary>The pose of the cluster <paramref name="glyph"/> describes, within <see cref="Bounds"/>.</summary>
    GlyphPose Pose(in GlyphPoseInput glyph, GlyphPoseSettings settings);
}

/// <summary>
/// The glyph poses, by name (ignoring case). Built in (Idrak.Imaging): <c>none</c> (every glyph as set; the default) and
/// <c>drift</c> (a slow drift along the line and a little noise per cluster; its keys pose-slant, pose-size,
/// pose-baseline, pose-rotate and pose-pressure). Register another with <see cref="Register"/>; one under a library name
/// shadows the library's, which <see cref="Unregister"/> brings back, guarded by its <see cref="SetPolicy">policy</see>.
/// </summary>
public static class GlyphPoses
{
    /// <summary>The pose that draws every glyph as set.</summary>
    public const string None = "none";

    private static readonly KindTable<IGlyphPose> Table = new(nameof(GlyphPoses), "Glyph pose", "add it with GlyphPoses.Register",
        (slot, app, library) => new GuardedPose(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static GlyphPoses() => LibraryDefaults.Ensure(typeof(GlyphPoses));

    /// <summary>Registers <paramref name="pose"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IGlyphPose pose) => Table.Register(pose, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's pose <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The pose registered as <paramref name="name"/>, or null.</summary>
    public static IGlyphPose? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The pose registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IGlyphPose Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's pose <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IGlyphPose? Default(string name) => Table.Default(name);

    /// <summary>Who registered the pose <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's pose <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> poses with the library's; <see cref="SlotPolicy.Shadow"/> lets the library's pose and
    /// compares the app's on a sample of glyphs).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered pose as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The keys of every registered pose, each once, in registration order.</summary>
    public static IReadOnlyList<string> AllKeys() => [.. Table.Kinds.SelectMany(p => p.Keys).Distinct(StringComparer.OrdinalIgnoreCase)];

    // The app's settings and the library's, read from the same options (the library's null when it refuses them).
    private sealed class GuardedSettings(GlyphPoseSettings app, GlyphPoseSettings? library, IReadOnlyDictionary<string, string> options) : GlyphPoseSettings(options)
    {
        public GlyphPoseSettings App => app;

        public GlyphPoseSettings? Library => library;
    }

    private sealed class GuardedPose(Slot slot, IGlyphPose app, IGlyphPose library) : IGlyphPose
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public GlyphPoseSettings Read(IReadOnlyDictionary<string, string> options)
        {
            var own = app.Read(options);
            GlyphPoseSettings? theirs;
            try
            {
                theirs = library.Read(PageOptions.Only(options, library.Keys));
            }
            catch (ArgumentException)
            {
                theirs = null;
            }

            return new GuardedSettings(own, theirs, options);
        }

        public GlyphPoseBounds Bounds(GlyphPoseSettings settings) => app.Bounds(settings is GuardedSettings g ? g.App : settings);

        public GlyphPose Pose(in GlyphPoseInput glyph, GlyphPoseSettings settings)
        {
            var (own, theirs) = settings is GuardedSettings g ? (g.App, g.Library) : (settings, settings);
            var input = glyph;
            return theirs is null
                ? app.Pose(input, own)
                : slot.Call(() => app.Pose(input, own), () => library.Pose(input, theirs), (a, b) => Comparisons.Exact(a, b));
        }
    }
}
