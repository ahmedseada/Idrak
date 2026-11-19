// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

// The page drawer's numbers as settings: how a page is cut into bands, a shading's table, the largest tile, how a face
// found for a font the document does not embed is fitted, and how closely curves are flattened. Each keeps the value the
// drawer always had as its default; those an app or a bench run may change without code also have a variable, read
// through DocumentSettings when the options are made.
public sealed partial record DocumentDrawOptions
{
    /// <summary>The variable that sets <see cref="BandPixels"/> (1 to 67,108,864; 131,072 unless set): "IDRAK_DOCUMENT_BAND_PIXELS".</summary>
    public const string BandPixelsVariable = "IDRAK_DOCUMENT_BAND_PIXELS";

    /// <summary>The variable that sets <see cref="ShadingSteps"/> (2 to 1,048,576; 1,024 unless set): "IDRAK_DOCUMENT_SHADING_STEPS".</summary>
    public const string ShadingStepsVariable = "IDRAK_DOCUMENT_SHADING_STEPS";

    /// <summary>The variable that sets <see cref="MaxTilePixels"/> (1 to 67,108,864; 1,048,576 unless set): "IDRAK_DOCUMENT_MAX_TILE_PIXELS".</summary>
    public const string MaxTilePixelsVariable = "IDRAK_DOCUMENT_MAX_TILE_PIXELS";

    /// <summary>The variable that sets <see cref="FillFlatness"/> (pixels, 0.001 to 16; 1/32 unless set): "IDRAK_DOCUMENT_FILL_FLATNESS".</summary>
    public const string FillFlatnessVariable = "IDRAK_DOCUMENT_FILL_FLATNESS";

    /// <summary>The variable that sets <see cref="StrokeFlatness"/> (pixels, 0.001 to 16; 0.1 unless set): "IDRAK_DOCUMENT_STROKE_FLATNESS".</summary>
    public const string StrokeFlatnessVariable = "IDRAK_DOCUMENT_STROKE_FLATNESS";

    // A band's or a tile's pixels at most: its float buffers (16 bytes a pixel each) stay within what one array holds.
    private const int MostPixels = 1 << 26;
    private const int MostRows = 1 << 16;
    private const int MostSteps = 1 << 20;
    private const float LeastFlatness = 1e-3f, MostFlatness = 16f;

    private readonly int _bandPixels = DocumentSettings.Int(BandPixelsVariable, 1, MostPixels) ?? (1 << 17);
    private readonly int _minBandRows = 8;
    private readonly int _maxBandRows = 1024;
    private readonly int _shadingSteps = DocumentSettings.Int(ShadingStepsVariable, 2, MostSteps) ?? 1024;
    private readonly int _maxTilePixels = DocumentSettings.Int(MaxTilePixelsVariable, 1, MostPixels) ?? (1 << 20);
    private readonly float _minSubstituteStretch = 0.75f;
    private readonly float _maxSubstituteStretch = 4f / 3f;
    private readonly float _substituteSlant = 0.21f;
    private readonly float _substituteStroke = 0.03f;
    private readonly float _fillFlatness = DocumentSettings.Float(FillFlatnessVariable, LeastFlatness, MostFlatness) ?? 1f / 32;
    private readonly float _strokeFlatness = DocumentSettings.Float(StrokeFlatnessVariable, LeastFlatness, MostFlatness) ?? 0.1f;

    /// <summary>
    /// About how many pixels a band of the page holds (131,072, that is 128 Ki, unless <see cref="BandPixelsVariable"/>
    /// sets it; 1 to 67,108,864). A page is drawn a band of rows at a time, the bands spread over the cores: a band's rows
    /// are this many pixels over the page's width, within <see cref="MinBandRows"/> and <see cref="MaxBandRows"/>. Each
    /// band in work holds its coverage, colours and layers (about 40 bytes a pixel): more pixels a band, more memory a
    /// core and fewer bands to spread. The library's drawer draws the same pixels whatever the bands.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 1 to 67,108,864 (the message names the setting).</exception>
    public int BandPixels
    {
        get => _bandPixels;
        init => _bandPixels = value is >= 1 and <= MostPixels ? value : throw OutOfRange(nameof(BandPixels), value, "a whole number of pixels from 1 to 67108864");
    }

    /// <summary>
    /// The fewest rows a band holds (8 unless set; 1 to 65,536): a page so wide that <see cref="BandPixels"/> gives fewer
    /// rows takes this many a band. Drawing refuses options whose <see cref="MaxBandRows"/> is less.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 1 to 65,536 (the message names the setting).</exception>
    public int MinBandRows
    {
        get => _minBandRows;
        init => _minBandRows = value is >= 1 and <= MostRows ? value : throw OutOfRange(nameof(MinBandRows), value, "a whole number of rows from 1 to 65536");
    }

    /// <summary>
    /// The most rows a band holds (1,024 unless set; 1 to 65,536): a page so narrow that <see cref="BandPixels"/> gives
    /// more rows takes this many a band, so its bands still spread over the cores. At least <see cref="MinBandRows"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 1 to 65,536 (the message names the setting).</exception>
    public int MaxBandRows
    {
        get => _maxBandRows;
        init => _maxBandRows = value is >= 1 and <= MostRows ? value : throw OutOfRange(nameof(MaxBandRows), value, "a whole number of rows from 1 to 65536");
    }

    /// <summary>
    /// The steps of the table an axial or radial shading's colours are read through, read between them (1,024 unless
    /// <see cref="ShadingStepsVariable"/> sets it; 2 to 1,048,576): the shading's colour function is asked this many times
    /// when the page is made ready (12 bytes a step); fewer steps are quicker to make and coarser to read.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 2 to 1,048,576 (the message names the setting).</exception>
    public int ShadingSteps
    {
        get => _shadingSteps;
        init => _shadingSteps = value is >= 2 and <= MostSteps ? value : throw OutOfRange(nameof(ShadingSteps), value, "a whole number of steps from 2 to 1048576");
    }

    /// <summary>
    /// The most pixels the tile a tiling pattern's cell is drawn into may hold (1,048,576, that is 1 Mi, unless
    /// <see cref="MaxTilePixelsVariable"/> sets it; 1 to 67,108,864; 16 bytes a pixel). An upright tiling whose cell
    /// covers no more pixels on the page is drawn once at the page's scale and repeated snapped to whole pixels; a larger
    /// one, and any turned or slanted one, is drawn into a tile at the page's resolution made smaller to hold no more,
    /// and each pixel read back from it.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 1 to 67,108,864 (the message names the setting).</exception>
    public int MaxTilePixels
    {
        get => _maxTilePixels;
        init => _maxTilePixels = value is >= 1 and <= MostPixels ? value : throw OutOfRange(nameof(MaxTilePixels), value, "a whole number of pixels from 1 to 67108864");
    }

    /// <summary>
    /// How much narrower than its own a face found in <see cref="Fonts"/> for a font the document does not embed may draw
    /// a glyph: each glyph is stretched across to the width the document gives it, by a factor from this to
    /// <see cref="MaxSubstituteStretch"/> (0.75 unless set; more than 0, at most 1; 1: never narrower).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not more than 0 and at most 1 (the message names the setting).</exception>
    public float MinSubstituteStretch
    {
        get => _minSubstituteStretch;
        init => _minSubstituteStretch = value > 0f && value <= 1f ? value : throw OutOfRange(nameof(MinSubstituteStretch), value, "a factor more than 0 and at most 1");
    }

    /// <summary>
    /// How much wider than its own a face found in <see cref="Fonts"/> may draw a glyph, stretched to the width the
    /// document gives it (4/3 unless set; 1 to 100; 1: never wider). See <see cref="MinSubstituteStretch"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 1 to 100 (the message names the setting).</exception>
    public float MaxSubstituteStretch
    {
        get => _maxSubstituteStretch;
        init => _maxSubstituteStretch = value is >= 1f and <= 100f ? value : throw OutOfRange(nameof(MaxSubstituteStretch), value, "a factor from 1 to 100");
    }

    /// <summary>
    /// How far a face found in <see cref="Fonts"/> for an italic or oblique font (by its name) that is not slanted itself
    /// is slanted: across per unit up, the tangent of the angle (0.21, about 12 degrees, unless set; 0 to 1; 0: not
    /// slanted).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 0 to 1 (the message names the setting).</exception>
    public float SubstituteSlant
    {
        get => _substituteSlant;
        init => _substituteSlant = value is >= 0f and <= 1f ? value : throw OutOfRange(nameof(SubstituteSlant), value, "a slant from 0 to 1");
    }

    /// <summary>
    /// How wide a face found in <see cref="Fonts"/> for a bold font (by its name) that is not bold itself is stroked in its
    /// own colour, as a share of the font's size (0.03 unless set; 0 to 0.25; 0: not stroked).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 0 to 0.25 (the message names the setting).</exception>
    public float SubstituteStroke
    {
        get => _substituteStroke;
        init => _substituteStroke = value is >= 0f and <= 0.25f ? value : throw OutOfRange(nameof(SubstituteStroke), value, "a share of the font's size from 0 to 0.25");
    }

    /// <summary>
    /// How far, in pixels, the lines a curve of a fill, a glyph or a clip is flattened into may stray from it (1/32 of a
    /// pixel unless <see cref="FillFlatnessVariable"/> sets it; 0.001 to 16): the outline rasterizer is asked for it (one
    /// that does not take it flattens by its own measure). Coarser is quicker, finer closer to the curve.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 0.001 to 16 (the message names the setting).</exception>
    public float FillFlatness
    {
        get => _fillFlatness;
        init => _fillFlatness = value is >= LeastFlatness and <= MostFlatness ? value : throw OutOfRange(nameof(FillFlatness), value, "a number of pixels from 0.001 to 16");
    }

    /// <summary>
    /// How far, in pixels, the lines a stroked curve is flattened into, and the polygon of a round join, cap or dot, may
    /// stray from the true curve (0.1 unless <see cref="StrokeFlatnessVariable"/> sets it; 0.001 to 16).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not within 0.001 to 16 (the message names the setting).</exception>
    public float StrokeFlatness
    {
        get => _strokeFlatness;
        init => _strokeFlatness = value is >= LeastFlatness and <= MostFlatness ? value : throw OutOfRange(nameof(StrokeFlatness), value, "a number of pixels from 0.001 to 16");
    }

    private static ArgumentOutOfRangeException OutOfRange(string setting, object value, string range) =>
        new(setting, value, $"DocumentDrawOptions.{setting} is {range}.");
}
