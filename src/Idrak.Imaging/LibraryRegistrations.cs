// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

[assembly: LibraryRegistrations(typeof(Idrak.Imaging.LibraryRegistrations))]

namespace Idrak.Imaging;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the image codecs (png, jpeg, bmp,
/// netpbm) and encoders (png, jpeg, netpbm), Pillow's image transforms byte for byte, the flip and shift augmentations (Idrak.Vision adds the others) and
/// the "images" sample source, the "opentype", "cff" and "type1" font formats and the font mapper, the "truetype" font writer, the library's font maps (legacy layouts, as data), the "coverage"
/// outline rasterizer and page drawer, the "shaped" document text setter, the reading orders (bidi, runs), the page layouts (letter, plain, shapes), the shapes, shape profiles, pens and text
/// bindings the "shapes" layout draws with, and the glyph poses every layout writes its text with. <c>LibraryDefaults.Ensure</c> (in
/// Idrak.Abstraction) calls <see cref="RegisterFor"/> once
/// per registry, before that registry is first used.
/// </summary>
internal sealed class LibraryRegistrations : ILibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(ImageTransforms)] = () =>
        {
            LibraryImageTransforms.RegisterDefaults();   // Pillow's operations, byte for byte
            ScanTransforms.RegisterDefaults();           // scan degradations (plan 18)
        },
        [typeof(ScanProfiles)] = ScanTransforms.RegisterProfiles,
        [typeof(Augmentations)] = LibraryAugmentations.RegisterDefaults,       // flip and shift (Idrak.Vision adds the others)
        [typeof(ImageCodecs)] = RegisterImageCodecs,
        [typeof(ImageEncoders)] = () =>                                          // png, jpeg, netpbm
        {
            ImageEncoders.Register(new PngEncoder());
            ImageEncoders.Register(new JpegImageEncoder());
            ImageEncoders.Register(new NetpbmEncoder());
        },                            // png, jpeg, bmp, netpbm (asked newest first)
        [typeof(SampleSources)] = () => SampleSources.Register(new ImagesSource()),
        [typeof(FontFormats)] = () =>                                            // TrueType and CFF outlines (single fonts, collections); bare CFF and Type 1 programs
        {
            FontFormats.Register(new OpenTypeFormat());
            FontFormats.Register(new CffFontFormat());
            FontFormats.Register(new Type1FontFormat());
        },
        [typeof(FontWriters)] = () => FontWriters.Register(new TrueTypeWriter()),            // .ttf, whole and subsets, the format Idrak works from
        [typeof(FontMappers)] = () => FontMappers.Register(new OpenTypeFontMapper()),       // a map's forms, ligatures and marks as OpenType tables
        [typeof(FontMaps)] = RegisterFontMaps,                                   // the legacy layouts the library ships, as data
        [typeof(OutlineRasterizers)] = () => OutlineRasterizers.Register(new CoverageRasterizer()),
        [typeof(DocumentPageDrawers)] = () => DocumentPageDrawers.Register(new CoverageDocumentDrawer()),   // "coverage": any document's pages to pixels
        [typeof(DocumentTextSetters)] = () => DocumentTextSetters.Register(new ShapedTextSetter()),        // "shaped": a form field's text set in any script
        [typeof(OutlineTracers)] = () => OutlineTracers.Register(new ContourTracer()),     // contour (plan 38, F1)
        [typeof(FontMakers)] = () => FontMakers.Register(new TracedFontMaker()),           // traced (plan 38, F3)
        [typeof(TextShapers)] = RegisterShapers,                                 // general for every script, arabic for the joining scripts
        [typeof(ReadingOrders)] = () =>                                          // bidi (UAX #9), runs
        {
            ReadingOrders.Register(new Shaping.BidiReadingOrder());
            ReadingOrders.Register(new Shaping.RunsReadingOrder());
        },
        [typeof(PageLayouts)] = () =>                                            // letter, plain, shapes
        {
            PageLayouts.Register(new LetterLayout());
            PageLayouts.Register(new PlainLayout());
            PageLayouts.Register(new ShapesLayout());
        },
        [typeof(Shapes)] = RegisterShapes,                                       // line, polyline, polygon, box, grid, circle, ellipse, arc, arrow, fill, hatch, text; ring-text, lettering, star, shield, scribble
        [typeof(ShapeProfiles)] = RegisterShapeProfiles,                         // none, grid, ruled, boxed, text-grid-text, framed, circled; the letter's emblems, stamps and signature
        [typeof(Pens)] = () =>                                                   // ideal, ballpoint
        {
            Pens.Register(new IdealPen());
            Pens.Register(new BallpointPen());
        },
        [typeof(TextBindings)] = () => TextBindings.Register(new SequenceBinding()),
        [typeof(GlyphPoses)] = () =>                                             // none, drift
        {
            GlyphPoses.Register(new NonePose());
            GlyphPoses.Register(new DriftPose());
        },
    };

    private static void RegisterShapes()
    {
        foreach (var shape in (IShape[])[new LineShape(), new PolylineShape(), new PolygonShape(), new BoxShape(), new GridShape(), new CircleShape(), new EllipseShape(),
                     new ArcShape(), new ArrowShape(), new FillShape(), new HatchShape(), new TextShape(), new RingTextShape(), new LetteringShape(), new StarShape(), new ShieldShape(),
                     new ScribbleShape()])
        {
            Shapes.Register(shape);
        }
    }

    // Generic page looks, each written as a profile.
    private static void RegisterShapeProfiles()
    {
        ShapeProfiles.Register(new ShapeProfile("none", "", "no shapes: the text set in rows, nothing drawn"));
        ShapeProfiles.Register(new ShapeProfile("grid", "grid=auto x auto", "a grid fitted to the text: a row a line, a column a unit of its longest line"));
        ShapeProfiles.Register(new ShapeProfile("ruled", "grid=auto x 1,inner=rows,border=bottom", "a line under every text line: a region a line"));
        ShapeProfiles.Register(new ShapeProfile("boxed", "grid=auto x 1,inner=off", "a box around the text: a region a line"));
        ShapeProfiles.Register(new ShapeProfile("text-grid-text", "text,lines=2,flow=down,grid=auto x auto,flow=down,text,lines=1,flow=down",
            "two lines of paragraphs, a grid of the lines between (a row a line), a closing line of paragraph, stacked down the page"));
        ShapeProfiles.Register(new ShapeProfile("framed", "box,holds=paragraphs,flow=down,pad=0.6", "a box around every line of the text, set as wrapped paragraphs"));
        ShapeProfiles.Register(new ShapeProfile("circled", "circle,holds=paragraphs,flow=down,pad=0.3", "a ring with the text written inside it, its lines shortened to the ring"));
        Ornaments.Register();
    }

    private static void RegisterImageCodecs()
    {
        foreach (var codec in (IImageCodec[])[new NetpbmCodec(), new BmpCodec(), new JpegCodec(), new PngCodec()])
        {
            ImageCodecs.Register(codec);
        }
    }

    // The library's font maps: every Resources/font-maps/*.json, read once when FontMaps is first used, in name order
    // (FontMaps asks the newest first: the last registered is tried first).
    private static void RegisterFontMaps()
    {
        const string prefix = "Idrak.Imaging.font-maps.";
        var assembly = typeof(LibraryRegistrations).Assembly;
        foreach (string name in assembly.GetManifestResourceNames().Where(n => n.StartsWith(prefix, StringComparison.Ordinal)).Order(StringComparer.Ordinal))
        {
            using var stream = assembly.GetManifestResourceStream(name)!;
            var bytes = new byte[stream.Length];
            stream.ReadExactly(bytes);
            FontMaps.Register(FontMap.Parse(bytes, "Idrak.Imaging/" + name[prefix.Length..]));
        }
    }

    private static void RegisterShapers()
    {
        TextShapers.Register(TextShapers.AnyScript, new Shaping.OpenTypeShaper());
        var arabic = new Shaping.ArabicShaper();
        foreach (string script in Shaping.ArabicShaper.Scripts)
        {
            TextShapers.Register(script, arabic);
        }
    }

    private static readonly OptionKeys ImageOptions = new("channels", "height", "width");

    // "images": ImageFolderSource over class folders; options channels, height, width (each the first image's by default).
    // "images": ImageFolderSource over class folders; channels, height and width default to the first image's.
    private sealed class ImagesSource : ISampleSourceKind
    {
        public string Name => "images";

        public string Summary => "ImageFolderSource over class folders (channels, height, width: the first image's unless set)";

        public IReadOnlySet<string> Keys => ImageOptions;

        public ISampleSource Create(SampleSourceRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            var options = request.Options;
            foreach (string key in options.Keys)
            {
                if (!Known(key))
                {
                    throw new ArgumentException($"The images source has no option '{key}' ({ImageOptions}).");
                }
            }

            int? Number(string name) => options.TryGetValue(name, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : null;
            return ImageFolderSource.Open(request.Path, Number("channels"), Number("height"), Number("width"));
        }

        // Option names ignore case.
        private static bool Known(string key)
        {
            foreach (string name in ImageOptions.Ordered)
            {
                if (string.Equals(name, key, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }
    }

    public void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
