// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;

namespace Idrak.Vision;

/// <summary>
/// Reads the text of a page with a character classifier: <see cref="PageSegmenter"/> finds the lines and characters,
/// <see cref="GlyphFrame"/> frames every character into one reused batch buffer, the model classifies them in batches,
/// and the text is put together line by line.
/// </summary>
/// <remarks>
/// <para>For several scripts in one model (Latin and Arabic, say), each line takes the script most of its probability is
/// on and every character on it is read within that script, so look-alikes across scripts (V and ٧, l and ا) are told
/// apart by their neighbours. Scripts and directions come from the class names' Unicode blocks
/// (<see cref="WritingScript.Of(string)"/>).</para>
/// <para>Within a word, a digit among letters becomes the letter it looks like (1 → I, ١ → ا) and a letter among digits
/// the digit (O → 0), else the likeliest of the other kind; a word of a cased script takes the case of most of its
/// letters. Right-to-left lines are returned in reading order, numbers left to right.</para>
/// <code>
/// using var ocr = TextRecognizer.Load("letters.ikm").Build();     // a package from Predictor.Save
/// RecognizedPage page = ocr.Read("scan.png");
/// Console.WriteLine(page.Text);
/// </code>
/// </remarks>
public sealed class TextRecognizer : IDisposable
{
    private readonly Module _model;
    private readonly bool _ownsModel;
    private readonly Device _device;
    private readonly string[] _classes;
    private readonly int[] _scriptOf;          // class -> index into _scripts
    private readonly bool[] _isDigit;
    private readonly WritingScript[] _scripts;
    private readonly Dictionary<string, int> _index;
    private readonly Dictionary<int, int> _letterFor;   // digit class -> letter class
    private readonly Dictionary<int, int> _digitFor;    // letter class -> digit class
    private readonly TextRecognizerBuilder _settings;

    internal TextRecognizer(TextRecognizerBuilder settings)
    {
        _settings = settings;
        _model = settings.Model;
        _ownsModel = settings.OwnsModel;
        _device = settings.Device ?? _model.Parameters().FirstOrDefault()?.Device ?? Device.Default;
        _classes = [.. settings.ClassNames ?? throw new InvalidOperationException("Give the model's classes in output order: TextRecognizer.For(model).Characters(...).")];
        _index = new Dictionary<string, int>(_classes.Length);
        for (int i = 0; i < _classes.Length; i++)
        {
            _index.TryAdd(_classes[i], i);
        }

        var scripts = _classes.Select(WritingScript.Of).ToArray();
        _scripts = [.. scripts.Distinct()];
        _scriptOf = [.. scripts.Select(s => Array.IndexOf(_scripts, s))];
        _isDigit = [.. _classes.Select(TextOrder.IsNumber)];
        _letterFor = Pairs(settings.LettersForDigits);
        _digitFor = Pairs(settings.DigitsForLetters);

        Dictionary<int, int> Pairs(IEnumerable<KeyValuePair<string, string>> map) =>
            map.Where(p => _index.ContainsKey(p.Key) && _index.ContainsKey(p.Value)
                           && WritingScript.Of(p.Key) == WritingScript.Of(p.Value))
               .ToDictionary(p => _index[p.Key], p => _index[p.Value]);
    }

    /// <summary>Starts a recognizer for a character classifier: images [N, 1, size, size] in, one logit per class out.</summary>
    public static TextRecognizerBuilder For(Module model) => new(model ?? throw new ArgumentNullException(nameof(model)), ownsModel: false);

    /// <summary>
    /// Starts a recognizer from a package written by <see cref="Predictor{TIn, TOut}.Save"/>: its model, class names and
    /// input shape (which sets the character size). The recognizer disposes the model.
    /// </summary>
    public static TextRecognizerBuilder Load(string path, Device? device = null)
    {
        var settings = Predictor.Load(path, device).Settings;
        var builder = new TextRecognizerBuilder(settings.Model!, ownsModel: true) { Device = device };
        if (settings.StoredClasses is { } classes)
        {
            builder.Characters(classes);
        }

        if (settings.InputShape is [1, int h, int w] && h == w)
        {
            builder.GlyphSize(h);
        }

        return builder;
    }

    /// <summary>The model's classes, in output order.</summary>
    public IReadOnlyList<string> Classes => _classes;

    /// <summary>The scripts of the classes, in order of first appearance.</summary>
    public IReadOnlyList<WritingScript> Scripts => _scripts;

    /// <summary>Reads an image file (PNG, BMP, PGM, PPM, or a registered codec's format).</summary>
    public RecognizedPage Read(string path) => Read(ImageCodecs.Decode(path));

    /// <summary>Reads a page image (any channels; colour is read as grey; dark on light or light on dark).</summary>
    public RecognizedPage Read(ImageData page) => Read(PageSegmenter.Segment(page, _settings.SegmentationOptions));

    /// <summary>Reads a page segmented already (to reuse or adjust a <see cref="PageSegmenter"/> result).</summary>
    public RecognizedPage Read(PageLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        var probabilities = Classify(layout);
        int classes = _classes.Length, g = 0;
        var lines = new List<RecognizedLine>(layout.Lines.Count);
        foreach (var line in layout.Lines)
        {
            lines.Add(ReadLine(line, probabilities.AsSpan(g * classes, line.Glyphs.Count * classes)));
            g += line.Glyphs.Count;
        }

        return new RecognizedPage(layout.Width, layout.Height, lines);
    }

    /// <summary>Disposes the model when the recognizer loaded it (<see cref="Load"/>).</summary>
    public void Dispose()
    {
        if (_ownsModel)
        {
            _model.Dispose();
        }
    }

    // Every character's class probabilities, [glyphs, classes]: frames go into one buffer of BatchSize frames, reused.
    private float[] Classify(PageLayout layout)
    {
        int size = _settings.Size, frame = size * size, classes = _classes.Length, count = layout.GlyphCount;
        var probabilities = new float[count * classes];
        if (count == 0)
        {
            return probabilities;
        }

        int batch = Math.Min(_settings.Batch, count);
        var buffer = ArrayPool<float>.Shared.Rent(batch * frame);
        try
        {
            var boxes = layout.Lines.SelectMany(l => l.Glyphs).Select(r => r.Box).ToArray();
            for (int start = 0; start < count; start += batch)
            {
                int n = Math.Min(batch, count - start);
                for (int i = 0; i < n; i++)
                {
                    GlyphFrame.Extract(layout, boxes[start + i], buffer.AsSpan(i * frame, frame), size, _settings.Border);
                }

                using var x = Tensor.From(buffer.AsSpan(0, n * frame), [n, 1, size, size], _device);
                using var logits = _model.Predict(x);
                if (logits.Size != n * classes)
                {
                    throw new InvalidOperationException($"The model gave {logits.Size / n} outputs per character for {classes} classes.");
                }

                using var p = logits.Softmax();
                p.ToArray().CopyTo(probabilities, start * classes);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }

        return probabilities;
    }

    private RecognizedLine ReadLine(TextLineRegion line, ReadOnlySpan<float> probabilities)
    {
        int classes = _classes.Length, glyphs = line.Glyphs.Count;

        // The line's script: the one most of its probability is on (all scripts allowed when there is one, or when turned off).
        int script = -1;
        if (_settings.ScriptPerLine && _scripts.Length > 1)
        {
            Span<double> mass = stackalloc double[_scripts.Length];
            mass.Clear();
            for (int i = 0; i < probabilities.Length; i++)
            {
                mass[_scriptOf[i % classes]] += probabilities[i];
            }

            script = 0;
            for (int s = 1; s < mass.Length; s++)
            {
                if (mass[s] > mass[script])
                {
                    script = s;
                }
            }
        }

        var chosen = new int[glyphs];
        for (int g = 0; g < glyphs; g++)
        {
            chosen[g] = Best(probabilities.Slice(g * classes, classes), script, digit: null);
        }

        if (_settings.WordContext)
        {
            for (int start = 0; start < glyphs;)
            {
                int end = start + 1;
                while (end < glyphs && !line.Glyphs[end].SpaceBefore)
                {
                    end++;
                }

                ApplyWordContext(chosen.AsSpan(start, end - start), probabilities.Slice(start * classes, (end - start) * classes), script);
                start = end;
            }
        }

        var lineScript = script >= 0 ? _scripts[script] : MajorityScript(chosen);
        var characters = new RecognizedCharacter[glyphs];
        var words = new List<string>();
        var word = new System.Text.StringBuilder();
        for (int g = 0; g < glyphs; g++)
        {
            var p = probabilities.Slice(g * classes, classes);
            characters[g] = new RecognizedCharacter(_classes[chosen[g]], p[chosen[g]], line.Glyphs[g].Box, Top(p, 3));
            if (g > 0 && line.Glyphs[g].SpaceBefore)
            {
                words.Add(word.ToString());
                word.Clear();
            }

            word.Append(_classes[chosen[g]]);
        }

        words.Add(word.ToString());
        if (_settings.WordContext && lineScript.HasCase)
        {
            for (int w = 0; w < words.Count; w++)
            {
                words[w] = OneCase(words[w]);
            }
        }

        return new RecognizedLine(TextOrder.Reorder(words, lineScript.RightToLeft), lineScript, line.Box, characters);
    }

    // A word's characters kept to letters or to digits, whichever most of them are: look-alikes first, else the likeliest.
    private void ApplyWordContext(Span<int> word, ReadOnlySpan<float> probabilities, int script)
    {
        int digits = 0;
        foreach (int c in word)
        {
            digits += _isDigit[c] ? 1 : 0;
        }

        bool? keepDigits = digits * 2 < word.Length ? false : digits * 2 > word.Length ? true : null;
        if (keepDigits is not bool wantDigits)
        {
            return;
        }

        int classes = _classes.Length;
        for (int i = 0; i < word.Length; i++)
        {
            if (_isDigit[word[i]] == wantDigits)
            {
                continue;
            }

            var map = wantDigits ? _digitFor : _letterFor;
            word[i] = map.TryGetValue(word[i], out int lookAlike) && (script < 0 || _scriptOf[lookAlike] == script)
                ? lookAlike
                : Best(probabilities.Slice(i * classes, classes), script, wantDigits);
        }
    }

    // The likeliest class within a script (-1: any) and kind (null: any).
    private int Best(ReadOnlySpan<float> p, int script, bool? digit)
    {
        int best = -1;
        for (int c = 0; c < p.Length; c++)
        {
            if ((script < 0 || _scriptOf[c] == script) && (digit is null || _isDigit[c] == digit) && (best < 0 || p[c] > p[best]))
            {
                best = c;
            }
        }

        if (best >= 0 || digit is null)
        {
            return Math.Max(best, 0);
        }

        return Best(p, script, digit: null);   // the script has no class of that kind: any of its classes
    }

    private WritingScript MajorityScript(int[] chosen) =>
        chosen.Length == 0 ? WritingScript.Common : _scripts[chosen.GroupBy(c => _scriptOf[c]).MaxBy(g => g.Count())!.Key];

    private IReadOnlyList<CharacterCandidate> Top(ReadOnlySpan<float> p, int count)
    {
        var top = new List<CharacterCandidate>(count);
        Span<bool> taken = p.Length <= 1024 ? stackalloc bool[p.Length] : new bool[p.Length];
        taken.Clear();
        for (int k = 0; k < Math.Min(count, p.Length); k++)
        {
            int best = -1;
            for (int c = 0; c < p.Length; c++)
            {
                if (!taken[c] && (best < 0 || p[c] > p[best]))
                {
                    best = c;
                }
            }

            taken[best] = true;
            top.Add(new CharacterCandidate(_classes[best], p[best]));
        }

        return top;
    }

    private static string OneCase(string word)
    {
        int upper = 0, letters = 0;
        foreach (char c in word)
        {
            if (char.IsLetter(c))
            {
                letters++;
                upper += char.IsUpper(c) ? 1 : 0;
            }
        }

        return upper * 2 >= letters ? word.ToUpperInvariant() : word.ToLowerInvariant();
    }
}

/// <summary>Sets up a <see cref="TextRecognizer"/>; every setting has a working default.</summary>
public sealed class TextRecognizerBuilder
{
    internal TextRecognizerBuilder(Module model, bool ownsModel)
    {
        Model = model;
        OwnsModel = ownsModel;
    }

    internal Module Model { get; }

    internal bool OwnsModel { get; }

    internal IReadOnlyList<string>? ClassNames { get; private set; }

    internal Device? Device { get; set; }

    internal int Size { get; private set; } = 28;

    internal int Border { get; private set; } = 1;

    internal int Batch { get; private set; } = 512;

    internal SegmentationOptions SegmentationOptions { get; private set; } = new();

    internal bool ScriptPerLine { get; private set; } = true;

    internal bool WordContext { get; private set; } = true;

    internal Dictionary<string, string> LettersForDigits { get; } = new()
    {
        ["0"] = "O", ["1"] = "I", ["2"] = "Z", ["5"] = "S", ["6"] = "G", ["8"] = "B",
        ["١"] = "ا", ["٥"] = "ه",
    };

    internal Dictionary<string, string> DigitsForLetters { get; } = new()
    {
        ["O"] = "0", ["o"] = "0", ["D"] = "0", ["I"] = "1", ["l"] = "1", ["L"] = "1", ["Z"] = "2", ["S"] = "5", ["s"] = "5",
        ["G"] = "6", ["b"] = "6", ["B"] = "8", ["g"] = "9", ["q"] = "9",
        ["ا"] = "١", ["ه"] = "٥",
    };

    /// <summary>The model's classes, in output order (one character each: "A", "7", "ب"...). Required for <see cref="TextRecognizer.For"/>.</summary>
    public TextRecognizerBuilder Characters(IReadOnlyList<string> classes)
    {
        ArgumentNullException.ThrowIfNull(classes);
        if (classes.Count == 0 || classes.Any(string.IsNullOrEmpty))
        {
            throw new ArgumentException("Every class needs its character.", nameof(classes));
        }

        ClassNames = classes;
        return this;
    }

    /// <summary>The device the model runs on (default: where its weights are).</summary>
    public TextRecognizerBuilder OnDevice(Device device)
    {
        Device = device;
        return this;
    }

    /// <summary>The model's input: size x size characters, <paramref name="border"/> pixels from the edge (default 28 and 1, as EMNIST).</summary>
    public TextRecognizerBuilder GlyphSize(int size, int border = 1)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(size, 4);
        Size = size;
        Border = border is >= 0 && 2 * border < size ? border : throw new ArgumentOutOfRangeException(nameof(border));
        return this;
    }

    /// <summary>Characters classified per batch (default 512): a larger batch is faster on a GPU and holds more memory.</summary>
    public TextRecognizerBuilder BatchSize(int glyphs)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(glyphs, 1);
        Batch = glyphs;
        return this;
    }

    /// <summary>How pages are split into lines and characters.</summary>
    public TextRecognizerBuilder Segmentation(SegmentationOptions options)
    {
        SegmentationOptions = options ?? throw new ArgumentNullException(nameof(options));
        return this;
    }

    /// <summary>Whether each line is read within one script, the likeliest (default true; matters only for models of several scripts).</summary>
    public TextRecognizerBuilder OneScriptPerLine(bool enabled = true)
    {
        ScriptPerLine = enabled;
        return this;
    }

    /// <summary>Whether words are kept to letters or to digits and to one case (default true).</summary>
    public TextRecognizerBuilder UseWordContext(bool enabled = true)
    {
        WordContext = enabled;
        return this;
    }

    /// <summary>Adds or replaces a look-alike: in a word of letters, <paramref name="digit"/> reads as <paramref name="letter"/>.</summary>
    public TextRecognizerBuilder LetterFor(string digit, string letter)
    {
        LettersForDigits[digit] = letter;
        return this;
    }

    /// <summary>Adds or replaces a look-alike: in a number, <paramref name="letter"/> reads as <paramref name="digit"/>.</summary>
    public TextRecognizerBuilder DigitFor(string letter, string digit)
    {
        DigitsForLetters[letter] = digit;
        return this;
    }

    /// <summary>Creates the recognizer.</summary>
    public TextRecognizer Build() => new(this);
}
