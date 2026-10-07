// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

// Sample sources, batch sources and the built-in loaders: each trains to the same weights as an in-memory Dataset of
// the same samples, and the image codecs decode what independent encoders here write.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] DataLoaderGroup =
    [
        ("data loaders: a Dataset and any other source give the same batches (shuffle, dropLast, prefetch); Batch is public", LoaderAnySource),
        ("data loaders: views (subset, shuffle, split, concat) read the right samples and train like the copied Dataset", LoaderViews),
        ("data loaders: the streamed CSV source reads Dataset.LoadCsv's samples, in order as a stream, and trains to the same weights", LoaderCsv),
        ("data loaders: PNG (every depth, palette, interlaced), BMP and Netpbm decode as independent encoders wrote them", LoaderCodecs),
        ("data loaders: the image folder source with transforms trains like its in-memory Dataset; codecs register by name", LoaderImageFolder),
        ("data loaders: image transforms flip, shift, rotate and add noise as computed directly, seeded by epoch and sample", LoaderTransforms),
        ("data loaders: token files and .npy arrays are memory-mapped and train like the same samples in memory", LoaderMappedFiles),
        ("data loaders: a stream is batched in order or shuffled in a buffer, counted after an epoch, and trains like a Dataset", LoaderStreams),
        ("data loaders: SampleSources opens the built-ins by name, registers, unregisters and lists the names in errors", LoaderRegistry),
        ("data loaders: TableSamples reads JSON Lines, Parquet and CSV columns (arrays, classes) and streams them", LoaderTables),
    ];

    // A source that only forwards to a dataset, so the loader cannot take any Dataset shortcut.
    private sealed class Forwarding(Dataset data) : ISampleSource
    {
        public int Count => data.Count;

        public IReadOnlyList<int> FeatureShape => data.FeatureShape;

        public IReadOnlyList<int> TargetShape => data.TargetShape;

        public void Read(int index, Span<float> features, Span<float> targets) => data.Read(index, features, targets);
    }

    // The samples of a dataset in order, as a stream.
    private sealed class DatasetStream(Dataset data) : ISampleStream
    {
        public IReadOnlyList<int> FeatureShape => data.FeatureShape;

        public IReadOnlyList<int> TargetShape => data.TargetShape;

        public int Opened { get; private set; }

        public ISampleReader Open()
        {
            Opened++;
            return new Reader(data);
        }

        private sealed class Reader(Dataset data) : ISampleReader
        {
            private int _next;

            public bool Read(Span<float> features, Span<float> targets)
            {
                if (_next >= data.Count)
                {
                    return false;
                }

                data.Read(_next++, features, targets);
                return true;
            }

            public void Dispose()
            {
            }
        }
    }

    private static Dataset RandomData(int count, int features, int targets, int seed)
    {
        var random = new Random(seed);
        var x = new float[count * features];
        var y = new float[count * targets];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = random.NextSingle() * 2 - 1;
        }

        for (int i = 0; i < y.Length; i++)
        {
            y[i] = random.NextSingle();
        }

        return Dataset.FromFlat(x, y, count, [.. Enumerable.Range(0, features).Select(i => $"x{i}")], [.. Enumerable.Range(0, targets).Select(i => $"y{i}")]);
    }

    // Every batch of an epoch, as (features, targets, size) on the host.
    private static List<(float[] X, float[] Y, int Size, int[] Shape)> Epoch(IEnumerable<Batch> loader)
    {
        var batches = new List<(float[], float[], int, int[])>();
        foreach (var batch in loader)
        {
            using (batch)
            {
                batches.Add((batch.Features.ToArray(), batch.Targets.ToArray(), batch.Size, batch.Features.Shape.ToArray()));
            }
        }

        return batches;
    }

    private static void SameEpoch(List<(float[] X, float[] Y, int Size, int[] Shape)> a, List<(float[] X, float[] Y, int Size, int[] Shape)> b, string what)
    {
        Check(a.Count == b.Count, $"{what}: {a.Count} batches, expected {b.Count}");
        for (int i = 0; i < a.Count; i++)
        {
            Check(a[i].Size == b[i].Size && a[i].Shape.SequenceEqual(b[i].Shape), $"{what}: batch {i} has {a[i].Size} samples [{string.Join(", ", a[i].Shape)}]");
            AssertClose(b[i].X, a[i].X, 0, $"{what}: batch {i} features");
            AssertClose(b[i].Y, a[i].Y, 0, $"{what}: batch {i} targets");
        }
    }

    // Trains a flatten + linear model for a few epochs and returns its weights, so two loaders can be compared.
    private static float[] TrainedWeights(IBatchSource loader, int inputs, int outputs, Device device, int epochs = 3)
    {
        using var model = new Sequential { new Flatten(), new Linear(inputs, outputs, device: device, random: new Random(7)) };
        using var optimizer = new Adam(model.Parameters(), 0.01f);
        using var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError);
        var history = trainer.Fit(loader, epochs);
        Check(history.Epochs.Count == epochs && double.IsFinite(history.Epochs[^1].Loss), "training ran every epoch");
        return [.. model.Parameters().SelectMany(p => p.ToArray())];
    }

    private static void SameTraining(ISampleSource source, Dataset memory, Device device, string what, IReadOnlyList<ISampleTransform>? transforms = null, int batch = 8)
    {
        int inputs = memory.FeatureCount, outputs = memory.TargetCount;
        var fromSource = TrainedWeights(new DataLoader(source, batch, shuffle: true, device: device, seed: 11) { Transforms = transforms ?? [] }, inputs, outputs, device);
        var fromMemory = TrainedWeights(new DataLoader(memory, batch, shuffle: true, device: device, seed: 11) { Transforms = transforms ?? [] }, inputs, outputs, device);
        AssertClose(fromMemory, fromSource, 1e-6f, $"{what}: weights trained from the source and from memory");
    }

    private static void LoaderAnySource(Device device)
    {
        var data = RandomData(103, 5, 2, 1).WithFeatureShape(5);
        var source = new Forwarding(data);
        foreach (var (batch, dropLast) in new[] { (16, false), (16, true), (103, false), (200, false) })
        {
            var a = Epoch(new DataLoader(data, batch, shuffle: true, dropLast: dropLast, device: device, seed: 3));
            var b = Epoch(new DataLoader(source, batch, shuffle: true, dropLast: dropLast, device: device, seed: 3));
            SameEpoch(b, a, $"batch {batch}, dropLast {dropLast}");
        }

        // Batches large enough to be gathered on a worker thread.
        var large = RandomData(300, 120, 20, 2);
        var loader = new DataLoader(new Forwarding(large), 128, shuffle: true, device: device, seed: 5);
        Check(loader.BatchCount == 3 && loader.SampleCount == 300 && loader.Source is Forwarding && loader.Dataset is null, "the loader's counts and source");
        IBatchSource counted = loader;
        Check(counted.BatchCount == 3 && counted.SampleCount == 300 && counted.BatchSize == 128, "the batch source counts");
        SameEpoch(Epoch(loader), Epoch(new DataLoader(large, 128, shuffle: true, device: device, seed: 5)), "prefetched batches");
        SameTraining(source, data, device, "forwarding source");
        Check(Trainer2D(data, device).SequenceEqual(Trainer2D(source, device)), "Trainer.Predict on a source");

        // The public Batch constructor.
        using (var x = Tensor.From(new float[6], [3, 2], device))
        using (var y = Tensor.From(new float[3], [3, 1], device))
        {
            var made = new Batch(x, y, 4);
            Check(made.Size == 3 && made.Index == 4, "a batch made outside the loader");
        }

        Throws<ArgumentException>(() =>
        {
            using var x = Tensor.From(new float[6], [3, 2], device);
            using var y = Tensor.From(new float[2], [2, 1], device);
            _ = new Batch(x, y);
        }, "features and targets with different sample counts");
    }

    private static float[] Trainer2D(ISampleSource data, Device device)
    {
        using var model = new Sequential { new Linear(5, 2, device: device, random: new Random(1)) };
        using var optimizer = new Sgd(model.Parameters(), 0.1f);
        using var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError);
        return [.. trainer.Predict(data, 16).Cast<float>()];
    }

    private static void LoaderViews(Device device)
    {
        var data = RandomData(50, 3, 1, 4);
        ISampleSource source = new Forwarding(data);
        var (train, test) = source.Split(0.7, seed: 9);
        var (copiedTrain, copiedTest) = data.Split(0.7, seed: 9, removeDuplicates: false);
        Check(train.Count == copiedTrain.Count && test.Count == copiedTest.Count, "split sizes");
        AssertClose(copiedTrain.Features.ToArray(), train.ToDataset().Features.ToArray(), 0, "the training view holds Dataset.Split's samples");
        AssertClose(copiedTest.Targets.ToArray(), test.ToDataset().Targets.ToArray(), 0, "the test view holds Dataset.Split's samples");
        SameTraining(train, copiedTrain, device, "split view");

        var shuffled = source.Shuffle(seed: 2).ToDataset();
        int[] order = [.. Enumerable.Range(0, 50)];
        new Random(2).Shuffle(order);
        AssertClose(data.Subset(order).Features.ToArray(), shuffled.Features.ToArray(), 0, "the shuffled view");

        var subset = source.Subset([4, 4, 0]).ToDataset();
        AssertClose([.. data.GetFeatures(4), .. data.GetFeatures(4), .. data.GetFeatures(0)], subset.Features.ToArray(), 0, "a subset with a repeated index");
        Throws<ArgumentOutOfRangeException>(() => source.Subset([50]), "an index past the end");

        var empty = source.Subset([]);
        var joined = empty.Concat(source.Subset([1, 2]), empty, source.Subset([3]), empty);
        Check(joined.Count == 3, "concatenation count");
        AssertClose([.. data.GetFeatures(1), .. data.GetFeatures(2), .. data.GetFeatures(3)], joined.ToDataset().Features.ToArray(), 0, "concatenation across empty parts");
        Throws<ArgumentException>(() => source.Concat(RandomData(2, 4, 1, 1)), "sources of different shapes");
        var named = Dataset.FromSource(joined, ["a", "b", "c"], ["t"]);
        Check(named.FeatureNames.SequenceEqual(["a", "b", "c"]) && named.TargetNames.SequenceEqual(["t"]), "names given to FromSource");
        Check(ReferenceEquals(Dataset.FromSource(data), data), "FromSource of a dataset is the dataset");
    }

    private static void LoaderCsv(Device device)
    {
        string folder = TempFolder();
        try
        {
            // A byte order mark, CRLF and lone CR line ends, blank and space-only lines, quoted fields, an ignored column.
            string path = Path.Combine(folder, "rows.csv");
            var text = new StringBuilder("id,size,rooms,\"price\"\r\n");
            var random = new Random(3);
            for (int i = 0; i < 120; i++)
            {
                text.Append($"{i},{random.Next(50, 300)},{random.Next(1, 6)},\"{random.Next(100, 900) + 0.25 * random.Next(4)}\"");
                text.Append(i % 17 == 0 ? "\r\n   \r\n" : i % 11 == 0 ? "\r" : i == 119 ? "" : "\n");
            }

            File.WriteAllText(path, text.ToString(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
            var options = new CsvOptions { TargetColumns = ["price"], IgnoreColumns = ["id"] };
            var loaded = Dataset.LoadCsv(path, options);
            using var csv = CsvSource.Open(path, options);
            Check(csv.Count == 120 && loaded.Count == 120, $"rows: {csv.Count} streamed, {loaded.Count} loaded");
            Check(csv.FeatureNames.SequenceEqual(loaded.FeatureNames) && csv.TargetNames.SequenceEqual(loaded.TargetNames), "column names");
            var read = csv.ToDataset();
            AssertClose(loaded.Features.ToArray(), read.Features.ToArray(), 0, "features of every row");
            AssertClose(loaded.Targets.ToArray(), read.Targets.ToArray(), 0, "targets of every row");
            SameTraining(csv, loaded, device, "CSV source");

            // In order as a stream, without the scan.
            var streamed = Epoch(new DataLoader(csv.AsStream(), 32, device: device));
            SameEpoch(streamed, Epoch(new DataLoader(loaded, 32, device: device)), "the CSV rows as a stream");

            // Errors name the row; a header-less file names its columns by number.
            File.WriteAllText(Path.Combine(folder, "bad.csv"), "a,b\n1,2\n3,x\n");
            using var bad = CsvSource.Open(Path.Combine(folder, "bad.csv"), new CsvOptions { TargetColumns = ["b"] });
            try
            {
                bad.ToDataset();
                Check(false, "a value that is not a number is an error");
            }
            catch (FormatException e)
            {
                Check(e.Message.Contains("row 2", StringComparison.Ordinal), e.Message);
            }

            Throws<ArgumentException>(() => CsvSource.Open(path, new CsvOptions { TargetColumns = ["missing"] }), "an unknown column");
            File.WriteAllText(Path.Combine(folder, "plain.csv"), "1,2,3\n4,5,6\n");
            using var plain = CsvSource.Open(Path.Combine(folder, "plain.csv"), new CsvOptions { TargetColumns = ["2"], HasHeader = false });
            AssertClose([1, 2, 4, 5], plain.ToDataset().Features.ToArray(), 0, "a file without a header");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // ---------------------------------------------------------------- image files written here, read by the library

    // A PNG of any colour type and depth, interlaced or not, with the five filters in turn; sample(x, y, s) gives raw values.
    private static byte[] TestPng(int w, int h, int color, int depth, bool interlace, Func<int, int, int, int> sample, byte[]? palette = null)
    {
        int samples = color switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, _ => 4 };
        var passes = interlace
            ? new[] { (0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2) }
            : [(0, 0, 1, 1)];
        int bits = samples * depth, bpp = Math.Max(1, bits / 8), filter = 0;
        var raw = new MemoryStream();
        foreach (var (x0, y0, dx, dy) in passes)
        {
            int pw = (w - x0 + dx - 1) / dx, ph = (h - y0 + dy - 1) / dy;
            if (pw == 0 || ph == 0)
            {
                continue;
            }

            int stride = (pw * bits + 7) / 8;
            var previous = new byte[stride];
            for (int py = 0; py < ph; py++)
            {
                // Pack the row: samples high bits first, 16-bit values big-endian.
                var line = new byte[stride];
                for (int px = 0; px < pw; px++)
                {
                    for (int s = 0; s < samples; s++)
                    {
                        int v = sample(x0 + px * dx, y0 + py * dy, s), index = px * samples + s;
                        if (depth == 16)
                        {
                            line[index * 2] = (byte)(v >> 8);
                            line[index * 2 + 1] = (byte)v;
                        }
                        else if (depth == 8)
                        {
                            line[index] = (byte)v;
                        }
                        else
                        {
                            line[index * depth / 8] |= (byte)(v << (8 - depth - index * depth % 8));
                        }
                    }
                }

                int type = filter++ % 5;
                raw.WriteByte((byte)type);
                for (int i = 0; i < stride; i++)
                {
                    int left = i >= bpp ? line[i - bpp] : 0, up = previous[i], upLeft = i >= bpp ? previous[i - bpp] : 0;
                    int p = left + up - upLeft;
                    int paeth = Math.Abs(p - left) <= Math.Abs(p - up) && Math.Abs(p - left) <= Math.Abs(p - upLeft) ? left : Math.Abs(p - up) <= Math.Abs(p - upLeft) ? up : upLeft;
                    int predicted = type switch { 1 => left, 2 => up, 3 => (left + up) / 2, 4 => paeth, _ => 0 };
                    raw.WriteByte((byte)(line[i] - predicted));
                }

                previous = line;
            }
        }

        var compressed = new MemoryStream();
        using (var z = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
        {
            z.Write(raw.ToArray());
        }

        var png = new MemoryStream();
        png.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        void Chunk(string type, byte[] data)
        {
            var header = new byte[4];
            BinaryPrimitives.WriteInt32BigEndian(header, data.Length);
            png.Write(header);
            byte[] body = [.. Encoding.ASCII.GetBytes(type), .. data];
            png.Write(body);
            BinaryPrimitives.WriteUInt32BigEndian(header, Crc32(body));
            png.Write(header);
        }

        var ihdr = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(ihdr, w);
        BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(4), h);
        ihdr[8] = (byte)depth;
        ihdr[9] = (byte)color;
        ihdr[12] = (byte)(interlace ? 1 : 0);
        Chunk("IHDR", ihdr);
        Chunk("tEXt", Encoding.ASCII.GetBytes("Comment\0written by the tests"));
        if (palette is not null)
        {
            Chunk("PLTE", palette);
        }

        // The image data split over two chunks, as encoders may do.
        var all = compressed.ToArray();
        Chunk("IDAT", all[..(all.Length / 2)]);
        Chunk("IDAT", all[(all.Length / 2)..]);
        Chunk("IEND", []);
        return png.ToArray();
    }

    // A BMP: 1, 4 or 8 bits with the palette given, 16 bits as 5-6-5 bit fields, 24 bits, or 32 bits; top-down when asked.
    private static byte[] TestBmp(int w, int h, int bits, bool topDown, Func<int, int, (int R, int G, int B)> color, Func<int, int, int>? index = null, (byte R, byte G, byte B)[]? palette = null)
    {
        int stride = (w * bits + 31) / 32 * 4;
        int paletteBytes = palette is null ? 0 : palette.Length * 4, masks = bits == 16 ? 12 : 0;
        int offset = 14 + 40 + masks + paletteBytes;
        var file = new byte[offset + stride * h];
        file[0] = (byte)'B';
        file[1] = (byte)'M';
        BinaryPrimitives.WriteInt32LittleEndian(file.AsSpan(2), file.Length);
        BinaryPrimitives.WriteInt32LittleEndian(file.AsSpan(10), offset);
        BinaryPrimitives.WriteInt32LittleEndian(file.AsSpan(14), 40);
        BinaryPrimitives.WriteInt32LittleEndian(file.AsSpan(18), w);
        BinaryPrimitives.WriteInt32LittleEndian(file.AsSpan(22), topDown ? -h : h);
        BinaryPrimitives.WriteInt16LittleEndian(file.AsSpan(26), 1);
        BinaryPrimitives.WriteInt16LittleEndian(file.AsSpan(28), (short)bits);
        BinaryPrimitives.WriteInt32LittleEndian(file.AsSpan(30), bits == 16 ? 3 : 0);
        BinaryPrimitives.WriteInt32LittleEndian(file.AsSpan(46), palette?.Length ?? 0);
        if (bits == 16)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(54), 0xF800);
            BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(58), 0x07E0);
            BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(62), 0x001F);
        }

        for (int i = 0; palette is not null && i < palette.Length; i++)
        {
            int at = 14 + 40 + i * 4;
            (file[at], file[at + 1], file[at + 2]) = (palette[i].B, palette[i].G, palette[i].R);
        }

        for (int y = 0; y < h; y++)
        {
            int row = offset + (topDown ? y : h - 1 - y) * stride;
            for (int x = 0; x < w; x++)
            {
                var (r, g, b) = color(x, y);
                switch (bits)
                {
                    case 1 or 4 or 8:
                        file[row + x * bits / 8] |= (byte)(index!(x, y) << (8 - bits - x * bits % 8));
                        break;
                    case 16:
                        BinaryPrimitives.WriteUInt16LittleEndian(file.AsSpan(row + x * 2), (ushort)(r << 11 | g << 5 | b));
                        break;
                    default:
                        int at = row + x * bits / 8;
                        (file[at], file[at + 1], file[at + 2]) = ((byte)b, (byte)g, (byte)r);
                        if (bits == 32)
                        {
                            file[at + 3] = 77;                                         // unused byte, ignored
                        }

                        break;
                }
            }
        }

        return file;
    }

    private static void CheckImage(byte[] file, string extension, int channels, int w, int h, Func<int, int, int, float> expected, string what)
    {
        string path = Path.Combine(Path.GetTempPath(), $"idrak-codec-{Guid.NewGuid():N}{extension}");
        File.WriteAllBytes(path, file);
        try
        {
            var info = ImageCodecs.ReadInfo(path);
            Check(info is { } i && i.Width == w && i.Height == h && i.Channels == channels, $"{what}: header {info}");
            var image = ImageCodecs.Decode(path);
            Check(image.Channels == channels && image.Width == w && image.Height == h, $"{what}: decoded {image.Channels} x {image.Height} x {image.Width}");
            var reference = new float[channels * h * w];
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        reference[c * h * w + y * w + x] = expected(c, y, x);
                    }
                }
            }

            AssertClose(reference, image.Pixels, 1e-6f, what);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void LoaderCodecs(Device device)
    {
        const int W = 13, H = 11;                                                       // odd sizes: partial bytes and Adam7 passes
        int Value(int x, int y, int s, int max) => (x * 7 + y * 13 + s * 29 + x * y) % (max + 1);
        foreach (bool interlace in new[] { false, true })
        {
            foreach (int depth in new[] { 1, 2, 4, 8, 16 })
            {
                int max = (1 << depth) - 1;
                CheckImage(TestPng(W, H, 0, depth, interlace, (x, y, s) => Value(x, y, s, max)), ".png", 1, W, H,
                    (c, y, x) => Value(x, y, 0, max) / (float)max, $"PNG grey {depth} bits{(interlace ? ", interlaced" : "")}");
            }

            foreach (int depth in new[] { 8, 16 })
            {
                int max = (1 << depth) - 1;
                CheckImage(TestPng(W, H, 2, depth, interlace, (x, y, s) => Value(x, y, s, max)), ".png", 3, W, H,
                    (c, y, x) => Value(x, y, c, max) / (float)max, $"PNG RGB {depth} bits{(interlace ? ", interlaced" : "")}");
                CheckImage(TestPng(W, H, 6, depth, interlace, (x, y, s) => Value(x, y, s, max)), ".png", 3, W, H,
                    (c, y, x) => Value(x, y, c, max) / (float)max, $"PNG RGBA {depth} bits (alpha dropped)");
                CheckImage(TestPng(W, H, 4, depth, interlace, (x, y, s) => Value(x, y, s, max)), ".png", 1, W, H,
                    (c, y, x) => Value(x, y, 0, max) / (float)max, $"PNG grey and alpha {depth} bits");
            }

            foreach (int depth in new[] { 1, 2, 4, 8 })
            {
                int entries = 1 << depth;
                var palette = new byte[entries * 3];
                for (int i = 0; i < palette.Length; i++)
                {
                    palette[i] = (byte)(i * 37 + 5);
                }

                CheckImage(TestPng(W, H, 3, depth, interlace, (x, y, s) => Value(x, y, 0, entries - 1), palette), ".png", 3, W, H,
                    (c, y, x) => palette[Value(x, y, 0, entries - 1) * 3 + c] / 255f, $"PNG palette {depth} bits{(interlace ? ", interlaced" : "")}");
            }
        }

        (int, int, int) Colour(int x, int y) => ((x * 19 + y) % 256, (y * 23 + 3) % 256, (x * y + 7) % 256);
        foreach (bool topDown in new[] { false, true })
        {
            CheckImage(TestBmp(W, H, 24, topDown, Colour), ".bmp", 3, W, H, (c, y, x) => (c == 0 ? Colour(x, y).Item1 : c == 1 ? Colour(x, y).Item2 : Colour(x, y).Item3) / 255f,
                $"BMP 24 bits{(topDown ? ", top-down" : "")}");
            CheckImage(TestBmp(W, H, 32, topDown, Colour), ".bmp", 3, W, H, (c, y, x) => (c == 0 ? Colour(x, y).Item1 : c == 1 ? Colour(x, y).Item2 : Colour(x, y).Item3) / 255f,
                "BMP 32 bits");
        }

        (int, int, int) Packed(int x, int y) => ((x + y) % 32, (x * 3 + y) % 64, (x * y) % 32);
        CheckImage(TestBmp(W, H, 16, false, Packed), ".bmp", 3, W, H,
            (c, y, x) => c == 0 ? Packed(x, y).Item1 / 31f : c == 1 ? Packed(x, y).Item2 / 63f : Packed(x, y).Item3 / 31f, "BMP 16 bits, 5-6-5 bit fields");
        var grey = Enumerable.Range(0, 256).Select(i => ((byte)i, (byte)i, (byte)i)).ToArray();
        CheckImage(TestBmp(W, H, 8, false, (x, y) => (0, 0, 0), (x, y) => (x * 17 + y * 5) % 256, grey), ".bmp", 1, W, H,
            (c, y, x) => (x * 17 + y * 5) % 256 / 255f, "BMP 8 bits with a grey palette");
        foreach (int bits in new[] { 1, 4, 8 })
        {
            var colours = Enumerable.Range(0, 1 << bits).Select(i => ((byte)(i * 41 % 256), (byte)(i * 13 % 256), (byte)(255 - i % 256))).ToArray();
            int Index(int x, int y) => (x + 2 * y) % (1 << bits);
            CheckImage(TestBmp(W, H, bits, true, (x, y) => (0, 0, 0), Index, colours), ".bmp", 3, W, H,
                (c, y, x) => (c == 0 ? colours[Index(x, y)].Item1 : c == 1 ? colours[Index(x, y)].Item2 : colours[Index(x, y)].Item3) / 255f, $"BMP {bits} bits with a colour palette");
        }

        // Netpbm: binary 8 and 16 bits, text with comments.
        var p5 = new List<byte>(Encoding.ASCII.GetBytes($"P5\n# made by the tests\n{W} {H}\n255\n"));
        p5.AddRange(Enumerable.Range(0, W * H).Select(i => (byte)(i * 3 % 256)));
        CheckImage([.. p5], ".pgm", 1, W, H, (c, y, x) => (y * W + x) * 3 % 256 / 255f, "PGM binary");
        var p6 = new List<byte>(Encoding.ASCII.GetBytes($"P6 {W} {H} 1000\n"));
        foreach (int i in Enumerable.Range(0, W * H * 3))
        {
            p6.Add((byte)(i * 7 % 1001 >> 8));
            p6.Add((byte)(i * 7 % 1001));
        }

        CheckImage([.. p6], ".ppm", 3, W, H, (c, y, x) => ((y * W + x) * 3 + c) * 7 % 1001 / 1000f, "PPM binary 16 bits");
        string p2 = $"P2\n{W} {H}\n# comment\n15\n" + string.Join(' ', Enumerable.Range(0, W * H).Select(i => i % 16));
        CheckImage(Encoding.ASCII.GetBytes(p2), ".pgm", 1, W, H, (c, y, x) => (y * W + x) % 16 / 15f, "PGM text");

        // Resizing: grey repeated to colour, colour averaged, corners aligned.
        var image = new ImageData([0f, 1f, 2f, 3f], 1, 2, 2);
        AssertClose([0f, 0.5f, 1f, 1f, 1.5f, 2f, 2f, 2.5f, 3f], image.Resize(1, 3, 3), 1e-6f, "bilinear resize");
        AssertClose([0f, 1f, 2f, 3f, 0f, 1f, 2f, 3f], image.Resize(2, 2, 2), 0, "grey to two channels");
        AssertClose([2f], new ImageData([1f, 2f, 3f], 3, 1, 1).Resize(1, 1, 1), 1e-6f, "colour to grey");

        // Shrinking averages the pixels each output pixel covers: blocks of a 4 x 4 image, one axis shrunk and the other
        // grown, and a 3-pixel line in a 280 x 280 image, which bilinear sampling at 28 points per row stepped over.
        AssertClose([2.5f, 4.5f, 10.5f, 12.5f], new ImageData([.. Enumerable.Range(0, 16).Select(i => (float)i)], 1, 4, 4).Resize(1, 2, 2), 1e-6f, "shrink: block means");
        AssertClose([0.5f, 2.5f, 0.5f, 2.5f], new ImageData([0f, 1f, 2f, 3f], 1, 1, 4).Resize(1, 2, 2), 1e-6f, "shrink one axis, grow the other");
        foreach (int left in new[] { 100, 104, 108, 140 })
        {
            var line = new float[280 * 280];
            for (int y = 40; y <= 240; y++)
            {
                for (int x = left; x < left + 3; x++)
                {
                    line[y * 280 + x] = 1f;
                }
            }

            float ink = new ImageData(line, 1, 280, 280).Resize(1, 28, 28).Sum();
            Check(MathF.Abs(ink - 3f * 201 / 100) < 1e-3f, $"shrink: a 3-pixel line at x = {left} keeps its ink ({ink:F3} of {3f * 201 / 100:F3})");
        }

        // A damaged file names itself; an unknown format names the codecs.
        string damaged = Path.Combine(Path.GetTempPath(), $"idrak-damaged-{Guid.NewGuid():N}.png");
        File.WriteAllBytes(damaged, TestPng(W, H, 0, 8, false, (x, y, s) => x)[..60]);
        string unknown = Path.ChangeExtension(damaged, ".gif");
        File.WriteAllBytes(unknown, Encoding.ASCII.GetBytes("GIF89a......"));
        try
        {
            foreach (var (file, words) in new[] { (damaged, "png"), (unknown, "netpbm") })
            {
                try
                {
                    ImageCodecs.Decode(file);
                    Check(false, $"{file} decoded");
                }
                catch (InvalidDataException e)
                {
                    Check(e.Message.Contains(file, StringComparison.Ordinal) && e.Message.Contains(words, StringComparison.Ordinal), e.Message);
                }
            }
        }
        finally
        {
            File.Delete(damaged);
            File.Delete(unknown);
        }
    }

    // A made-up format for the codec registry: "TINY", width, height, then one byte per grey pixel.
    private sealed class TinyCodec : IImageCodec
    {
        public string Name => "tiny";

        public IReadOnlyCollection<string> Extensions { get; } = [".tiny"];

        public ImageInfo? ReadInfo(ReadOnlySpan<byte> header) =>
            header.Length >= 6 && header[..4].SequenceEqual("TINY"u8) ? new ImageInfo(header[4], header[5], 1, Name) : null;

        public ImageData Decode(ReadOnlySpan<byte> file)
        {
            int w = file[4], h = file[5];
            var pixels = new float[w * h];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = file[6 + i] / 255f;
            }

            return new ImageData(pixels, 1, h, w);
        }
    }

    // Three classes of 8x8 drawings (a bar, a column, a dot) in PGM, PNG, BMP and the registered "tiny" format, some of other sizes.
    private static string ImageFolder(string root, int perClass)
    {
        string[] classes = ["bar", "column", "dot"];
        var random = new Random(4);
        for (int c = 0; c < classes.Length; c++)
        {
            Directory.CreateDirectory(Path.Combine(root, classes[c], "more"));
            for (int n = 0; n < perClass; n++)
            {
                int size = n % 5 == 4 ? 10 : 8, at = random.Next(2, 6);
                var pixels = new byte[size * size];
                for (int y = 0; y < size; y++)
                {
                    for (int x = 0; x < size; x++)
                    {
                        bool on = c == 0 ? y == at : c == 1 ? x == at : Math.Abs(x - at) + Math.Abs(y - at) <= 1;
                        pixels[y * size + x] = (byte)(on ? 200 + random.Next(55) : random.Next(40));
                    }
                }

                string name = Path.Combine(root, classes[c], n % 7 == 6 ? "more" : "", $"{n:D3}");
                switch (n % 4)
                {
                    case 0:
                        File.WriteAllBytes(name + ".pgm", [.. Encoding.ASCII.GetBytes($"P5\n{size} {size}\n255\n"), .. pixels]);
                        break;
                    case 1:
                        File.WriteAllBytes(name + ".png", GrayPng(pixels, size, size));
                        break;
                    case 2:
                        File.WriteAllBytes(name + ".bmp", TestBmp(size, size, 24, false, (x, y) => (pixels[y * size + x], pixels[y * size + x], pixels[y * size + x])));
                        break;
                    default:
                        File.WriteAllBytes(name + ".tiny", [.. "TINY"u8.ToArray(), (byte)size, (byte)size, .. pixels]);
                        break;
                }
            }

            File.WriteAllText(Path.Combine(root, classes[c], "notes.txt"), "not an image");
        }

        return root;
    }

    private static void LoaderImageFolder(Device device)
    {
        string folder = TempFolder();
        ImageCodecs.Register(new TinyCodec());
        try
        {
            Check(ImageCodecs.Names.First() == "tiny" && ImageCodecs.Extensions.Contains(".tiny"), "a registered codec is asked first");
            string root = ImageFolder(Path.Combine(folder, "images"), 24);
            var images = new ImageFolderSource(root, 1, 8, 8);
            Check(images.Classes.SequenceEqual(["bar", "column", "dot"]) && images.Count == 72, $"{images.Count} images of {string.Join(", ", images.Classes)}");
            Check(images.FeatureShape.SequenceEqual([1, 8, 8]) && images.TargetShape.SequenceEqual([3]), "the shapes");
            Check(images.Files.Select(Path.GetFileName).Count(f => f!.EndsWith(".tiny", StringComparison.Ordinal)) == 18, "the registered format's files are picked up");

            // The pixels of one PGM file, and its one-hot class.
            var x = new float[64];
            var y = new float[3];
            int pgm = images.Files.ToList().FindIndex(f => f.EndsWith("column" + Path.DirectorySeparatorChar + "000.pgm", StringComparison.Ordinal));
            images.Read(pgm, x, y);
            var bytes = File.ReadAllBytes(images.Files[pgm])[^64..];
            AssertClose([.. bytes.Select(b => b / 255f)], x, 1e-6f, "a PGM image's pixels");
            AssertClose([0, 1, 0], y, 0, "its class");

            var memory = Dataset.FromSource(images, targetNames: images.Classes);
            Check(memory.TargetNames.SequenceEqual(images.Classes) && memory.FeatureShape.SequenceEqual([1, 8, 8]), "the in-memory copy");
            ISampleTransform[] transforms = [new RandomFlip(), new RandomShift(1), new RandomRotation(10), new GaussianNoise(0.02f)];
            SameTraining(images, memory, device, "image folder source with transforms", transforms);

            // The colour size of the first image by default, through the registry.
            var opened = SampleSources.Open("images", root);
            Check(opened.FeatureShape.SequenceEqual([1, 8, 8]), $"sizes from the first image: [{string.Join(", ", opened.FeatureShape)}]");
            var colour = (ImageFolderSource)SampleSources.Open("images", root, new Dictionary<string, string> { ["channels"] = "3", ["height"] = "4", ["width"] = "4" });
            Check(colour.FeatureShape.SequenceEqual([3, 4, 4]), "sizes given as options");

            // Unlabelled images, as predict reads them.
            var unlabelled = new ImageFolderSource(images.Files.Take(5).ToList(), 1, 8, 8);
            Check(unlabelled.TargetShape.SequenceEqual([0]) && unlabelled.Labels is null && Dataset.FromSource(unlabelled).Count == 5, "unlabelled images");
            Throws<DirectoryNotFoundException>(() => _ = new ImageFolderSource(Path.Combine(folder, "nowhere"), 1, 8, 8), "a missing folder");
        }
        finally
        {
            ImageCodecs.Unregister("tiny");
            Directory.Delete(folder, recursive: true);
        }

        Check(!ImageCodecs.Names.Contains("tiny") && ImageCodecs.Names.SequenceEqual(["png", "bmp", "netpbm"]), "unregistered");
        try
        {
            ImageCodecs.Get("jpeg");
            Check(false, "no JPEG codec is built in");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("png, bmp, netpbm", StringComparison.Ordinal) && e.Message.Contains("ImageCodecs.Register", StringComparison.Ordinal), e.Message);
        }
    }

    private static void LoaderTransforms(Device device)
    {
        // Two channels of 4 x 5 pixels, numbered.
        int[] shape = [2, 4, 5];
        float[] image = [.. Enumerable.Range(0, 40).Select(i => (float)i)];
        float At(float[] values, int c, int y, int x) => values[c * 20 + y * 5 + x];

        var flipped = (float[])image.Clone();
        new RandomFlip(horizontal: true, vertical: true, probability: 1).Apply(flipped, [], shape, new Random(1));
        Check(Enumerable.Range(0, 40).All(i => flipped[i] == At(image, i / 20, 3 - i % 20 / 5, 4 - i % 5)), "flipped both ways");
        var kept = (float[])image.Clone();
        new RandomFlip(probability: 0).Apply(kept, [], shape, new Random(1));
        Check(kept.SequenceEqual(image), "probability 0 flips nothing");

        // The shift chosen by the same random numbers, checked pixel by pixel.
        for (int seed = 0; seed < 6; seed++)
        {
            var shifted = (float[])image.Clone();
            new RandomShift(2, fill: -1).Apply(shifted, [], shape, new Random(seed));
            var replay = new Random(seed);
            int dx = replay.Next(-2, 3), dy = replay.Next(-2, 3);
            for (int i = 0; i < 40; i++)
            {
                int c = i / 20, y = i % 20 / 5, x = i % 5, sy = y - dy, sx = x - dx;
                float expected = sy is >= 0 and < 4 && sx is >= 0 and < 5 ? At(image, c, sy, sx) : -1;
                Check(shifted[i] == expected, $"shift ({dx}, {dy}) at {i}: {shifted[i]}, expected {expected}");
            }
        }

        // A bright pixel off centre moves to where the angle drawn from the same random numbers puts it.
        const int N = 21;
        for (int seed = 1; seed < 5; seed++)
        {
            var dot = new float[N * N];
            dot[5 * N + 15] = 1;                                                        // (x 15, y 5): 5 right of and 5 above the centre (10, 10)
            new RandomRotation(40).Apply(dot, [], [1, N, N], new Random(seed));
            double angle = (new Random(seed).NextDouble() * 2 - 1) * 40 * Math.PI / 180;
            double mass = dot.Sum(), mx = 0, my = 0;
            for (int i = 0; i < dot.Length; i++)
            {
                mx += dot[i] * (i % N) / mass;
                my += dot[i] * (i / N) / mass;
            }

            // Output offset o samples the input at M o, M = [[cos, sin], [-sin, cos]], so content at offset v = (5, -5)
            // moves to the transpose of M times v (image axes, y down).
            double ex = 10 + Math.Cos(angle) * 5 + Math.Sin(angle) * 5, ey = 10 + Math.Sin(angle) * 5 - Math.Cos(angle) * 5;
            Check(Math.Abs(mx - ex) < 0.35 && Math.Abs(my - ey) < 0.35 && Math.Abs(mass - 1) < 0.35, $"rotation by {angle * 180 / Math.PI:F1} degrees: centre ({mx:F2}, {my:F2}), expected ({ex:F2}, {ey:F2}), mass {mass:F2}");
        }

        var noisy = new float[20000];
        new GaussianNoise(0.5f).Apply(noisy, [], [20000], new Random(3));
        double mean = noisy.Average(), sd = Math.Sqrt(noisy.Select(v => (v - mean) * (v - mean)).Average());
        Check(Math.Abs(mean) < 0.02 && Math.Abs(sd - 0.5) < 0.02, $"noise mean {mean:F4}, standard deviation {sd:F4}");
        var clamped = Enumerable.Repeat(0.5f, 1000).ToArray();
        new GaussianNoise(2f, clamp: true).Apply(clamped, [], [1000], new Random(3));
        Check(clamped.All(v => v is >= 0 and <= 1) && clamped.Any(v => v == 0) && clamped.Any(v => v == 1), "clamped noise");

        // Seeded per epoch and sample: two loaders with one seed agree, epochs differ, and gathering on a worker thread
        // (large batches) gives the samples of small batches.
        var data = RandomData(96, 3 * 16 * 16, 2, 6).WithFeatureShape(3, 16, 16);
        ISampleTransform[] transforms = [new RandomShift(2), new RandomRotation(15), new GaussianNoise(0.1f)];
        var a = new DataLoader(data, 64, device: device, seed: 8) { Transforms = transforms };
        var b = new DataLoader(new Forwarding(data), 64, device: device, seed: 8) { Transforms = transforms };
        var first = Epoch(a);
        SameEpoch(Epoch(b), first, "the same seed");
        var second = Epoch(a);
        Check(!second[0].X.SequenceEqual(first[0].X), "another epoch, other changes");
        var small = Epoch(new DataLoader(data, 7, device: device, seed: 8) { Transforms = transforms });
        AssertClose([.. first.SelectMany(e => e.X)], [.. small.SelectMany(e => e.X)], 0, "large (prefetched) and small batches transform alike");
        Throws<ArgumentException>(() => new RandomShift(1).Apply(new float[4], [], [4], new Random(1)), "image transforms need an image shape");
    }

    private static void LoaderMappedFiles(Device device)
    {
        string folder = TempFolder();
        try
        {
            // 16-bit token ids, back to back.
            var tokens = Enumerable.Range(0, 1000).Select(i => (ushort)((i * 7919 + i / 3) % 50000)).ToArray();
            string raw = Path.Combine(folder, "train.bin");
            File.WriteAllBytes(raw, [.. tokens.SelectMany(BitConverter.GetBytes)]);
            const int T = 16;
            using (var windows = new TokenFileSource(raw, T, stride: 10))
            {
                Check(windows.TokenCount == 1000 && windows.Count == (1000 - T - 1) / 10 + 1, $"{windows.Count} windows of {windows.TokenCount} tokens");
                var x = new List<float>();
                var y = new List<float>();
                for (int w = 0; w < windows.Count; w++)
                {
                    x.AddRange(tokens.Skip(w * 10).Take(T).Select(t => (float)t));
                    y.AddRange(tokens.Skip(w * 10 + 1).Take(T).Select(t => (float)t));
                }

                string[] names = [.. Enumerable.Range(0, T).Select(i => $"t{i}")];
                var memory = Dataset.FromFlat([.. x], [.. y], windows.Count, names, names);
                AssertClose(memory.Features.ToArray(), windows.ToDataset().Features.ToArray(), 0, "the windows' tokens");
                AssertClose(memory.Targets.ToArray(), windows.ToDataset().Targets.ToArray(), 0, "the next tokens");
                SameTraining(windows, memory, device, "token windows");
            }

            // The same ids as a one-dimensional int32 .npy file, non-overlapping windows; 32-bit raw ids.
            string npyTokens = Path.Combine(folder, "tokens.npy");
            WriteNpy(npyTokens, "<i4", [1000], tokens.SelectMany(t => BitConverter.GetBytes((int)t)));
            using (var windows = new TokenFileSource(npyTokens, T))
            {
                var check = windows.ToDataset();
                Check(windows.Count == (1000 - T - 1) / T + 1 && check.GetFeatures(2)[0] == tokens[2 * T] && check.GetTargets(2)[T - 1] == tokens[3 * T], "the .npy token file");
            }

            string raw32 = Path.Combine(folder, "train32.bin");
            File.WriteAllBytes(raw32, [.. tokens.SelectMany(t => BitConverter.GetBytes((uint)t + 70000u))]);
            using (var windows = new TokenFileSource(raw32, T, type: TokenType.UInt32))
            {
                Check(windows.ToDataset().GetFeatures(1)[3] == tokens[T + 3] + 70000f, "32-bit token ids");
            }

            Throws<InvalidDataException>(() => _ = new TokenFileSource(raw, 2000), "a window longer than the file");

            // Images [N, 2, 3] as float32 and int64 class labels; float64 targets [N, 2]; uint8 features.
            const int N = 40;
            var random = new Random(5);
            float[] pixels = [.. Enumerable.Range(0, N * 6).Select(_ => random.NextSingle())];
            long[] labels = [.. Enumerable.Range(0, N).Select(i => (long)(i % 3))];
            double[] values = [.. Enumerable.Range(0, N * 2).Select(i => i * 0.25)];
            WriteNpy(Path.Combine(folder, "x.npy"), "<f4", [N, 2, 3], pixels.SelectMany(BitConverter.GetBytes));
            WriteNpy(Path.Combine(folder, "labels.npy"), "<i8", [N], labels.SelectMany(BitConverter.GetBytes));
            WriteNpy(Path.Combine(folder, "y.npy"), "<f8", [N, 2], values.SelectMany(BitConverter.GetBytes), version: 2);
            using (var classes = new NpySource(Path.Combine(folder, "x.npy"), Path.Combine(folder, "labels.npy"), classes: 3))
            {
                Check(classes.Count == N && classes.FeatureShape.SequenceEqual([2, 3]) && classes.TargetShape.SequenceEqual([3]), "the .npy shapes");
                var features = new float[N, 6];
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < 6; j++)
                    {
                        features[i, j] = pixels[i * 6 + j];
                    }
                }

                var memory = Dataset.FromClassLabels(features, [.. labels.Select(l => (int)l)], 3).WithFeatureShape(2, 3);
                AssertClose(memory.Features.ToArray(), classes.ToDataset().Features.ToArray(), 0, ".npy features");
                AssertClose(memory.Targets.ToArray(), classes.ToDataset().Targets.ToArray(), 0, ".npy one-hot classes");
                SameTraining(classes, memory, device, ".npy arrays");
            }

            using (var regression = (NpySource)SampleSources.Open("npy", Path.Combine(folder, "x.npy"), new Dictionary<string, string> { ["targets"] = Path.Combine(folder, "y.npy") }))
            {
                AssertClose([.. values.Select(v => (float)v)], regression.ToDataset().Targets.ToArray(), 0, ".npy float64 targets (format version 2)");
            }

            WriteNpy(Path.Combine(folder, "u8.npy"), "|u1", [3, 2], [1, 2, 3, 250, 0, 9]);
            using (var bytes = new NpySource(Path.Combine(folder, "u8.npy")))
            {
                Check(bytes.TargetShape.SequenceEqual([0]), "no targets");
                AssertClose([1, 2, 3, 250, 0, 9], bytes.ToDataset().Features.ToArray(), 0, ".npy uint8 features");
            }

            WriteNpy(Path.Combine(folder, "big.npy"), ">f4", [2], new byte[8]);
            Throws<InvalidDataException>(() => _ = new NpySource(Path.Combine(folder, "big.npy")), "a big-endian array");
            WriteNpy(Path.Combine(folder, "short.npy"), "<f4", [100], new byte[8]);
            Throws<InvalidDataException>(() => _ = new NpySource(Path.Combine(folder, "short.npy")), "an array shorter than its shape");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A .npy file as numpy.save writes it: magic, version, header length, the header dictionary padded to 64 bytes.
    private static void WriteNpy(string path, string descr, int[] shape, IEnumerable<byte> data, int version = 1)
    {
        string dims = shape.Length == 1 ? $"{shape[0]}," : string.Join(", ", shape);
        string header = $"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims}), }}";
        int prefix = version == 1 ? 10 : 12;
        int padded = (prefix + header.Length + 1 + 63) / 64 * 64 - prefix;
        header = header.PadRight(padded - 1) + "\n";
        var bytes = new List<byte> { 0x93 };
        bytes.AddRange("NUMPY"u8.ToArray());
        bytes.Add((byte)version);
        bytes.Add(0);
        bytes.AddRange(version == 1 ? BitConverter.GetBytes((ushort)header.Length) : BitConverter.GetBytes(header.Length));
        bytes.AddRange(Encoding.ASCII.GetBytes(header));
        bytes.AddRange(data);
        File.WriteAllBytes(path, [.. bytes]);
    }

    private static void LoaderStreams(Device device)
    {
        var data = RandomData(70, 4, 1, 7);
        var stream = new DatasetStream(data);
        var loader = new DataLoader(stream, 16, device: device);
        IBatchSource counted = loader;
        Check(loader.Source is null && loader.Stream == stream && counted.BatchCount is null && counted.SampleCount is null && loader.BatchCount == 0, "unknown counts before an epoch");
        SameEpoch(Epoch(loader), Epoch(new DataLoader(data, 16, device: device)), "a stream in order");
        Check(counted.BatchCount == 5 && counted.SampleCount == 70 && loader.BatchCount == 5 && stream.Opened == 1, "the counts after an epoch");
        Check(Epoch(new DataLoader(stream, 16, dropLast: true, device: device)).Count == 4, "dropLast on a stream");
        Check(Epoch(new DataLoader(new DatasetStream(RandomData(32, 4, 1, 1)), 16, device: device)).Count == 2, "a stream that ends at a batch boundary");

        // Shuffled in a buffer: every sample once per epoch, in another order each epoch, repeatable by seed.
        var shuffled = new DataLoader(stream, 16, shuffleBuffer: 20, device: device, seed: 3);
        var one = Epoch(shuffled);
        var two = Epoch(shuffled);
        var again = Epoch(new DataLoader(stream, 16, shuffleBuffer: 20, device: device, seed: 3));
        string Key(float[] x, int i) => string.Join(",", x.Skip(i * 4).Take(4));
        var all = Enumerable.Range(0, 70).Select(i => string.Join(",", data.GetFeatures(i).ToArray())).Order().ToList();
        Check(one.SelectMany(b => Enumerable.Range(0, b.Size).Select(i => Key(b.X, i))).Order().SequenceEqual(all), "every sample once in a shuffled epoch");
        Check(!one[0].X.SequenceEqual(two[0].X) && !one[0].X.SequenceEqual(Epoch(new DataLoader(data, 16, device: device))[0].X), "shuffled, and differently each epoch");
        SameEpoch(again, one, "the same seed");

        // Large samples are gathered on a worker thread, in order.
        var wide = RandomData(50, 600, 2, 2);
        SameEpoch(Epoch(new DataLoader(new DatasetStream(wide), 32, device: device)), Epoch(new DataLoader(wide, 32, device: device)), "a prefetched stream");

        // Trained in order, a stream and its dataset give the same weights.
        var fromStream = TrainedWeights(new DataLoader(new DatasetStream(data), 8, device: device), 4, 1, device);
        var fromMemory = TrainedWeights(new DataLoader(data, 8, device: device), 4, 1, device);
        AssertClose(fromMemory, fromStream, 1e-6f, "weights trained from a stream and from memory");
    }

    private static void LoaderRegistry(Device device)
    {
        Check(new[] { "csv", "images", "tokens", "npy" }.All(SampleSources.Contains) && SampleSources.Contains("CSV"), "the built-ins, any case");
        string folder = TempFolder();
        try
        {
            string path = Path.Combine(folder, "rows.csv");
            File.WriteAllText(path, "a;b;c\n1;2;3\n4;5;6\n");
            using (var csv = (CsvSource)SampleSources.Open("csv", path, new Dictionary<string, string> { ["target"] = "c", ["Delimiter"] = ";", ["ignore"] = "a" }))
            {
                AssertClose([2, 5], csv.ToDataset().Features.ToArray(), 0, "csv by name with options");
            }

            Throws<ArgumentException>(() => SampleSources.Open("csv", path), "csv without a target");
            Throws<ArgumentException>(() => SampleSources.Open("csv", path, new Dictionary<string, string> { ["target"] = "c", ["colour"] = "red" }), "an unknown option");

            // A source of its own, by name; then removed.
            SampleSources.Register("Squares", (p, options) =>
            {
                int n = int.Parse(options["count"], System.Globalization.CultureInfo.InvariantCulture);
                float[] x = [.. Enumerable.Range(0, n).Select(i => (float)i)];
                return Dataset.FromFlat(x, [.. x.Select(v => v * v)], n, ["x"], ["y"]);
            });
            var squares = SampleSources.Open("squares", "", new Dictionary<string, string> { ["count"] = "5" });
            Check(squares.Count == 5 && squares.ToDataset().GetTargets(4)[0] == 16 && SampleSources.Names.Contains("Squares"), "a registered source");
            Check(SampleSources.Unregister("SQUARES") && !SampleSources.Unregister("squares") && !SampleSources.Contains("squares"), "unregistered");
            try
            {
                SampleSources.Open("squares", "");
                Check(false, "an unregistered source opened");
            }
            catch (NotSupportedException e)
            {
                Check(e.Message.Contains("csv, images, tokens, npy", StringComparison.Ordinal) && e.Message.Contains("SampleSources.Register", StringComparison.Ordinal), e.Message);
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void LoaderTables(Device device)
    {
        // Parquet and its JSON Lines twin: the same numbers, booleans as 1 and 0.
        string data = Path.Combine(AppContext.BaseDirectory, "data", "parquet");
        if (!Directory.Exists(data))
        {
            data = Path.Combine(FindRepositoryRoot(), "tests", "Idrak.Tests", "data", "parquet");
        }

        string[] features = ["id", "half", "flag", "small"], targets = ["price"];
        var parquet = TableSamples.Load(Path.Combine(data, "snappy-v1.parquet"), features, targets);
        var jsonl = TableSamples.Load(Path.Combine(data, "snappy-v1.jsonl"), features, targets);
        Check(parquet.Count == 40 && parquet.FeatureNames.SequenceEqual(features) && parquet.TargetNames.SequenceEqual(targets), $"parquet: {parquet}");
        AssertClose(jsonl.Features.ToArray(), parquet.Features.ToArray(), 0, "Parquet and JSON Lines features");
        AssertClose(jsonl.Targets.ToArray(), parquet.Targets.ToArray(), 0, "Parquet and JSON Lines targets");
        var rows = File.ReadAllLines(Path.Combine(data, "snappy-v1.jsonl")).Select(l => JsonNode.Parse(l)!.AsObject()).ToList();
        float[] expected = [.. rows.SelectMany(r => features.Select(f => r[f]!.GetValueKind() == System.Text.Json.JsonValueKind.True ? 1f
            : r[f]!.GetValueKind() == System.Text.Json.JsonValueKind.False ? 0f : (float)(double)r[f]!))];
        AssertClose(expected, jsonl.Features.ToArray(), 0, "the values as written");
        SameTraining(new Forwarding(parquet), jsonl, device, "table");

        // Classes, arrays and a CSV file; streamed in order.
        string folder = TempFolder();
        try
        {
            string path = Path.Combine(folder, "points.jsonl");
            File.WriteAllLines(path, Enumerable.Range(0, 30).Select(i => new JsonObject
            {
                ["xy"] = new JsonArray(i * 0.5, -i),
                ["z"] = $"{i % 4}",
                ["label"] = i % 3 == 0 ? "red" : i % 3 == 1 ? "green" : "blue",
            }.ToJsonString()));
            var points = TableSamples.Load(path, ["xy", "z"], ["label"], classes: ["red", "green", "blue"]);
            Check(points.FeatureNames.SequenceEqual(["xy[0]", "xy[1]", "z"]) && points.TargetNames.SequenceEqual(["red", "green", "blue"]), $"names: {points}");
            AssertClose([3.5f, -7f, 3f], points.GetFeatures(7).ToArray(), 0, "an array and a number in text");
            AssertClose([0, 1, 0], points.GetTargets(7).ToArray(), 0, "a one-hot class");
            var streamed = Epoch(new DataLoader(TableSamples.Stream(path, ["xy", "z"], ["label"], ["red", "green", "blue"]), 8, device: device));
            SameEpoch(streamed, Epoch(new DataLoader(points, 8, device: device)), "the table as a stream");

            string csv = Path.Combine(folder, "rows.csv");
            File.WriteAllText(csv, "a,b,y\n1,2,3\n4,5,6\n");
            SampleSources.Register("table", TableSamples.Factory);
            try
            {
                var table = SampleSources.Open("table", csv, new Dictionary<string, string> { ["target"] = "y" });
                AssertClose([1, 2, 4, 5], table.ToDataset().Features.ToArray(), 0, "a CSV table by name, features defaulting to the other columns");
            }
            finally
            {
                SampleSources.Unregister("table");
            }

            File.WriteAllText(path, "{\"x\": 1, \"y\": 2}\n{\"x\": \"high\", \"y\": 3}\n");
            try
            {
                TableSamples.Load(path, ["x"], ["y"]);
                Check(false, "a cell that is not a number");
            }
            catch (FormatException e)
            {
                Check(e.Message.Contains("row 2", StringComparison.Ordinal) && e.Message.Contains("'x'", StringComparison.Ordinal), e.Message);
            }

            Throws<ArgumentException>(() => TableSamples.Load(path, ["w"], ["y"]), "an unknown column");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
