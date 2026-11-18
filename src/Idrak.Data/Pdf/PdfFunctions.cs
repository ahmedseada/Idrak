// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// A PDF function (ISO 32000-2, 7.10): m inputs to n outputs, its inputs clipped to its domain and its outputs to its
/// range. Type 0 (sampled, multilinear between its samples), type 2 (exponential), type 3 (stitching, within
/// <see cref="DocumentOptions.MaxDepth"/>) and type 4 (a PostScript calculator, bounded: 100 values on its stack, a
/// program of at most 64 Ki operators, each run at most a million steps), or an array of one-output functions. Immutable,
/// safe from several threads.
/// </summary>
internal sealed class PdfFunction
{
    private const int MaxInputs = 16, MaxOutputs = 32;

    private readonly Body _body;

    private PdfFunction(double[] domain, double[]? range, int outputs, Body body)
    {
        Domain = domain;
        Range = range;
        Inputs = domain.Length / 2;
        Outputs = outputs;
        _body = body;
    }

    // A function's work: its outputs from its inputs (clipped to its domain already).
    private delegate void Body(ReadOnlySpan<double> x, Span<double> y);

    public int Inputs { get; }

    /// <summary>The number of outputs (from the range, or the function's own count).</summary>
    public int Outputs { get; }

    private double[] Domain { get; }

    private double[]? Range { get; }

    /// <summary>The outputs for <paramref name="input"/> (as many as <see cref="Inputs"/>), clipped to the range.</summary>
    public void Evaluate(ReadOnlySpan<double> input, Span<double> output)
    {
        Span<double> x = stackalloc double[MaxInputs];
        int m = Math.Min(Inputs, MaxInputs);
        for (int i = 0; i < m; i++)
        {
            double v = i < input.Length ? input[i] : 0;
            x[i] = Math.Clamp(double.IsFinite(v) ? v : 0, Math.Min(Domain[2 * i], Domain[2 * i + 1]), Math.Max(Domain[2 * i], Domain[2 * i + 1]));
        }

        output.Clear();
        _body(x[..m], output);
        if (Range is { } range)
        {
            for (int j = 0; j < output.Length && 2 * j + 1 < range.Length; j++)
            {
                output[j] = Math.Clamp(double.IsFinite(output[j]) ? output[j] : 0, Math.Min(range[2 * j], range[2 * j + 1]), Math.Max(range[2 * j], range[2 * j + 1]));
            }
        }
    }


    /// <summary>The function <paramref name="value"/> holds (a dictionary, a stream, or an array of one-output functions).</summary>
    /// <exception cref="InvalidDataException">It is damaged or of an unknown type (named).</exception>
    public static PdfFunction Read(PdfFile file, PdfValue value, int depth = 0)
    {
        if (depth > file.MaxDepth)
        {
            throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture,
                $"functions within functions {depth} levels deep, deeper than the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth, {file.MaxDepth})"));
        }

        var resolved = file.Resolve(value);
        if (resolved.AsArray is { Count: > 0 } array)
        {
            var parts = new PdfFunction[Math.Min(array.Count, MaxOutputs)];
            for (int i = 0; i < parts.Length; i++)
            {
                parts[i] = Read(file, array[i], depth + 1);
            }

            // An array of one-output functions: output j from function j.
            return new PdfFunction(parts[0].Domain, null, parts.Length, (x, y) =>
            {
                Span<double> one = stackalloc double[MaxOutputs];
                for (int j = 0; j < parts.Length && j < y.Length; j++)
                {
                    var output = one[..Math.Max(1, parts[j].Outputs)];
                    parts[j].Evaluate(x, output);
                    y[j] = output[0];
                }
            });
        }

        var stream = resolved.AsStream;
        var dictionary = stream?.Dictionary ?? resolved.AsDictionary ?? throw new InvalidDataException($"a function that is {resolved}, not a dictionary or a stream");
        long type = file.Resolve(dictionary["FunctionType"u8]).TryInteger(out long t) ? t : -1;
        double[] domain = Numbers(file, dictionary["Domain"u8]) is { Length: >= 2 } d && d.Length % 2 == 0 && d.Length <= 2 * MaxInputs ? d : [0, 1];
        double[]? range = Numbers(file, dictionary["Range"u8]) is { Length: >= 2 } r && r.Length % 2 == 0 && r.Length <= 2 * MaxOutputs ? r : null;
        switch (type)
        {
            case 0 when stream is not null && range is not null:
            {
                var sampled = Sampled.Read(file, stream, domain, range);
                return new PdfFunction(domain, range, range.Length / 2, sampled.Compute);
            }

            case 2:
            {
                var exponential = Exponential.Read(file, dictionary);
                return new PdfFunction(domain[..2], range, range is null ? exponential.Outputs : range.Length / 2, exponential.Compute);
            }

            case 3:
            {
                var stitching = Stitching.Read(file, dictionary, domain, depth);
                return new PdfFunction(domain[..2], range, range is null ? stitching.Outputs : range.Length / 2, stitching.Compute);
            }

            case 4 when stream is not null && range is not null:
                return new PdfFunction(domain, range, range.Length / 2, Calculator.Read(file, stream, range.Length / 2).Compute);
            default:
                throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture,
                    $"a function of type {type} (types 0 and 4 are streams with a /Range; 2 and 3 dictionaries)"));
        }
    }

    internal static double[]? Numbers(PdfFile file, PdfValue value)
    {
        if (file.Resolve(value).AsArray is not { } array)
        {
            return null;
        }

        var numbers = new double[array.Count];
        for (int i = 0; i < numbers.Length; i++)
        {
            if (!file.Resolve(array[i]).TryNumber(out numbers[i]) || !double.IsFinite(numbers[i]))
            {
                return null;
            }
        }

        return numbers;
    }

    // Type 0: samples on a grid, read multilinearly between the grid's points (Order 3 read as 1).
    private sealed class Sampled
    {
        private readonly int[] _size;
        private readonly double[] _encode, _decode, _samples, _domain;
        private readonly int _outputs;

        private Sampled(double[] domain, double[] range, int[] size, double[] encode, double[] decode, double[] samples)
        {
            (_domain, _size, _encode, _decode, _samples, _outputs) = (domain, size, encode, decode, samples, range.Length / 2);
        }

        public static Sampled Read(PdfFile file, PdfStream stream, double[] domain, double[] range)
        {
            var dictionary = stream.Dictionary;
            int m = domain.Length / 2, n = range.Length / 2;
            var sizes = Numbers(file, dictionary["Size"u8]);
            if (sizes is null || sizes.Length != m || m > 8)
            {
                throw new InvalidDataException("a sampled function whose /Size does not give one count an input (8 inputs at most)");
            }

            int bits = file.Resolve(dictionary["BitsPerSample"u8]).TryInteger(out long b) && b is 1 or 2 or 4 or 8 or 12 or 16 or 24 or 32 ? (int)b : 0;
            if (bits == 0)
            {
                throw new InvalidDataException($"a sampled function of {dictionary["BitsPerSample"u8]} bits a sample");
            }

            var size = new int[m];
            long total = n;
            for (int i = 0; i < m; i++)
            {
                size[i] = (int)Math.Clamp(sizes[i], 1, 1 << 20);
                total *= size[i];
                if (total > file.MaxStreamBytes / 8 || total > (1 << 24))
                {
                    throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture,
                        $"a sampled function of {total:N0} samples or more, past the limit {nameof(DocumentOptions.MaxStreamBytes)} (DocumentOptions.MaxStreamBytes, or {DocumentOptions.MaxStreamBytesVariable})"));
                }
            }

            double[] encode = Numbers(file, dictionary["Encode"u8]) is { } e && e.Length == 2 * m ? e : [.. size.SelectMany(s => new double[] { 0, s - 1 })];
            double[] decode = Numbers(file, dictionary["Decode"u8]) is { } d && d.Length == 2 * n ? d : range;
            var samples = new double[(int)total];
            using (var data = file.Decode(stream))
            {
                var bytes = data.Span;
                double max = bits == 32 ? uint.MaxValue : (1L << bits) - 1;
                long bit = 0;
                for (int i = 0; i < samples.Length; i++)
                {
                    ulong v = 0;
                    for (int k = 0; k < bits; k++, bit++)
                    {
                        long at = bit >> 3;
                        int one = at < bytes.Length ? (bytes[(int)at] >> (7 - (int)(bit & 7))) & 1 : 0;
                        v = v << 1 | (uint)one;
                    }

                    samples[i] = v / max;
                }
            }

            return new Sampled(domain, range, size, encode, decode, samples);
        }

        public void Compute(ReadOnlySpan<double> x, Span<double> y)
        {
            int m = _size.Length;
            Span<int> lo = stackalloc int[8];
            Span<double> frac = stackalloc double[8];
            for (int i = 0; i < m; i++)
            {
                double d0 = _domain[2 * i], d1 = _domain[2 * i + 1];
                double e = d1 == d0 ? _encode[2 * i] : _encode[2 * i] + (x[i] - d0) * (_encode[2 * i + 1] - _encode[2 * i]) / (d1 - d0);
                e = Math.Clamp(e, 0, _size[i] - 1);
                lo[i] = Math.Min((int)Math.Floor(e), Math.Max(0, _size[i] - 2));
                frac[i] = _size[i] == 1 ? 0 : e - lo[i];
            }

            for (int j = 0; j < _outputs && j < y.Length; j++)
            {
                // Every corner of the cell, weighted.
                double sum = 0;
                for (int corner = 0; corner < 1 << m; corner++)
                {
                    double weight = 1;
                    int index = 0, stride = 1;
                    for (int i = 0; i < m; i++)
                    {
                        bool high = (corner >> i & 1) != 0;
                        int at = Math.Min(lo[i] + (high ? 1 : 0), _size[i] - 1);
                        weight *= high ? frac[i] : 1 - frac[i];
                        index += at * stride;
                        stride *= _size[i];
                    }

                    if (weight != 0)
                    {
                        sum += weight * _samples[index * _outputs + j];
                    }
                }

                y[j] = _decode[2 * j] + sum * (_decode[2 * j + 1] - _decode[2 * j]);
            }
        }
    }

    // Type 2: C0 + x^N (C1 - C0), one input.
    private sealed class Exponential
    {
        private readonly double[] _c0, _c1;
        private readonly double _n;

        private Exponential(double[] c0, double[] c1, double n) => (_c0, _c1, _n) = (c0, c1, n);

        public int Outputs => _c0.Length;

        public static Exponential Read(PdfFile file, PdfDictionary dictionary)
        {
            double[] c0 = Numbers(file, dictionary["C0"u8]) ?? [0], c1 = Numbers(file, dictionary["C1"u8]) ?? [1];
            if (c0.Length != c1.Length || c0.Length == 0 || c0.Length > MaxOutputs)
            {
                throw new InvalidDataException("an exponential function whose /C0 and /C1 differ in length");
            }

            double n = file.Resolve(dictionary["N"u8]).TryNumber(out double v) && double.IsFinite(v) ? v : 1;
            return new Exponential(c0, c1, n);
        }

        public void Compute(ReadOnlySpan<double> x, Span<double> y)
        {
            double p = _n == 1 ? x[0] : x[0] <= 0 && _n < 1 && _n != Math.Floor(_n) ? 0 : Math.Pow(x[0], _n);
            for (int j = 0; j < _c0.Length && j < y.Length; j++)
            {
                y[j] = _c0[j] + p * (_c1[j] - _c0[j]);
            }
        }
    }

    // Type 3: k functions over the subdomains /Bounds cuts, each input encoded into its /Encode pair.
    private sealed class Stitching
    {
        private readonly PdfFunction[] _functions;
        private readonly double[] _bounds, _encode, _domain;

        private Stitching(double[] domain, PdfFunction[] functions, double[] bounds, double[] encode) =>
            (_domain, _functions, _bounds, _encode) = (domain, functions, bounds, encode);

        public int Outputs => _functions[0].Outputs;

        public static Stitching Read(PdfFile file, PdfDictionary dictionary, double[] domain, int depth)
        {
            var list = file.Resolve(dictionary["Functions"u8]).AsArray;
            if (list is not { Count: > 0 } || list.Count > 1024)
            {
                throw new InvalidDataException("a stitching function without /Functions");
            }

            var functions = new PdfFunction[list.Count];
            for (int i = 0; i < functions.Length; i++)
            {
                functions[i] = PdfFunction.Read(file, list[i], depth + 1);
            }

            double[] bounds = Numbers(file, dictionary["Bounds"u8]) ?? [];
            double[] encode = Numbers(file, dictionary["Encode"u8]) ?? [.. Enumerable.Range(0, functions.Length).SelectMany(_ => new double[] { 0, 1 })];
            if (bounds.Length != functions.Length - 1 || encode.Length != 2 * functions.Length)
            {
                throw new InvalidDataException("a stitching function whose /Bounds or /Encode do not fit its /Functions");
            }

            return new Stitching(domain[..2], functions, bounds, encode);
        }

        public void Compute(ReadOnlySpan<double> x, Span<double> y)
        {
            double v = x[0];
            int k = 0;
            while (k < _bounds.Length && v >= _bounds[k])
            {
                k++;
            }

            double lo = k == 0 ? _domain[0] : _bounds[k - 1], hi = k == _bounds.Length ? _domain[1] : _bounds[k];
            double e0 = _encode[2 * k], e1 = _encode[2 * k + 1];
            double t = hi == lo ? e0 : e0 + (v - lo) * (e1 - e0) / (hi - lo);
            _functions[k].Evaluate([t], y);
        }
    }

    // Type 4: a PostScript calculator program (7.10.5), compiled once into steps; run with a bounded stack.
    private sealed class Calculator
    {
        private const int MaxValues = 100, MaxSteps = 1_000_000, MaxProgram = 1 << 16;

        private readonly Step[] _program;
        private readonly int _outputs;

        private Calculator(Step[] program, int outputs) => (_program, _outputs) = (program, outputs);

        public static Calculator Read(PdfFile file, PdfStream stream, int outputs)
        {
            using var data = file.Decode(stream);
            var text = data.Span;
            int at = text.IndexOf((byte)'{');
            if (at < 0)
            {
                throw new InvalidDataException("a calculator function without its braces");
            }

            var steps = new List<Step>();
            int i = at + 1;
            Compile(text, ref i, steps, 0);
            return new Calculator([.. steps], outputs);
        }

        // One procedure's steps up to its closing brace: numbers pushed, operators, and `if` / `ifelse` over the procedures
        // before them (each a jump over its steps).
        private static void Compile(ReadOnlySpan<byte> text, ref int i, List<Step> steps, int depth)
        {
            if (depth > 64)
            {
                throw new InvalidDataException("a calculator function's procedures nested more than 64 deep");
            }

            var pending = new List<(int Start, int End)>();   // the procedures just met, for if and ifelse
            while (i < text.Length)
            {
                byte b = text[i];
                if (b is (byte)' ' or (byte)'\r' or (byte)'\n' or (byte)'\t' or (byte)'\f' or 0)
                {
                    i++;
                    continue;
                }

                if (b == '}')
                {
                    i++;
                    return;
                }

                if (b == '{')
                {
                    i++;
                    int start = steps.Count;
                    steps.Add(default);                                          // a jump over the procedure, set below
                    Compile(text, ref i, steps, depth + 1);
                    steps[start] = new Step(Op.Jump, steps.Count);
                    pending.Add((start + 1, steps.Count));
                    continue;
                }

                int s = i;
                while (i < text.Length && text[i] is not ((byte)' ' or (byte)'\r' or (byte)'\n' or (byte)'\t' or (byte)'\f' or (byte)'{' or (byte)'}' or 0))
                {
                    i++;
                }

                var word = text[s..i];
                if (steps.Count > MaxProgram)
                {
                    throw new InvalidDataException("a calculator function of more than 64 Ki operators");
                }

                if (double.TryParse(word, NumberStyles.Float, CultureInfo.InvariantCulture, out double number))
                {
                    steps.Add(new Step(Op.Push, 0, number));
                    continue;
                }

                if (word.SequenceEqual("if"u8) && pending.Count >= 1)
                {
                    var p = pending[^1];
                    pending.RemoveAt(pending.Count - 1);
                    steps.Add(new Step(Op.If, p.Start, p.End));
                    continue;
                }

                if (word.SequenceEqual("ifelse"u8) && pending.Count >= 2)
                {
                    var (yes, no) = (pending[^2], pending[^1]);
                    pending.RemoveRange(pending.Count - 2, 2);
                    steps.Add(new Step(Op.IfElse, yes.Start, yes.End, no.Start, no.End));
                    continue;
                }

                steps.Add(new Step(Operator(word)));
            }
        }

        private static Op Operator(ReadOnlySpan<byte> word) => System.Text.Encoding.ASCII.GetString(word) switch
        {
            "abs" => Op.Abs, "add" => Op.Add, "atan" => Op.Atan, "ceiling" => Op.Ceiling, "cos" => Op.Cos, "cvi" => Op.Cvi, "cvr" => Op.Cvr, "div" => Op.Div,
            "exp" => Op.Exp, "floor" => Op.Floor, "idiv" => Op.Idiv, "ln" => Op.Ln, "log" => Op.Log, "mod" => Op.Mod, "mul" => Op.Mul, "neg" => Op.Neg,
            "round" => Op.Round, "sin" => Op.Sin, "sqrt" => Op.Sqrt, "sub" => Op.Sub, "truncate" => Op.Truncate, "and" => Op.And, "bitshift" => Op.Bitshift,
            "eq" => Op.Eq, "false" => Op.False, "ge" => Op.Ge, "gt" => Op.Gt, "le" => Op.Le, "lt" => Op.Lt, "ne" => Op.Ne, "not" => Op.Not, "or" => Op.Or,
            "true" => Op.True, "xor" => Op.Xor, "copy" => Op.Copy, "dup" => Op.Dup, "exch" => Op.Exch, "index" => Op.Index, "pop" => Op.Pop, "roll" => Op.Roll,
            var other => throw new InvalidDataException($"a calculator function with the operator '{other}', which PDF's calculator does not have"),
        };

        public void Compute(ReadOnlySpan<double> x, Span<double> y)
        {
            Span<double> stack = stackalloc double[MaxValues];
            int top = 0;
            foreach (double v in x)
            {
                stack[top++] = v;
            }

            int budget = MaxSteps;
            Execute(0, _program.Length, stack, ref top, ref budget);
            // The outputs are the values left, the first output deepest.
            int n = Math.Min(_outputs, y.Length);
            for (int j = 0; j < n; j++)
            {
                int at = top - n + j;
                y[j] = at >= 0 ? stack[at] : 0;
            }
        }

        private void Execute(int from, int to, Span<double> s, ref int top, ref int budget)
        {
            for (int pc = from; pc < to; pc++)
            {
                if (--budget < 0)
                {
                    return;
                }

                var step = _program[pc];
                switch (step.Op)
                {
                    case Op.Jump:
                        pc = step.A - 1;
                        continue;
                    case Op.Push:
                        Push(s, ref top, step.Value);
                        continue;
                    case Op.If:
                        if (Pop(s, ref top) != 0)
                        {
                            Execute(step.A, (int)step.Value, s, ref top, ref budget);
                        }

                        continue;
                    case Op.IfElse:
                        if (Pop(s, ref top) != 0)
                        {
                            Execute(step.A, (int)step.Value, s, ref top, ref budget);
                        }
                        else
                        {
                            Execute(step.C, step.D, s, ref top, ref budget);
                        }

                        continue;
                }

                Apply(step.Op, s, ref top);
            }
        }

        private static void Apply(Op op, Span<double> s, ref int top)
        {
            double b, a;
            switch (op)
            {
                case Op.Abs: Push(s, ref top, Math.Abs(Pop(s, ref top))); break;
                case Op.Neg: Push(s, ref top, -Pop(s, ref top)); break;
                case Op.Ceiling: Push(s, ref top, Math.Ceiling(Pop(s, ref top))); break;
                case Op.Floor: Push(s, ref top, Math.Floor(Pop(s, ref top))); break;
                case Op.Round: Push(s, ref top, Math.Floor(Pop(s, ref top) + 0.5)); break;
                case Op.Truncate or Op.Cvi: Push(s, ref top, Math.Truncate(Pop(s, ref top))); break;
                case Op.Cvr: break;
                case Op.Sqrt: Push(s, ref top, Math.Sqrt(Math.Max(0, Pop(s, ref top)))); break;
                case Op.Sin: Push(s, ref top, PortableMath.Sin(Pop(s, ref top) * Math.PI / 180)); break;
                case Op.Cos: Push(s, ref top, PortableMath.Cos(Pop(s, ref top) * Math.PI / 180)); break;
                case Op.Ln: Push(s, ref top, PortableMath.Log(Pop(s, ref top))); break;
                case Op.Log: Push(s, ref top, PortableMath.Log10(Pop(s, ref top))); break;
                case Op.Not: a = Pop(s, ref top); Push(s, ref top, a == 0 ? 1 : a == 1 ? 0 : ~(long)a); break;
                case Op.True: Push(s, ref top, 1); break;
                case Op.False: Push(s, ref top, 0); break;
                case Op.Dup: a = Pop(s, ref top); Push(s, ref top, a); Push(s, ref top, a); break;
                case Op.Pop: Pop(s, ref top); break;
                case Op.Exch: b = Pop(s, ref top); a = Pop(s, ref top); Push(s, ref top, b); Push(s, ref top, a); break;
                case Op.Index:
                {
                    int n = (int)Pop(s, ref top);
                    Push(s, ref top, n >= 0 && top - 1 - n >= 0 ? s[top - 1 - n] : 0);
                    break;
                }

                case Op.Copy:
                {
                    int n = (int)Pop(s, ref top);
                    if (n > 0 && n <= top && top + n <= s.Length)
                    {
                        s.Slice(top - n, n).CopyTo(s[top..]);
                        top += n;
                    }

                    break;
                }

                case Op.Roll:
                {
                    int j = (int)Pop(s, ref top), n = (int)Pop(s, ref top);
                    if (n > 0 && n <= top)
                    {
                        var part = s.Slice(top - n, n);
                        j = ((j % n) + n) % n;
                        part.Reverse();
                        part[..j].Reverse();
                        part[j..].Reverse();
                    }

                    break;
                }

                default:
                    b = Pop(s, ref top);
                    a = Pop(s, ref top);
                    Push(s, ref top, op switch
                    {
                        Op.Add => a + b,
                        Op.Sub => a - b,
                        Op.Mul => a * b,
                        Op.Div => b == 0 ? 0 : a / b,
                        Op.Idiv => b == 0 ? 0 : Math.Truncate(a / b),
                        Op.Mod => b == 0 ? 0 : a % b,
                        Op.Atan => Atan(a, b),
                        Op.Exp => PortableMath.Pow(a, b),
                        Op.Eq => a == b ? 1 : 0,
                        Op.Ne => a != b ? 1 : 0,
                        Op.Gt => a > b ? 1 : 0,
                        Op.Ge => a >= b ? 1 : 0,
                        Op.Lt => a < b ? 1 : 0,
                        Op.Le => a <= b ? 1 : 0,
                        Op.And => (long)a & (long)b,
                        Op.Or => (long)a | (long)b,
                        Op.Xor => (long)a ^ (long)b,
                        Op.Bitshift => b >= 0 ? (long)a << (int)Math.Min(b, 63) : (long)a >> (int)Math.Min(-b, 63),
                        _ => 0,
                    });
                    break;
            }
        }

        // atan num den: the angle in degrees, 0 to 360.
        private static double Atan(double num, double den)
        {
            double angle = PortableMath.Atan2(num, den) * 180 / Math.PI;
            return angle < 0 ? angle + 360 : angle;
        }

        private static void Push(Span<double> s, ref int top, double v)
        {
            if (top < s.Length)
            {
                s[top++] = double.IsFinite(v) ? v : 0;
            }
        }

        private static double Pop(Span<double> s, ref int top) => top > 0 ? s[--top] : 0;

        private readonly record struct Step(Op Op, int A = 0, double Value = 0, int C = 0, int D = 0)
        {
            public Step(Op op, int a, int end, int c, int d) : this(op, a, (double)end, c, d)
            {
            }
        }

        private enum Op : byte
        {
            Push, Jump, If, IfElse, Abs, Add, Atan, Ceiling, Cos, Cvi, Cvr, Div, Exp, Floor, Idiv, Ln, Log, Mod, Mul, Neg, Round, Sin, Sqrt, Sub, Truncate,
            And, Bitshift, Eq, False, Ge, Gt, Le, Lt, Ne, Not, Or, True, Xor, Copy, Dup, Exch, Index, Pop, Roll,
        }
    }
}
