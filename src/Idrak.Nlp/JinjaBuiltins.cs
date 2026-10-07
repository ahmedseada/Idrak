// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections;
using System.Globalization;
using System.Text;
using Idrak.Models;

namespace Idrak.Nlp;

public sealed partial class JinjaTemplate
{
    // Python semantics for values, operators, methods, filters and tests.
    private static class Builtins
    {
        public static readonly IReadOnlyList<(string Name, object? Value)> Globals =
        [
            ("namespace", (Callable)((args, kwargs) =>
            {
                var ns = new Namespace();
                foreach (var arg in args)
                {
                    if (arg is IDictionary d)
                    {
                        foreach (DictionaryEntry e in d)
                        {
                            ns.Values[Str(e.Key)] = e.Value;
                        }
                    }
                }

                foreach (var (k, v) in kwargs)
                {
                    ns.Values[k] = v;
                }

                return ns;
            })),
            ("dict", (Callable)((_, kwargs) => new Dictionary<string, object?>(kwargs))),
            ("range", (Callable)((args, _) =>
            {
                long start = args.Count > 1 ? ToLong(args[0]) : 0, stop = ToLong(args.Count > 1 ? args[1] : args[0]), step = args.Count > 2 ? ToLong(args[2]) : 1;
                if (step == 0)
                {
                    throw new InvalidOperationException("range() step must not be zero.");
                }

                var list = new List<object?>();
                for (long i = start; step > 0 ? i < stop : i > stop; i += step)
                {
                    list.Add(i);
                }

                return list;
            })),
            ("raise_exception", (Callable)((args, _) => throw new InvalidOperationException("Chat template error: " + (args.Count > 0 ? Str(args[0]) : "")))),
            ("strftime_now", (Callable)((args, _) => StrFTime(DateTime.Now, args.Count > 0 ? Str(args[0]) : "%Y-%m-%d"))),
        ];

        // ------------------------------------------------------------------ conversions

        // Entries of any dictionary (a generic dictionary enumerates KeyValuePairs, its IDictionaryEnumerator DictionaryEntries).
        private static IEnumerable<DictionaryEntry> Entries(IDictionary d)
        {
            var e = d.GetEnumerator();
            while (e.MoveNext())
            {
                yield return e.Entry;
            }
        }

        public static bool IsNumber(object? v) => v is long or double or int;

        public static bool IsUndefined(object? v) => v is Undefined;

        public static string Str(object? v) => v switch
        {
            null => "None",
            Undefined => "",
            string s => s,
            bool b => b ? "True" : "False",
            long or int => Convert.ToInt64(v, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture),
            double d => FloatRepr(d),
            float f => FloatRepr(f),
            _ => Repr(v),
        };

        private static string Repr(object? v) => v switch
        {
            string s => s.Contains('\'') && !s.Contains('"') ? $"\"{s}\"" : "'" + s.Replace("\\", "\\\\").Replace("'", "\\'") + "'",
            IDictionary d => "{" + string.Join(", ", Entries(d).Select(e => Repr(e.Key) + ": " + Repr(e.Value))) + "}",
            Tuple t => "(" + string.Join(", ", t.Select(Repr)) + (t.Count == 1 ? ",)" : ")"),
            IList l => "[" + string.Join(", ", l.Cast<object?>().Select(Repr)) + "]",
            Namespace => "<Namespace>",
            Callable => "<function>",
            _ => Str(v),
        };

        private static string FloatRepr(double d)
        {
            if (double.IsNaN(d))
            {
                return "nan";
            }

            if (double.IsInfinity(d))
            {
                return d > 0 ? "inf" : "-inf";
            }

            string s = d.ToString("R", CultureInfo.InvariantCulture);
            if (s.Contains('E'))
            {
                // Python writes at least two exponent digits: 1e-05, 1e+16.
                int e = s.IndexOf('E');
                string mantissa = s[..e], exponent = s[(e + 1)..];
                char sign = exponent[0] == '-' ? '-' : '+';
                exponent = exponent.TrimStart('+', '-');
                return mantissa + "e" + sign + exponent.PadLeft(2, '0');
            }

            return s.Contains('.') ? s : s + ".0";
        }

        public static bool Truthy(object? v) => v switch
        {
            null or Undefined => false,
            bool b => b,
            long l => l != 0,
            int i => i != 0,
            double d => d != 0,
            string s => s.Length > 0,
            ICollection c => c.Count > 0,
            _ => true,
        };

        private static long ToLong(object? v) => v switch
        {
            long l => l,
            int i => i,
            double d => (long)d,
            bool b => b ? 1 : 0,
            string s when long.TryParse(s.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out long l) => l,
            _ => throw new InvalidOperationException($"Expected an integer, got {Repr(v)}."),
        };

        private static double ToDouble(object? v) => v switch
        {
            long l => l,
            int i => i,
            double d => d,
            bool b => b ? 1 : 0,
            string s when double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double d) => d,
            _ => throw new InvalidOperationException($"Expected a number, got {Repr(v)}."),
        };

        public static IEnumerable<object?> Iterate(object? v) => v switch
        {
            null or Undefined => [],
            string s => s.Select(c => (object?)c.ToString()),
            IDictionary d => d.Keys.Cast<object?>(),
            IEnumerable e => e.Cast<object?>(),
            _ => throw new InvalidOperationException($"{Repr(v)} is not iterable."),
        };

        private static List<object?> ToList(object? v) => [.. Iterate(v)];

        // ------------------------------------------------------------------ access

        public static object? GetAttribute(object? target, string name)
        {
            switch (target)
            {
                case Namespace ns:
                    return ns.Values.TryGetValue(name, out var nv) ? nv : Undefined.Value;
                case IDictionary d:
                    return d.Contains(name) ? d[name] : Undefined.Value;
                default:
                    return Undefined.Value;
            }
        }

        public static object? GetItem(object? target, object? index)
        {
            switch (target)
            {
                case IDictionary d:
                    string key = index as string ?? Str(index);
                    return d.Contains(key) ? d[key] : Undefined.Value;
                case Namespace ns:
                    return GetAttribute(ns, Str(index));
                case string s when IsNumber(index):
                    long i = ToLong(index);
                    i = i < 0 ? i + s.Length : i;
                    return i >= 0 && i < s.Length ? s[(int)i].ToString() : Undefined.Value;
                case IList l when IsNumber(index):
                    long j = ToLong(index);
                    j = j < 0 ? j + l.Count : j;
                    return j >= 0 && j < l.Count ? l[(int)j] : Undefined.Value;
                case null or Undefined:
                    return Undefined.Value;
                default:
                    return index is string attribute ? GetAttribute(target, attribute) : Undefined.Value;
            }
        }

        public static object? SliceOf(object? target, object? start, object? stop, object? step)
        {
            var items = target is string s ? s.Select(c => (object?)c.ToString()).ToList() : ToList(target);
            int n = items.Count;
            long st = step is null or Undefined ? 1 : ToLong(step);
            if (st == 0)
            {
                throw new InvalidOperationException("Slice step must not be zero.");
            }

            long Clamp(object? v, long fallback, long low, long high)
            {
                if (v is null or Undefined)
                {
                    return fallback;
                }

                long x = ToLong(v);
                x = x < 0 ? x + n : x;
                return Math.Clamp(x, low, high);
            }

            var result = new List<object?>();
            if (st > 0)
            {
                for (long i = Clamp(start, 0, 0, n), end = Clamp(stop, n, 0, n); i < end; i += st)
                {
                    result.Add(items[(int)i]);
                }
            }
            else
            {
                for (long i = Clamp(start, n - 1, -1, n - 1), end = Clamp(stop, -1, -1, n - 1); i > end; i += st)
                {
                    result.Add(items[(int)i]);
                }
            }

            return target is string ? string.Concat(result.Cast<string>()) : result;
        }

        // ------------------------------------------------------------------ methods

        public static bool TryMethod(object? target, string name, List<object?> args, out object? result)
        {
            object? Arg(int i) => i < args.Count ? args[i] : null;
            switch (target)
            {
                case string s:
                    result = name switch
                    {
                        "strip" => Arg(0) is string c0 ? s.Trim(c0.ToCharArray()) : s.Trim(),
                        "lstrip" => Arg(0) is string c1 ? s.TrimStart(c1.ToCharArray()) : s.TrimStart(),
                        "rstrip" => Arg(0) is string c2 ? s.TrimEnd(c2.ToCharArray()) : s.TrimEnd(),
                        "split" => Split(s, Arg(0) as string, Arg(1) is null ? -1 : ToLong(Arg(1))),
                        "rsplit" when Arg(1) is null => Split(s, Arg(0) as string, -1),
                        "splitlines" => s.Split('\n').Select(l => (object?)l.TrimEnd('\r')).Take(s.EndsWith('\n') ? s.Split('\n').Length - 1 : int.MaxValue).ToList(),
                        "startswith" => Affixes(Arg(0)).Any(p => s.StartsWith(p, StringComparison.Ordinal)),
                        "endswith" => Affixes(Arg(0)).Any(p => s.EndsWith(p, StringComparison.Ordinal)),
                        "replace" => Replace(s, Str(Arg(0)), Str(Arg(1)), Arg(2) is null ? -1 : ToLong(Arg(2))),
                        "lower" => s.ToLowerInvariant(),
                        "upper" => s.ToUpperInvariant(),
                        "title" => Title(s),
                        "capitalize" => Capitalize(s),
                        "count" => (long)Count(s, Str(Arg(0))),
                        "find" => (long)s.IndexOf(Str(Arg(0)), StringComparison.Ordinal),
                        "rfind" => (long)s.LastIndexOf(Str(Arg(0)), StringComparison.Ordinal),
                        "join" => string.Join(s, Iterate(Arg(0)).Select(Str)),
                        "isdigit" => s.Length > 0 && s.All(char.IsDigit),
                        "isalpha" => s.Length > 0 && s.All(char.IsLetter),
                        "isspace" => s.Length > 0 && s.All(char.IsWhiteSpace),
                        "format" => Format(s, args),
                        _ => Undefined.Value,
                    };
                    return result is not Undefined;
                case IDictionary d:
                    switch (name)
                    {
                        case "items":
                            result = Entries(d).Select(e => (object?)new Tuple([e.Key, e.Value])).ToList();
                            return true;
                        case "keys":
                            result = d.Keys.Cast<object?>().ToList();
                            return true;
                        case "values":
                            result = d.Values.Cast<object?>().ToList();
                            return true;
                        case "get":
                            string key = Str(Arg(0));
                            result = d.Contains(key) ? d[key] : Arg(1);
                            return true;
                    }

                    break;
                case IList l:
                    switch (name)
                    {
                        case "append":
                            l.Add(Arg(0));
                            result = null;
                            return true;
                        case "extend":
                            foreach (var item in Iterate(Arg(0)))
                            {
                                l.Add(item);
                            }

                            result = null;
                            return true;
                        case "pop":
                            int at = args.Count > 0 ? (int)ToLong(Arg(0)) : l.Count - 1;
                            at = at < 0 ? at + l.Count : at;
                            result = l[at];
                            l.RemoveAt(at);
                            return true;
                        case "index":
                            result = (long)l.Cast<object?>().ToList().FindIndex(x => Equal(x, Arg(0)));
                            return true;
                        case "count":
                            result = (long)l.Cast<object?>().Count(x => Equal(x, Arg(0)));
                            return true;
                    }

                    break;
            }

            result = null;
            return false;
        }

        private static IEnumerable<string> Affixes(object? v) => v is string s ? [s] : Iterate(v).Select(Str);

        private static List<object?> Split(string s, string? separator, long max)
        {
            if (separator is null)
            {
                var words = s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
                if (max >= 0 && words.Length > max + 1)
                {
                    // Keep the remainder (without leading whitespace) as the last part.
                    var head = words.Take((int)max).ToList();
                    int position = 0;
                    foreach (var w in head)
                    {
                        position = s.IndexOf(w, position, StringComparison.Ordinal) + w.Length;
                    }

                    return [.. head, s[position..].TrimStart()];
                }

                return [.. words];
            }

            return [.. (max >= 0 ? s.Split(separator, (int)max + 1) : s.Split(separator))];
        }

        private static string Replace(string s, string old, string replacement, long count)
        {
            if (count < 0)
            {
                return old.Length == 0 ? s : s.Replace(old, replacement, StringComparison.Ordinal);
            }

            var sb = new StringBuilder();
            int position = 0;
            for (long i = 0; i < count; i++)
            {
                int found = s.IndexOf(old, position, StringComparison.Ordinal);
                if (found < 0 || old.Length == 0)
                {
                    break;
                }

                sb.Append(s, position, found - position).Append(replacement);
                position = found + old.Length;
            }

            return sb.Append(s, position, s.Length - position).ToString();
        }

        private static int Count(string s, string part)
        {
            if (part.Length == 0)
            {
                return s.Length + 1;
            }

            int count = 0;
            for (int i = s.IndexOf(part, StringComparison.Ordinal); i >= 0; i = s.IndexOf(part, i + part.Length, StringComparison.Ordinal))
            {
                count++;
            }

            return count;
        }

        private static string Title(string s)
        {
            var sb = new StringBuilder(s.Length);
            bool previousLetter = false;
            foreach (char c in s)
            {
                sb.Append(previousLetter ? char.ToLowerInvariant(c) : char.ToUpperInvariant(c));
                previousLetter = char.IsLetter(c);
            }

            return sb.ToString();
        }

        private static string Capitalize(string s) => s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant();

        private static string Format(string s, List<object?> args)
        {
            // "{} and {}".format(a, b) and "{0}".format(a).
            var sb = new StringBuilder();
            int next = 0;
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '{' && i + 1 < s.Length && s[i + 1] == '{' || s[i] == '}' && i + 1 < s.Length && s[i + 1] == '}')
                {
                    sb.Append(s[i++]);
                }
                else if (s[i] == '{')
                {
                    int end = s.IndexOf('}', i);
                    string field = s[(i + 1)..end];
                    int index = field.Length == 0 ? next++ : int.Parse(field, CultureInfo.InvariantCulture);
                    sb.Append(Str(index < args.Count ? args[index] : Undefined.Value));
                    i = end;
                }
                else
                {
                    sb.Append(s[i]);
                }
            }

            return sb.ToString();
        }

        // ------------------------------------------------------------------ operators

        public static object? Negate(object? v) => v switch
        {
            long l => (object)-l,
            int i => -(long)i,
            double d => -d,
            _ => throw new InvalidOperationException($"Cannot negate {Repr(v)}."),
        };

        public static bool Equal(object? a, object? b)
        {
            if (a is Undefined || b is Undefined)
            {
                return a is Undefined && b is Undefined;
            }

            if (a is null || b is null)
            {
                return a is null && b is null;
            }

            if (a is bool ba && b is bool bb)
            {
                return ba == bb;
            }

            if (IsNumber(a) && IsNumber(b))
            {
                return a is double || b is double ? ToDouble(a) == ToDouble(b) : ToLong(a) == ToLong(b);
            }

            if (a is string sa && b is string sb)
            {
                return sa == sb;
            }

            if (a is IDictionary da && b is IDictionary db)
            {
                return da.Count == db.Count && Entries(da).All(e => db.Contains(e.Key) && Equal(e.Value, db[e.Key]));
            }

            if (a is IList la && b is IList lb)
            {
                return la.Count == lb.Count && Enumerable.Range(0, la.Count).All(i => Equal(la[i], lb[i]));
            }

            return ReferenceEquals(a, b) || a.Equals(b);
        }

        private static int Compare(object? a, object? b)
        {
            if (IsNumber(a) && IsNumber(b) || a is bool || b is bool)
            {
                return ToDouble(a).CompareTo(ToDouble(b));
            }

            if (a is string sa && b is string sb)
            {
                return string.CompareOrdinal(sa, sb);
            }

            if (a is IList la && b is IList lb)
            {
                for (int i = 0; i < Math.Min(la.Count, lb.Count); i++)
                {
                    int c = Compare(la[i], lb[i]);
                    if (c != 0)
                    {
                        return c;
                    }
                }

                return la.Count.CompareTo(lb.Count);
            }

            throw new InvalidOperationException($"Cannot compare {Repr(a)} and {Repr(b)}.");
        }

        private static bool Contains(object? container, object? item) => container switch
        {
            string s => s.Contains(Str(item), StringComparison.Ordinal),
            IDictionary d => item is not null && d.Contains(item is string k ? k : Str(item)),
            null or Undefined => false,
            _ => Iterate(container).Any(x => Equal(x, item)),
        };

        public static object? Operate(string op, object? a, object? b)
        {
            switch (op)
            {
                case "==": return Equal(a, b);
                case "!=": return !Equal(a, b);
                case "<": return Compare(a, b) < 0;
                case ">": return Compare(a, b) > 0;
                case "<=": return Compare(a, b) <= 0;
                case ">=": return Compare(a, b) >= 0;
                case "in": return Contains(b, a);
                case "~": return Str(a) + Str(b);
                case "+":
                    if (a is string sa && b is string sb)
                    {
                        return sa + sb;
                    }

                    if (a is IList la && b is IList lb)
                    {
                        return la.Cast<object?>().Concat(lb.Cast<object?>()).ToList();
                    }

                    return Arithmetic(a, b, (x, y) => x + y, (x, y) => x + y);
                case "-": return Arithmetic(a, b, (x, y) => x - y, (x, y) => x - y);
                case "*":
                    if (a is string rs && IsNumber(b))
                    {
                        return string.Concat(Enumerable.Repeat(rs, (int)Math.Max(0, ToLong(b))));
                    }

                    if (a is IList rl && IsNumber(b))
                    {
                        return Enumerable.Repeat(rl.Cast<object?>(), (int)Math.Max(0, ToLong(b))).SelectMany(x => x).ToList();
                    }

                    return Arithmetic(a, b, (x, y) => x * y, (x, y) => x * y);
                case "/": return ToDouble(a) / ToDouble(b);
                case "//": return Arithmetic(a, b, (x, y) => (long)Math.Floor((double)x / y), (x, y) => Math.Floor(x / y));
                case "%":
                    if (a is string format)
                    {
                        return PercentFormat(format, b is IList l ? l.Cast<object?>().ToList() : [b]);
                    }

                    return Arithmetic(a, b, (x, y) => ((x % y) + y) % y, (x, y) => x - y * Math.Floor(x / y));
                case "**":
                    return a is double || b is double || ToLong(b) < 0 ? Math.Pow(ToDouble(a), ToDouble(b)) : (object)(long)Math.Pow(ToLong(a), ToLong(b));
                default:
                    throw new InvalidOperationException($"Unknown operator '{op}'.");
            }
        }

        private static object Arithmetic(object? a, object? b, Func<long, long, long> integer, Func<double, double, double> real)
        {
            if (!(IsNumber(a) || a is bool) || !(IsNumber(b) || b is bool))
            {
                throw new InvalidOperationException($"Unsupported operands {Repr(a)} and {Repr(b)}.");
            }

            return a is double || b is double ? real(ToDouble(a), ToDouble(b)) : (object)integer(ToLong(a), ToLong(b));
        }

        private static string PercentFormat(string format, List<object?> values)
        {
            var sb = new StringBuilder();
            int next = 0;
            for (int i = 0; i < format.Length; i++)
            {
                if (format[i] != '%' || i + 1 >= format.Length)
                {
                    sb.Append(format[i]);
                    continue;
                }

                char spec = format[++i];
                if (spec == '%')
                {
                    sb.Append('%');
                    continue;
                }

                object? value = next < values.Count ? values[next++] : Undefined.Value;
                sb.Append(spec switch
                {
                    'd' or 'i' => ToLong(value).ToString(CultureInfo.InvariantCulture),
                    'r' => Repr(value),
                    _ => Str(value),
                });
            }

            return sb.ToString();
        }

        // ------------------------------------------------------------------ filters

        public static object? ApplyFilter(string name, object? value, List<object?> args, Dictionary<string, object?> kwargs, Scope scope)
        {
            _ = scope;
            object? Arg(int i, string key, object? fallback = null) => kwargs.TryGetValue(key, out var k) ? k : i >= 0 && i < args.Count ? args[i] : fallback;
            switch (name)
            {
                case "safe":
                    return value;
                case "string":
                    return Str(value);
                case "tojson":
                    var json = new StringBuilder();
                    object? indent = kwargs.TryGetValue("indent", out var ki) ? ki : args.Count > 0 && IsNumber(args[0]) ? args[0] : null;
                    WriteJson(json, value, indent is null or Undefined ? null : (int)ToLong(indent), 0, Truthy(Arg(-1, "sort_keys")),
                        Truthy(Arg(-1, "ensure_ascii", false)));
                    return json.ToString();
                case "trim":
                    return Arg(0, "chars") is string chars ? Str(value).Trim(chars.ToCharArray()) : Str(value).Trim();
                case "length" or "count":
                    return (long)(value switch
                    {
                        string s => s.Length,
                        ICollection c => c.Count,
                        null or Undefined => 0,
                        _ => Iterate(value).Count(),
                    });
                case "lower":
                    return Str(value).ToLowerInvariant();
                case "upper":
                    return Str(value).ToUpperInvariant();
                case "capitalize":
                    return Capitalize(Str(value));
                case "title":
                    return Title(Str(value));
                case "first":
                    return value is string fs ? fs.Length > 0 ? fs[..1] : Undefined.Value : Iterate(value).Cast<object?>().DefaultIfEmpty(Undefined.Value).First();
                case "last":
                    return value is string ls ? ls.Length > 0 ? ls[^1..] : Undefined.Value : Iterate(value).Cast<object?>().DefaultIfEmpty(Undefined.Value).Last();
                case "join":
                    var parts = Iterate(value);
                    if (Arg(1, "attribute") is string joinAttribute)
                    {
                        parts = parts.Select(x => Path(x, joinAttribute));
                    }

                    return string.Join(Str(Arg(0, "d", "")), parts.Select(Str));
                case "default" or "d":
                    bool boolean = Truthy(Arg(1, "boolean", false));
                    return value is Undefined || boolean && !Truthy(value) ? Arg(0, "default_value", "") : value;
                case "items":
                    return value is IDictionary id ? Entries(id).Select(e => (object?)new Tuple([e.Key, e.Value])).ToList() : new List<object?>();
                case "dictsort":
                    return value is IDictionary sd
                        ? Entries(sd).OrderBy(e => Str(e.Key), StringComparer.OrdinalIgnoreCase).Select(e => (object?)new Tuple([e.Key, e.Value])).ToList()
                        : throw new InvalidOperationException("dictsort needs a mapping.");
                case "list":
                    return ToList(value);
                case "reverse":
                    return value is string rs ? new string([.. rs.Reverse()]) : Enumerable.Reverse(ToList(value)).ToList();
                case "replace":
                    return Replace(Str(value), Str(Arg(0, "old")), Str(Arg(1, "new")), Arg(2, "count") is { } rc and not Undefined ? ToLong(rc) : -1);
                case "selectattr" or "rejectattr":
                {
                    string attribute = Str(Arg(0, "attr"));
                    string? test = args.Count > 1 ? Str(args[1]) : null;
                    var testArgs = args.Skip(2).ToList();
                    bool keep = name == "selectattr";
                    return Iterate(value).Where(x =>
                    {
                        var v = Path(x, attribute);
                        return (test is null ? Truthy(v) : RunTest(test, v, testArgs)) == keep;
                    }).ToList();
                }

                case "select" or "reject":
                {
                    string? test = args.Count > 0 ? Str(args[0]) : null;
                    var testArgs = args.Skip(1).ToList();
                    bool keep = name == "select";
                    return Iterate(value).Where(x => (test is null ? Truthy(x) : RunTest(test, x, testArgs)) == keep).ToList();
                }

                case "map":
                    if (kwargs.TryGetValue("attribute", out var mapAttribute))
                    {
                        kwargs.TryGetValue("default", out var fallback);
                        return Iterate(value).Select(x => Path(x, Str(mapAttribute)) is var v && v is Undefined && fallback is not null ? fallback : Path(x, Str(mapAttribute))).ToList();
                    }

                    string filter = Str(args[0]);
                    return Iterate(value).Select(x => ApplyFilter(filter, x, args.Skip(1).ToList(), [], scope)).ToList();
                case "attr":
                    return GetAttribute(value, Str(Arg(0, "name")));
                case "escape" or "e" or "forceescape":
                    return Str(value).Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&#34;").Replace("'", "&#39;");
                case "indent":
                {
                    object? width = Arg(0, "width", 4L);
                    string pad = width is string ws ? ws : new string(' ', (int)ToLong(width));
                    bool first = Truthy(Arg(1, "first", false)), blank = Truthy(Arg(2, "blank", false));
                    var lines = Str(value).Split('\n');
                    for (int i = 0; i < lines.Length; i++)
                    {
                        if ((i > 0 || first) && (blank || lines[i].Length > 0))
                        {
                            lines[i] = pad + lines[i];
                        }
                    }

                    return string.Join('\n', lines);
                }

                case "int":
                    try
                    {
                        return value is string si && !long.TryParse(si.Trim(), CultureInfo.InvariantCulture, out _) ? (long)ToDouble(si) : ToLong(value);
                    }
                    catch (InvalidOperationException)
                    {
                        return Arg(0, "default", 0L);
                    }

                case "float":
                    try
                    {
                        return ToDouble(value);
                    }
                    catch (InvalidOperationException)
                    {
                        return Arg(0, "default", 0.0);
                    }

                case "abs":
                    return value is double ad ? Math.Abs(ad) : (object)Math.Abs(ToLong(value));
                case "round":
                {
                    int precision = (int)ToLong(Arg(0, "precision", 0L));
                    string method = Str(Arg(1, "method", "common"));
                    double x = ToDouble(value), scale = Math.Pow(10, precision);
                    return method switch
                    {
                        "floor" => Math.Floor(x * scale) / scale,
                        "ceil" => Math.Ceiling(x * scale) / scale,
                        _ => Math.Round(x, precision, MidpointRounding.ToEven),              // Python's round
                    };
                }

                case "unique":
                {
                    var unique = new List<object?>();
                    foreach (var x in Iterate(value))
                    {
                        if (!unique.Any(u => Equal(u, x)))
                        {
                            unique.Add(x);
                        }
                    }

                    return unique;
                }

                case "sort":
                {
                    bool reverse = Truthy(Arg(0, "reverse", false));
                    var attribute = Arg(2, "attribute") as string;
                    var list = ToList(value);
                    var sorted = list.Select((x, i) => (Key: attribute is null ? x : Path(x, attribute), Index: i, Item: x)).ToList();
                    sorted.Sort((p, q) =>
                    {
                        int c = Compare(p.Key is string ps ? ps.ToLowerInvariant() : p.Key, q.Key is string qs ? qs.ToLowerInvariant() : q.Key);
                        return c != 0 ? (reverse ? -c : c) : p.Index.CompareTo(q.Index);
                    });
                    return sorted.Select(p => p.Item).ToList();
                }

                case "min" or "max":
                {
                    var list = ToList(value);
                    if (list.Count == 0)
                    {
                        return Undefined.Value;
                    }

                    var best = list[0];
                    foreach (var x in list.Skip(1))
                    {
                        int c = Compare(x, best);
                        if (name == "min" ? c < 0 : c > 0)
                        {
                            best = x;
                        }
                    }

                    return best;
                }

                case "sum":
                    return Iterate(value).Aggregate(Arg(1, "start", 0L), (acc, x) => Operate("+", acc, Arg(0, "attribute") is string sa ? Path(x, sa) : x));
                case "wordcount":
                    return (long)Str(value).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;
                case "center":
                {
                    string s = Str(value);
                    int width = (int)ToLong(Arg(0, "width", 80L));
                    int left = Math.Max(0, (width - s.Length) / 2);
                    return (new string(' ', left) + s).PadRight(width);
                }

                case "batch":
                {
                    int size = (int)ToLong(Arg(0, "linecount"));
                    return ToList(value).Chunk(size).Select(c => (object?)c.ToList()).ToList();
                }

                case "format":
                    return PercentFormat(Str(value), args);
                default:
                    throw new InvalidOperationException($"Unknown filter '{name}'.");
            }
        }

        // "a.b.0" as Jinja's attribute paths in filters (map, selectattr, sort).
        private static object? Path(object? target, string path)
        {
            foreach (var part in path.Split('.'))
            {
                target = long.TryParse(part, CultureInfo.InvariantCulture, out long index) && target is IList ? GetItem(target, index) : GetItem(target, part);
            }

            return target;
        }

        private static void WriteJson(StringBuilder sb, object? v, int? indent, int depth, bool sortKeys, bool ensureAscii)
        {
            string itemSeparator = indent is null ? ", " : ",";
            void NewLine(int level)
            {
                if (indent is int n)
                {
                    sb.Append('\n').Append(' ', n * level);
                }
            }

            switch (v)
            {
                case null or Undefined:
                    sb.Append("null");
                    break;
                case bool b:
                    sb.Append(b ? "true" : "false");
                    break;
                case long or int or double or float:
                    sb.Append(v is double or float ? FloatRepr(Convert.ToDouble(v, CultureInfo.InvariantCulture)) switch { "nan" => "NaN", "inf" => "Infinity", "-inf" => "-Infinity", var f => f } : Str(v));
                    break;
                case string s:
                    WriteJsonString(sb, s, ensureAscii);
                    break;
                case Namespace ns:
                    WriteJson(sb, ns.Values, indent, depth, sortKeys, ensureAscii);
                    break;
                case IDictionary d:
                {
                    if (d.Count == 0)
                    {
                        sb.Append("{}");
                        break;
                    }

                    var entries = Entries(d).ToList();
                    if (sortKeys)
                    {
                        entries = [.. entries.OrderBy(e => Str(e.Key), StringComparer.Ordinal)];
                    }

                    sb.Append('{');
                    for (int i = 0; i < entries.Count; i++)
                    {
                        if (i > 0)
                        {
                            sb.Append(itemSeparator);
                        }

                        NewLine(depth + 1);
                        WriteJsonString(sb, Str(entries[i].Key), ensureAscii);
                        sb.Append(": ");
                        WriteJson(sb, entries[i].Value, indent, depth + 1, sortKeys, ensureAscii);
                    }

                    NewLine(depth);
                    sb.Append('}');
                    break;
                }

                case IEnumerable e:
                {
                    var items = e.Cast<object?>().ToList();
                    if (items.Count == 0)
                    {
                        sb.Append("[]");
                        break;
                    }

                    sb.Append('[');
                    for (int i = 0; i < items.Count; i++)
                    {
                        if (i > 0)
                        {
                            sb.Append(itemSeparator);
                        }

                        NewLine(depth + 1);
                        WriteJson(sb, items[i], indent, depth + 1, sortKeys, ensureAscii);
                    }

                    NewLine(depth);
                    sb.Append(']');
                    break;
                }

                default:
                    WriteJsonString(sb, Str(v), ensureAscii);
                    break;
            }
        }

        // As Python's json.dumps: escapes quotes, backslashes and control characters (and non-ASCII with ensure_ascii).
        private static void WriteJsonString(StringBuilder sb, string s, bool ensureAscii)
        {
            sb.Append('"');
            var rest = s.AsSpan();
            int at;
            while ((at = rest.IndexOfAny(ensureAscii ? JsonEscapedAscii : JsonEscaped)) >= 0)
            {
                sb.Append(rest[..at]);                                         // the run that needs no escaping
                char c = rest[at];
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    default:
                    {
                        Span<char> escape = stackalloc char[6];
                        "\\u".CopyTo(escape);
                        ((int)c).TryFormat(escape[2..], out _, "x4", CultureInfo.InvariantCulture);
                        sb.Append(escape);
                        break;
                    }
                }

                rest = rest[(at + 1)..];
            }

            sb.Append(rest).Append('"');
        }

        // The characters WriteJsonString escapes: quotes, backslashes, control characters, and with ensure_ascii all above '~'.
        private static readonly System.Buffers.SearchValues<char> JsonEscaped = System.Buffers.SearchValues.Create(JsonEscapedChars(0x20));
        private static readonly System.Buffers.SearchValues<char> JsonEscapedAscii = System.Buffers.SearchValues.Create(JsonEscapedChars(char.MaxValue - 0x7e + 0x20));

        private static string JsonEscapedChars(int count) =>
            string.Create(count + 2, 0, static (chars, _) =>
            {
                chars[0] = '"';
                chars[1] = '\\';
                for (int i = 2; i < chars.Length; i++)
                {
                    int k = i - 2;
                    chars[i] = (char)(k < 0x20 ? k : k - 0x20 + 0x7f);         // 0x00-0x1F, then 0x7F-0xFFFF
                }
            });

        // ------------------------------------------------------------------ tests

        public static bool RunTest(string name, object? value, List<object?> args)
        {
            object? Arg(int i) => i < args.Count ? args[i] : null;
            return name switch
            {
                "defined" => value is not Undefined,
                "undefined" => value is Undefined,
                "none" => value is null,
                "string" => value is string,
                "number" => IsNumber(value),
                "integer" => value is long or int,
                "float" => value is double or float,
                "boolean" => value is bool,
                "true" => value is true,
                "false" => value is false,
                "mapping" => value is IDictionary or Namespace,
                "iterable" => value is string or IEnumerable,
                "sequence" => value is string or IList,
                "callable" => value is Callable,
                "even" => ToLong(value) % 2 == 0,
                "odd" => ToLong(value) % 2 != 0,
                "divisibleby" => ToLong(value) % ToLong(Arg(0)) == 0,
                "equalto" or "eq" or "==" or "sameas" => Equal(value, Arg(0)),
                "ne" or "!=" => !Equal(value, Arg(0)),
                "lt" or "lessthan" or "<" => Compare(value, Arg(0)) < 0,
                "le" or "<=" => Compare(value, Arg(0)) <= 0,
                "gt" or "greaterthan" or ">" => Compare(value, Arg(0)) > 0,
                "ge" or ">=" => Compare(value, Arg(0)) >= 0,
                "in" => Contains(Arg(0), value),
                "lower" => value is string lo && lo == lo.ToLowerInvariant(),
                "upper" => value is string up && up == up.ToUpperInvariant(),
                _ => throw new InvalidOperationException($"Unknown test '{name}'."),
            };
        }

        // ------------------------------------------------------------------ strftime

        private static string StrFTime(DateTime time, string format)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < format.Length; i++)
            {
                if (format[i] != '%' || i + 1 >= format.Length)
                {
                    sb.Append(format[i]);
                    continue;
                }

                char c = format[++i];
                var culture = CultureInfo.InvariantCulture;
                if (c == '-' && i + 1 < format.Length)
                {
                    // %-d, %-m, %-H: without zero padding.
                    c = format[++i];
                    sb.Append(c switch { 'd' => time.Day, 'm' => time.Month, 'H' => time.Hour, 'M' => time.Minute, _ => (object)("%-" + c) });
                    continue;
                }

                sb.Append(c switch
                {
                    'Y' => time.ToString("yyyy", culture),
                    'y' => time.ToString("yy", culture),
                    'm' => time.ToString("MM", culture),
                    'd' => time.ToString("dd", culture),
                    'B' => time.ToString("MMMM", culture),
                    'b' => time.ToString("MMM", culture),
                    'A' => time.ToString("dddd", culture),
                    'a' => time.ToString("ddd", culture),
                    'H' => time.ToString("HH", culture),
                    'I' => time.ToString("hh", culture),
                    'M' => time.ToString("mm", culture),
                    'S' => time.ToString("ss", culture),
                    'p' => time.Hour < 12 ? "AM" : "PM",
                    'j' => time.DayOfYear.ToString("000", culture),
                    '%' => "%",
                    _ => "%" + c,
                });
            }

            return sb.ToString();
        }
    }
}
