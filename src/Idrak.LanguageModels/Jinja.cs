using System.Globalization;
using System.Text;

namespace Idrak.LanguageModels;

/// <summary>
/// A Jinja template interpreter covering what chat templates use: output, if/elif/else, for (with loop variables,
/// filters on the iterable, else, break/continue), set (including namespace attributes and block sets), macros, raw
/// blocks, comments, whitespace control, and Hugging Face's trim_blocks/lstrip_blocks settings; expressions with
/// Python-like values (strings, numbers, lists, dicts, None), slices, attribute and item access, string and dict
/// methods, filters (tojson with Python spacing, trim, length, join, map, selectattr, …) and tests (defined, none,
/// string, mapping, …). Values are .NET strings, longs, doubles, bools, lists, dictionaries and null.
/// </summary>
public sealed partial class JinjaTemplate
{
    private readonly List<Node> _nodes;

    private JinjaTemplate(List<Node> nodes) => _nodes = nodes;

    /// <summary>Parses <paramref name="source"/> (with trim_blocks and lstrip_blocks, as Hugging Face renders chat templates).</summary>
    public static JinjaTemplate Parse(string source) =>
        new(new Parser(Lexer.Tokenize(source.EndsWith('\n') ? source[..(source.EndsWith("\r\n", StringComparison.Ordinal) ? ^2 : ^1)] : source)).ParseAll());   // Jinja drops one trailing newline

    /// <summary>Renders the template with <paramref name="variables"/>.</summary>
    public string Render(IReadOnlyDictionary<string, object?> variables)
    {
        var output = new StringBuilder();
        var scope = new Scope(null);
        foreach (var (name, value) in variables)
        {
            scope.Set(name, value);
        }

        foreach (var (name, value) in Builtins.Globals)
        {
            if (!scope.Has(name))
            {
                scope.Set(name, value);
            }
        }

        Interpreter.Execute(_nodes, scope, output);
        return output.ToString();
    }

    // ------------------------------------------------------------------ values

    /// <summary>The value of an undefined name or missing key: renders as nothing, false in conditions.</summary>
    public sealed class Undefined
    {
        /// <summary>The undefined value.</summary>
        public static readonly Undefined Value = new();

        private Undefined()
        {
        }

        /// <inheritdoc />
        public override string ToString() => "";
    }

    // A Python tuple: a list that prints as (a, b).
    internal sealed class Tuple(IEnumerable<object?> items) : List<object?>(items);

    internal sealed class Namespace
    {
        public Dictionary<string, object?> Values { get; } = [];
    }

    internal delegate object? Callable(List<object?> args, Dictionary<string, object?> kwargs);

    internal sealed class LoopBreak : Exception;

    internal sealed class LoopContinue : Exception;

    // ------------------------------------------------------------------ lexer

    private enum TokenKind { Text, Output, Statement }

    private sealed record Token(TokenKind Kind, string Content, int Line);

    private static class Lexer
    {
        // The end of a tag, skipping string literals (so "}}" inside a string does not close it).
        private static int FindClose(string source, string close, int start)
        {
            char quote = '\0';
            for (int i = start; i < source.Length; i++)
            {
                char c = source[i];
                if (quote != '\0')
                {
                    if (c == '\\')
                    {
                        i++;
                    }
                    else if (c == quote)
                    {
                        quote = '\0';
                    }
                }
                else if (c is '\'' or '"')
                {
                    quote = c;
                }
                else if (string.CompareOrdinal(source, i, close, 0, close.Length) == 0)
                {
                    return i;
                }
            }

            return -1;
        }

        public static List<Token> Tokenize(string source)
        {
            var tokens = new List<Token>();
            int i = 0, line = 1;
            var text = new StringBuilder();
            bool trimNextText = false;           // "-" at the end of the previous tag
            bool afterBlock = false;             // trim_blocks: drop one newline after a block tag
            void FlushText(bool stripTrailing, bool lstripBlock)
            {
                string t = text.ToString();
                text.Clear();
                if (trimNextText)
                {
                    t = t.TrimStart();
                }

                if (afterBlock && !trimNextText)
                {
                    if (t.StartsWith("\r\n", StringComparison.Ordinal))
                    {
                        t = t[2..];
                    }
                    else if (t.StartsWith('\n'))
                    {
                        t = t[1..];
                    }
                }

                if (stripTrailing)
                {
                    t = t.TrimEnd();
                }
                else if (lstripBlock)
                {
                    // lstrip_blocks: spaces and tabs between the line start and a block tag are removed.
                    int end = t.Length;
                    while (end > 0 && (t[end - 1] == ' ' || t[end - 1] == '\t'))
                    {
                        end--;
                    }

                    if (end == 0 || t[end - 1] == '\n')
                    {
                        t = t[..end];
                    }
                }

                if (t.Length > 0)
                {
                    tokens.Add(new Token(TokenKind.Text, t, line));
                }

                trimNextText = false;
                afterBlock = false;
            }

            while (i < source.Length)
            {
                if (source[i] == '{' && i + 1 < source.Length && source[i + 1] is '{' or '%' or '#')
                {
                    char kind = source[i + 1];
                    string close = kind switch { '{' => "}}", '%' => "%}", _ => "#}" };
                    int start = i + 2;
                    bool stripBefore = start < source.Length && source[start] == '-';
                    bool keepBefore = start < source.Length && source[start] == '+';
                    if (stripBefore || keepBefore)
                    {
                        start++;
                    }

                    int end = kind == '#' ? source.IndexOf(close, start, StringComparison.Ordinal) : FindClose(source, close, start);
                    if (end < 0)
                    {
                        throw new FormatException($"Unclosed tag at line {line}.");
                    }

                    string body = source[start..end];
                    bool stripAfter = body.EndsWith('-');
                    if (stripAfter || body.EndsWith('+'))
                    {
                        body = body[..^1];
                    }

                    // raw blocks: the content is text, taken literally.
                    if (kind == '%' && body.Trim() == "raw")
                    {
                        FlushText(stripBefore, !keepBefore);
                        int rawEnd = FindRawEnd(source, end + 2, out int after, out bool rawStrip);
                        string raw = source[(end + 2)..rawEnd];
                        if (stripAfter)
                        {
                            raw = raw.TrimStart();
                        }

                        if (rawStrip)
                        {
                            raw = raw.TrimEnd();
                        }

                        tokens.Add(new Token(TokenKind.Text, raw, line));
                        line += source.AsSpan(i, after - i).Count('\n');
                        i = after;
                        afterBlock = true;
                        continue;
                    }

                    FlushText(stripBefore, kind != '{' && !keepBefore);
                    if (kind == '{')
                    {
                        tokens.Add(new Token(TokenKind.Output, body, line));
                    }
                    else if (kind == '%')
                    {
                        tokens.Add(new Token(TokenKind.Statement, body.Trim(), line));
                    }

                    line += source.AsSpan(i, end + 2 - i).Count('\n');
                    i = end + 2;
                    trimNextText = stripAfter;
                    afterBlock = kind != '{';
                    continue;
                }

                if (source[i] == '\n')
                {
                    line++;
                }

                text.Append(source[i++]);
            }

            FlushText(false, false);
            return tokens;
        }

        private static int FindRawEnd(string source, int from, out int after, out bool strip)
        {
            int i = from;
            while (true)
            {
                int open = source.IndexOf("{%", i, StringComparison.Ordinal);
                if (open < 0)
                {
                    throw new FormatException("Unclosed raw block.");
                }

                int close = source.IndexOf("%}", open, StringComparison.Ordinal);
                string body = source[(open + 2)..close];
                strip = body.StartsWith('-');
                if (body.Trim('-', '+', ' ', '\t', '\n', '\r') == "endraw")
                {
                    after = close + 2;
                    return open;
                }

                i = close + 2;
            }
        }
    }

    // ------------------------------------------------------------------ syntax tree

    private abstract record Node;

    private sealed record TextNode(string Text) : Node;

    private sealed record OutputNode(Expr Value) : Node;

    private sealed record IfNode(List<(Expr? Condition, List<Node> Body)> Branches) : Node;

    private sealed record ForNode(List<string> Targets, Expr Iterable, Expr? Filter, List<Node> Body, List<Node>? Else) : Node;

    private sealed record SetNode(Expr Target, Expr? Value, List<Node>? Body) : Node;

    private sealed record MacroNode(string Name, List<(string Name, Expr? Default)> Parameters, List<Node> Body) : Node;

    private sealed record CallBlockNode(Expr Call, List<Node> Body) : Node;

    private sealed record ControlNode(bool Break) : Node;

    private sealed record FilterBlockNode(string Filter, List<Expr> Args, List<Node> Body) : Node;

    private abstract record Expr;

    private sealed record Literal(object? Value) : Expr;

    private sealed record Name(string Id) : Expr;

    private sealed record ListExpr(List<Expr> Items, bool Tuple = false) : Expr;

    private sealed record DictExpr(List<(Expr Key, Expr Value)> Items) : Expr;

    private sealed record Attribute(Expr Target, string Id) : Expr;

    private sealed record Item(Expr Target, Expr Index) : Expr;

    private sealed record Slice(Expr Target, Expr? Start, Expr? Stop, Expr? Step) : Expr;

    private sealed record Call(Expr Target, List<Expr> Args, List<(string Name, Expr Value)> Kwargs) : Expr;

    private sealed record Filter(Expr Target, string Id, List<Expr> Args, List<(string Name, Expr Value)> Kwargs) : Expr;

    private sealed record Test(Expr Target, string Id, List<Expr> Args, bool Negated) : Expr;

    private sealed record Unary(string Op, Expr Operand) : Expr;

    private sealed record Binary(string Op, Expr Left, Expr Right) : Expr;

    private sealed record Conditional(Expr Condition, Expr Then, Expr? Else) : Expr;

    // ------------------------------------------------------------------ parser

    private sealed class Parser(List<Token> tokens)
    {
        private int _position;

        public List<Node> ParseAll()
        {
            var nodes = ParseUntil(out string? end);
            return end is null ? nodes : throw new FormatException($"Unexpected {{% {end} %}}.");
        }

        // Parses nodes until a statement starting with one of the end words (returned) or the end of input (null).
        private List<Node> ParseUntil(out string? end, params string[] ends)
        {
            var nodes = new List<Node>();
            while (_position < tokens.Count)
            {
                var token = tokens[_position++];
                switch (token.Kind)
                {
                    case TokenKind.Text:
                        nodes.Add(new TextNode(token.Content));
                        break;
                    case TokenKind.Output:
                        nodes.Add(new OutputNode(new ExprParser(token.Content, token.Line).ParseComplete()));
                        break;
                    default:
                        string word = FirstWord(token.Content);
                        if (ends.Contains(word))
                        {
                            end = token.Content;
                            return nodes;
                        }

                        nodes.Add(ParseStatement(token, word));
                        break;
                }
            }

            end = null;
            return ends.Length == 0 ? nodes : throw new FormatException($"Missing {{% {ends[^1]} %}}.");
        }

        private static string FirstWord(string content)
        {
            int i = 0;
            while (i < content.Length && (char.IsLetterOrDigit(content[i]) || content[i] == '_'))
            {
                i++;
            }

            return content[..i];
        }

        private Node ParseStatement(Token token, string word)
        {
            string rest = token.Content[word.Length..].Trim();
            switch (word)
            {
                case "if":
                {
                    var branches = new List<(Expr?, List<Node>)>();
                    Expr? condition = new ExprParser(rest, token.Line).ParseComplete();
                    while (true)
                    {
                        var body = ParseUntil(out string? end, "elif", "else", "endif");
                        branches.Add((condition, body));
                        string endWord = FirstWord(end!);
                        if (endWord == "endif")
                        {
                            return new IfNode(branches);
                        }

                        condition = endWord == "elif" ? new ExprParser(end![4..].Trim(), token.Line).ParseComplete() : null;
                    }
                }

                case "for":
                {
                    var parser = new ExprParser(rest, token.Line);
                    var targets = new List<string> { parser.Identifier() };
                    while (parser.Accept(","))
                    {
                        targets.Add(parser.Identifier());
                    }

                    parser.Keyword("in");
                    var iterable = parser.ParseFilteredOr();
                    Expr? filter = parser.AcceptKeyword("if") ? parser.ParseOr() : null;
                    parser.AcceptKeyword("recursive");
                    parser.End();
                    var body = ParseUntil(out string? end, "else", "endfor");
                    List<Node>? elseBody = null;
                    if (FirstWord(end!) == "else")
                    {
                        elseBody = ParseUntil(out _, "endfor");
                    }

                    return new ForNode(targets, iterable, filter, body, elseBody);
                }

                case "set":
                {
                    var parser = new ExprParser(rest, token.Line);
                    var target = parser.ParsePrimaryTarget();
                    if (parser.Accept("="))
                    {
                        var value = parser.ParseTuple();
                        parser.End();
                        return new SetNode(target, value, null);
                    }

                    parser.End();
                    return new SetNode(target, null, ParseUntil(out _, "endset"));
                }

                case "macro":
                {
                    var parser = new ExprParser(rest, token.Line);
                    string name = parser.Identifier();
                    var parameters = new List<(string, Expr?)>();
                    parser.Expect("(");
                    if (!parser.Accept(")"))
                    {
                        do
                        {
                            string p = parser.Identifier();
                            parameters.Add((p, parser.Accept("=") ? parser.ParseExpression() : null));
                        }
                        while (parser.Accept(","));
                        parser.Expect(")");
                    }

                    return new MacroNode(name, parameters, ParseUntil(out _, "endmacro"));
                }

                case "call":
                    return new CallBlockNode(new ExprParser(rest, token.Line).ParseComplete(), ParseUntil(out _, "endcall"));
                case "filter":
                {
                    var parser = new ExprParser(rest, token.Line);
                    string name = parser.Identifier();
                    var args = parser.Accept("(") ? parser.Arguments().Args : [];
                    return new FilterBlockNode(name, args, ParseUntil(out _, "endfilter"));
                }

                case "generation":
                    return new IfNode([(new Literal(true), ParseUntil(out _, "endgeneration"))]);
                case "break":
                    return new ControlNode(true);
                case "continue":
                    return new ControlNode(false);
                default:
                    throw new FormatException($"Unsupported statement {{% {word} %}} at line {token.Line}.");
            }
        }
    }

    private sealed class ExprParser
    {
        private readonly List<string> _tokens;
        private readonly int _line;
        private int _position;

        public ExprParser(string source, int line)
        {
            _tokens = Split(source, line);
            _line = line;
        }

        public Expr ParseComplete()
        {
            var e = ParseTuple();
            End();
            return e;
        }

        public void End()
        {
            if (_position < _tokens.Count)
            {
                throw Error($"unexpected '{_tokens[_position]}'");
            }
        }

        private string? Peek(int ahead = 0) => _position + ahead < _tokens.Count ? _tokens[_position + ahead] : null;

        public bool Accept(string token)
        {
            if (Peek() == token)
            {
                _position++;
                return true;
            }

            return false;
        }

        public bool AcceptKeyword(string word) => Accept(word);

        public void Expect(string token)
        {
            if (!Accept(token))
            {
                throw Error($"expected '{token}', found '{Peek() ?? "end"}'");
            }
        }

        public void Keyword(string word) => Expect(word);

        public string Identifier()
        {
            string? t = Peek();
            if (t is null || !(char.IsLetter(t[0]) || t[0] == '_'))
            {
                throw Error($"expected a name, found '{t ?? "end"}'");
            }

            _position++;
            return t;
        }

        private FormatException Error(string message) => new($"Template expression error at line {_line}: {message}.");

        // A tuple without brackets (as in "set a = 1, 2") becomes a list.
        public Expr ParseTuple()
        {
            var first = ParseExpression();
            if (Peek() != ",")
            {
                return first;
            }

            var items = new List<Expr> { first };
            while (Accept(","))
            {
                if (Peek() is null)
                {
                    break;
                }

                items.Add(ParseExpression());
            }

            return new ListExpr(items);
        }

        public Expr ParsePrimaryTarget()
        {
            Expr target = new Name(Identifier());
            while (Accept("."))
            {
                target = new Attribute(target, Identifier());
            }

            return target;
        }

        public Expr ParseExpression()
        {
            var value = ParseOr();
            if (AcceptKeyword("if"))
            {
                var condition = ParseOr();
                Expr? otherwise = AcceptKeyword("else") ? ParseExpression() : null;
                return new Conditional(condition, value, otherwise);
            }

            return value;
        }

        public Expr ParseFilteredOr() => ParseOr();

        public Expr ParseOr()
        {
            var left = ParseAnd();
            while (AcceptKeyword("or"))
            {
                left = new Binary("or", left, ParseAnd());
            }

            return left;
        }

        private Expr ParseAnd()
        {
            var left = ParseNot();
            while (AcceptKeyword("and"))
            {
                left = new Binary("and", left, ParseNot());
            }

            return left;
        }

        private Expr ParseNot() => AcceptKeyword("not") ? new Unary("not", ParseNot()) : ParseCompare();

        private Expr ParseCompare()
        {
            var left = ParseConcat();
            while (true)
            {
                if (Peek() is "==" or "!=" or "<" or ">" or "<=" or ">=")
                {
                    string op = _tokens[_position++];
                    left = new Binary(op, left, ParseConcat());
                }
                else if (Peek() == "in")
                {
                    _position++;
                    left = new Binary("in", left, ParseConcat());
                }
                else if (Peek() == "not" && Peek(1) == "in")
                {
                    _position += 2;
                    left = new Unary("not", new Binary("in", left, ParseConcat()));
                }
                else if (Peek() == "is")
                {
                    _position++;
                    bool negated = AcceptKeyword("not");
                    string test = Identifier();
                    var args = new List<Expr>();
                    if (Accept("("))
                    {
                        args = Arguments().Args;
                    }
                    else if (Peek() is { } next && (char.IsLetterOrDigit(next[0]) || next[0] is '\'' or '"') && next is not ("and" or "or" or "else" or "if" or "not"))
                    {
                        args.Add(ParseConcat());
                    }

                    left = new Test(left, test, args, negated);
                }
                else
                {
                    return left;
                }
            }
        }

        private Expr ParseConcat()
        {
            var left = ParseAdd();
            while (Accept("~"))
            {
                left = new Binary("~", left, ParseAdd());
            }

            return left;
        }

        private Expr ParseAdd()
        {
            var left = ParseMul();
            while (Peek() is "+" or "-")
            {
                string op = _tokens[_position++];
                left = new Binary(op, left, ParseMul());
            }

            return left;
        }

        private Expr ParseMul()
        {
            var left = ParsePower();
            while (Peek() is "*" or "/" or "//" or "%")
            {
                string op = _tokens[_position++];
                left = new Binary(op, left, ParsePower());
            }

            return left;
        }

        private Expr ParsePower()
        {
            var left = ParseUnary();
            while (Accept("**"))
            {
                left = new Binary("**", left, ParseUnary());
            }

            return left;
        }

        // As Jinja: a sign applies before filters, so -3|abs is abs(-3).
        private Expr ParseUnary(bool filters = true)
        {
            Expr value;
            if (Accept("-"))
            {
                value = new Unary("-", ParseUnary(filters: false));
            }
            else if (Accept("+"))
            {
                value = ParseUnary(filters: false);
            }
            else
            {
                value = ParsePostfix(ParsePrimary(), filters: false);
            }

            return filters ? ParsePostfix(value, filters: true) : value;
        }

        private Expr ParsePostfix(Expr value, bool filters)
        {
            while (true)
            {
                if (Accept("."))
                {
                    value = new Attribute(value, Identifier());
                }
                else if (Accept("["))
                {
                    Expr? start = Peek() is ":" ? null : ParseExpression();
                    if (Accept(":"))
                    {
                        Expr? stop = Peek() is ":" or "]" ? null : ParseExpression();
                        Expr? step = null;
                        if (Accept(":"))
                        {
                            step = Peek() == "]" ? null : ParseExpression();
                        }

                        Expect("]");
                        value = new Slice(value, start, stop, step);
                    }
                    else
                    {
                        Expect("]");
                        value = new Item(value, start!);
                    }
                }
                else if (Accept("("))
                {
                    var (args, kwargs) = Arguments();
                    value = new Call(value, args, kwargs);
                }
                else if (filters && Accept("|"))
                {
                    string name = Identifier();
                    var (args, kwargs) = Accept("(") ? Arguments() : ([], []);
                    value = new Filter(value, name, args, kwargs);
                }
                else
                {
                    return value;
                }
            }
        }

        // After "(": positional and keyword arguments, then ")".
        public (List<Expr> Args, List<(string Name, Expr Value)> Kwargs) Arguments()
        {
            var args = new List<Expr>();
            var kwargs = new List<(string, Expr)>();
            if (Accept(")"))
            {
                return (args, kwargs);
            }

            do
            {
                if (Peek(1) == "=" && Peek() is { } name && (char.IsLetter(name[0]) || name[0] == '_'))
                {
                    _position += 2;
                    kwargs.Add((name, ParseExpression()));
                }
                else
                {
                    args.Add(ParseExpression());
                }
            }
            while (Accept(",") && Peek() != ")");
            Expect(")");
            return (args, kwargs);
        }

        private Expr ParsePrimary()
        {
            string t = Peek() ?? throw Error("unexpected end of expression");
            _position++;
            if (t[0] is '\'' or '"')
            {
                var text = new StringBuilder(t[1..^1]);
                while (Peek() is { } more && more[0] is '\'' or '"')
                {
                    text.Append(more[1..^1]);                                   // adjacent strings are joined
                    _position++;
                }

                return new Literal(text.ToString());
            }

            if (char.IsDigit(t[0]))
            {
                return t.Contains('.') || t.Contains('e') || t.Contains('E') ? new Literal(double.Parse(t, CultureInfo.InvariantCulture)) : new Literal(long.Parse(t, CultureInfo.InvariantCulture));
            }

            switch (t)
            {
                case "(":
                {
                    if (Accept(")"))
                    {
                        return new ListExpr([], Tuple: true);
                    }

                    var inner = ParseExpression();
                    if (Accept(","))
                    {
                        var items = new List<Expr> { inner };
                        while (Peek() != ")")
                        {
                            items.Add(ParseExpression());
                            if (!Accept(","))
                            {
                                break;
                            }
                        }

                        Expect(")");
                        return new ListExpr(items, Tuple: true);
                    }

                    Expect(")");
                    return inner;
                }

                case "[":
                {
                    var items = new List<Expr>();
                    while (!Accept("]"))
                    {
                        items.Add(ParseExpression());
                        if (!Accept(","))
                        {
                            Expect("]");
                            break;
                        }
                    }

                    return new ListExpr(items);
                }

                case "{":
                {
                    var items = new List<(Expr, Expr)>();
                    while (!Accept("}"))
                    {
                        var key = ParseExpression();
                        Expect(":");
                        items.Add((key, ParseExpression()));
                        if (!Accept(","))
                        {
                            Expect("}");
                            break;
                        }
                    }

                    return new DictExpr(items);
                }

                case "true" or "True":
                    return new Literal(true);
                case "false" or "False":
                    return new Literal(false);
                case "none" or "None":
                    return new Literal(null);
            }

            if (char.IsLetter(t[0]) || t[0] == '_')
            {
                return new Name(t);
            }

            throw Error($"unexpected '{t}'");
        }

        private static List<string> Split(string source, int line)
        {
            var tokens = new List<string>();
            int i = 0;
            string[] operators = ["**", "//", "==", "!=", "<=", ">="];
            while (i < source.Length)
            {
                char c = source[i];
                if (char.IsWhiteSpace(c))
                {
                    i++;
                    continue;
                }

                if (c is '\'' or '"')
                {
                    var s = new StringBuilder();
                    s.Append(c);
                    i++;
                    while (i < source.Length && source[i] != c)
                    {
                        if (source[i] == '\\' && i + 1 < source.Length)
                        {
                            char e = source[++i];
                            s.Append(e switch { 'n' => '\n', 't' => '\t', 'r' => '\r', '0' => '\0', _ => e });
                        }
                        else
                        {
                            s.Append(source[i]);
                        }

                        i++;
                    }

                    if (i >= source.Length)
                    {
                        throw new FormatException($"Unterminated string at line {line}.");
                    }

                    s.Append(c);
                    i++;
                    tokens.Add(s.ToString());
                    continue;
                }

                if (char.IsDigit(c))
                {
                    int start = i;
                    while (i < source.Length && (char.IsDigit(source[i]) || source[i] == '.' && i + 1 < source.Length && char.IsDigit(source[i + 1])))
                    {
                        i++;
                    }

                    if (i < source.Length && source[i] is 'e' or 'E' && i + 1 < source.Length
                        && (char.IsDigit(source[i + 1]) || source[i + 1] is '+' or '-' && i + 2 < source.Length && char.IsDigit(source[i + 2])))
                    {
                        i += 2;
                        while (i < source.Length && char.IsDigit(source[i]))
                        {
                            i++;
                        }
                    }

                    tokens.Add(source[start..i]);
                    continue;
                }

                if (char.IsLetter(c) || c == '_')
                {
                    int start = i;
                    while (i < source.Length && (char.IsLetterOrDigit(source[i]) || source[i] == '_'))
                    {
                        i++;
                    }

                    tokens.Add(source[start..i]);
                    continue;
                }

                string? op = operators.FirstOrDefault(o => source.AsSpan(i).StartsWith(o, StringComparison.Ordinal));
                tokens.Add(op ?? c.ToString());
                i += op?.Length ?? 1;
            }

            return tokens;
        }
    }

    // ------------------------------------------------------------------ evaluation

    private sealed class Scope(Scope? parent)
    {
        private readonly Dictionary<string, object?> _values = [];

        public Scope? Parent => parent;

        public bool Has(string name) => _values.ContainsKey(name) || (parent?.Has(name) ?? false);

        public object? Get(string name) => _values.TryGetValue(name, out var v) ? v : parent is null ? Undefined.Value : parent.Get(name);

        public void Set(string name, object? value) => _values[name] = value;
    }

    private static class Interpreter
    {
        public static void Execute(List<Node> nodes, Scope scope, StringBuilder output)
        {
            foreach (var node in nodes)
            {
                Execute(node, scope, output);
            }
        }

        private static void Execute(Node node, Scope scope, StringBuilder output)
        {
            switch (node)
            {
                case TextNode t:
                    output.Append(t.Text);
                    break;
                case OutputNode o:
                    output.Append(Builtins.Str(Evaluate(o.Value, scope)));
                    break;
                case IfNode f:
                    foreach (var (condition, body) in f.Branches)
                    {
                        if (condition is null || Builtins.Truthy(Evaluate(condition, scope)))
                        {
                            Execute(body, scope, output);
                            break;
                        }
                    }

                    break;
                case ForNode f:
                    RunFor(f, scope, output);
                    break;
                case SetNode s:
                    object? value;
                    if (s.Body is not null)
                    {
                        var captured = new StringBuilder();
                        Execute(s.Body, scope, captured);
                        value = captured.ToString();
                    }
                    else
                    {
                        value = Evaluate(s.Value!, scope);
                    }

                    switch (s.Target)
                    {
                        case Name n:
                            scope.Set(n.Id, value);
                            break;
                        case Attribute { Target: var target, Id: var id } when Evaluate(target, scope) is Namespace ns:
                            ns.Values[id] = value;
                            break;
                        default:
                            throw new InvalidOperationException("Only names and namespace attributes can be set.");
                    }

                    break;
                case MacroNode m:
                    scope.Set(m.Name, MakeMacro(m, scope));
                    break;
                case CallBlockNode c:
                    var callScope = new Scope(scope);
                    var caller = new StringBuilder();
                    callScope.Set("caller", (Callable)((_, _) =>
                    {
                        caller.Clear();
                        Execute(c.Body, new Scope(scope), caller);
                        return caller.ToString();
                    }));
                    output.Append(Builtins.Str(Evaluate(c.Call, callScope)));
                    break;
                case FilterBlockNode fb:
                    var inner = new StringBuilder();
                    Execute(fb.Body, scope, inner);
                    output.Append(Builtins.Str(Builtins.ApplyFilter(fb.Filter, inner.ToString(), [.. fb.Args.Select(a => Evaluate(a, scope))], [], scope)));
                    break;
                case ControlNode c:
                    throw c.Break ? new LoopBreak() : new LoopContinue();
            }
        }

        private static Callable MakeMacro(MacroNode m, Scope definition) => (args, kwargs) =>
        {
            var local = new Scope(definition);
            for (int i = 0; i < m.Parameters.Count; i++)
            {
                var (name, fallback) = m.Parameters[i];
                local.Set(name, i < args.Count ? args[i] : kwargs.TryGetValue(name, out var v) ? v : fallback is null ? Undefined.Value : Evaluate(fallback, definition));
            }

            local.Set("varargs", args.Skip(m.Parameters.Count).ToList<object?>());
            local.Set("kwargs", kwargs.Where(k => m.Parameters.All(p => p.Name != k.Key)).ToDictionary(k => k.Key, k => k.Value));
            var output = new StringBuilder();
            Execute(m.Body, local, output);
            return output.ToString();
        };

        private static void RunFor(ForNode f, Scope scope, StringBuilder output)
        {
            var items = Builtins.Iterate(Evaluate(f.Iterable, scope)).ToList();
            if (f.Filter is not null)
            {
                items = [.. items.Where(item =>
                {
                    var probe = new Scope(scope);
                    Bind(f.Targets, item, probe);
                    return Builtins.Truthy(Evaluate(f.Filter, probe));
                })];
            }

            if (items.Count == 0)
            {
                if (f.Else is not null)
                {
                    Execute(f.Else, scope, output);
                }

                return;
            }

            // Jinja scoping: assignments inside the loop do not leak out (namespaces carry state).
            for (int i = 0; i < items.Count; i++)
            {
                var local = new Scope(scope);
                Bind(f.Targets, items[i], local);
                local.Set("loop", new Dictionary<string, object?>
                {
                    ["index"] = (long)i + 1, ["index0"] = (long)i, ["revindex"] = (long)(items.Count - i), ["revindex0"] = (long)(items.Count - i - 1),
                    ["first"] = i == 0, ["last"] = i == items.Count - 1, ["length"] = (long)items.Count,
                    ["previtem"] = i > 0 ? items[i - 1] : Undefined.Value, ["nextitem"] = i < items.Count - 1 ? items[i + 1] : Undefined.Value,
                });
                try
                {
                    Execute(f.Body, local, output);
                }
                catch (LoopContinue)
                {
                }
                catch (LoopBreak)
                {
                    break;
                }
            }
        }

        private static void Bind(List<string> targets, object? item, Scope scope)
        {
            if (targets.Count == 1)
            {
                scope.Set(targets[0], item);
                return;
            }

            var parts = Builtins.Iterate(item).ToList();
            for (int i = 0; i < targets.Count; i++)
            {
                scope.Set(targets[i], i < parts.Count ? parts[i] : Undefined.Value);
            }
        }

        public static object? Evaluate(Expr expr, Scope scope)
        {
            switch (expr)
            {
                case Literal l:
                    return l.Value;
                case Name n:
                    return scope.Get(n.Id);
                case ListExpr l:
                    var list = l.Items.Select(i => Evaluate(i, scope));
                    return l.Tuple ? new Tuple(list) : list.ToList();
                case DictExpr d:
                    var dict = new Dictionary<string, object?>();
                    foreach (var (key, value) in d.Items)
                    {
                        dict[Builtins.Str(Evaluate(key, scope))] = Evaluate(value, scope);
                    }

                    return dict;
                case Attribute a:
                    return Builtins.GetAttribute(Evaluate(a.Target, scope), a.Id);
                case Item i:
                    return Builtins.GetItem(Evaluate(i.Target, scope), Evaluate(i.Index, scope));
                case Slice s:
                    return Builtins.SliceOf(Evaluate(s.Target, scope), s.Start is null ? null : Evaluate(s.Start, scope),
                        s.Stop is null ? null : Evaluate(s.Stop, scope), s.Step is null ? null : Evaluate(s.Step, scope));
                case Call c:
                    var args = c.Args.Select(a => Evaluate(a, scope)).ToList();
                    var kwargs = c.Kwargs.ToDictionary(k => k.Name, k => Evaluate(k.Value, scope));
                    if (c.Target is Attribute method)
                    {
                        var target = Evaluate(method.Target, scope);
                        if (Builtins.TryMethod(target, method.Id, args, out var result))
                        {
                            return result;
                        }
                    }

                    return Evaluate(c.Target, scope) is Callable callable ? callable(args, kwargs)
                        : throw new InvalidOperationException($"'{Describe(c.Target)}' is not callable.");
                case Filter f:
                    return Builtins.ApplyFilter(f.Id, Evaluate(f.Target, scope), [.. f.Args.Select(a => Evaluate(a, scope))],
                        f.Kwargs.ToDictionary(k => k.Name, k => Evaluate(k.Value, scope)), scope);
                case Test t:
                    bool passed = Builtins.RunTest(t.Id, Evaluate(t.Target, scope), [.. t.Args.Select(a => Evaluate(a, scope))]);
                    return passed != t.Negated;
                case Unary u:
                    var operand = Evaluate(u.Operand, scope);
                    return u.Op == "not" ? !Builtins.Truthy(operand) : Builtins.Negate(operand);
                case Binary b when b.Op == "and":
                    var left = Evaluate(b.Left, scope);
                    return Builtins.Truthy(left) ? Evaluate(b.Right, scope) : left;
                case Binary b when b.Op == "or":
                    var first = Evaluate(b.Left, scope);
                    return Builtins.Truthy(first) ? first : Evaluate(b.Right, scope);
                case Binary b:
                    return Builtins.Operate(b.Op, Evaluate(b.Left, scope), Evaluate(b.Right, scope));
                case Conditional c:
                    return Builtins.Truthy(Evaluate(c.Condition, scope)) ? Evaluate(c.Then, scope) : c.Else is null ? Undefined.Value : Evaluate(c.Else, scope);
                default:
                    throw new InvalidOperationException($"Unknown expression {expr}.");
            }
        }

        private static string Describe(Expr e) => e switch { Name n => n.Id, Attribute a => a.Id, _ => e.GetType().Name };
    }
}
