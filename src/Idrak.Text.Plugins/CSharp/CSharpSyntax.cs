// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace Idrak.Text.Plugins.CSharp;

/// <summary>A method with a body declared in a type.</summary>
/// <param name="Node">Its declaration.</param>
/// <param name="Name">Its name (without type parameters).</param>
/// <param name="Type">The type it is declared in (nested types joined with '.').</param>
/// <param name="Namespace">Its namespace, or "" for none.</param>
internal sealed record CSharpMethod(MethodDeclarationSyntax Node, string Name, string Type, string Namespace)
{
    /// <summary>Its signature: modifiers, return type, name, type parameters, parameters and constraints, white space normalized (made when asked).</summary>
    public string Header => CSharpSyntax.Signature(Node);

    /// <summary>Whether it is <c>async</c>.</summary>
    public bool IsAsync => Node.Modifiers.Any(SyntaxKind.AsyncKeyword);

    /// <summary>Whether its body is an expression (<c>=&gt; …;</c>) rather than a block.</summary>
    public bool Expression => Node.Body is null;

    /// <summary>Its full name: namespace, type and name, joined with '.'.</summary>
    public string FullName => Namespace.Length > 0 ? $"{Namespace}.{Type}.{Name}" : $"{Type}.{Name}";
}

/// <summary>How a test is named and run.</summary>
internal enum CSharpTestKind
{
    /// <summary>A named entry of a test list, <c>("group: what it checks", d =&gt; …)</c>, run by its name through the runner's filter variable.</summary>
    Named,

    /// <summary>A test method marked <c>[Fact]</c>, <c>[Theory]</c>, <c>[Test]</c>, <c>[TestCase]</c>, <c>[TestMethod]</c> or <c>[DataTestMethod]</c>, run by <c>dotnet test --filter</c>.</summary>
    Attribute,
}

/// <summary>A test found in a file: its name, how it runs, and its code.</summary>
/// <param name="Name">Its name: a named entry's text, or a test method's full name.</param>
/// <param name="Kind">How it is named and run.</param>
/// <param name="Body">Its code (a lambda's body, a method group's name, a test method's body).</param>
/// <param name="Method">The method a named entry runs by name (a method group), or null for a lambda.</param>
internal sealed record CSharpTestEntry(string Name, CSharpTestKind Kind, SyntaxNode Body, string? Method);

/// <summary>
/// A C# file read once as a Roslyn syntax tree (<c>Microsoft.CodeAnalysis.CSharp</c>, syntax only: no compilation and no
/// semantic model, which the task families do not need): its methods with bodies, its tests, the names its code calls, and
/// the file with a method's body removed, as bytes, every other byte as it was.
/// </summary>
internal sealed class CSharpSyntax
{
    private static readonly HashSet<string> TestAttributes = new(StringComparer.Ordinal) { "Fact", "Theory", "Test", "TestCase", "TestMethod", "DataTestMethod" };

    private static readonly CSharpParseOptions Options = new(LanguageVersion.Preview, DocumentationMode.Parse);

    private readonly SourceText _text;
    private readonly int _preamble;

    private CSharpSyntax(byte[] bytes)
    {
        Bytes = bytes;
        _preamble = bytes.AsSpan().StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]) ? 3 : 0;
        _text = SourceText.From(bytes, bytes.Length, Encoding.UTF8);
        var root = CSharpSyntaxTree.ParseText(_text, Options).GetRoot();
        var methods = new List<CSharpMethod>();
        var tests = new List<CSharpTestEntry>();
        foreach (var node in root.DescendantNodes(n => n is not (BlockSyntax or ArrowExpressionClauseSyntax) || n.Parent is not MethodDeclarationSyntax))
        {
            switch (node)
            {
                case MethodDeclarationSyntax method when method.Body is not null || method.ExpressionBody is not null:
                    var found = new CSharpMethod(method, method.Identifier.ValueText, TypePath(method), NamespacePath(method));
                    methods.Add(found);
                    if (method.AttributeLists.SelectMany(l => l.Attributes).Any(a => TestAttributes.Contains(AttributeName(a))))
                    {
                        tests.Add(new CSharpTestEntry(found.FullName, CSharpTestKind.Attribute, (SyntaxNode?)method.Body ?? method.ExpressionBody!, method.Identifier.ValueText));
                    }

                    break;
                case TupleExpressionSyntax tuple when NamedEntry(tuple) is { } entry:
                    tests.Add(entry);
                    break;
            }
        }

        Methods = methods;
        Tests = tests;
    }

    /// <summary>The file's bytes.</summary>
    public byte[] Bytes { get; }

    /// <summary>The methods with bodies declared in its types, in file order (constructors, operators, properties and local functions are not among them).</summary>
    public IReadOnlyList<CSharpMethod> Methods { get; }

    /// <summary>The tests it declares, in file order.</summary>
    public IReadOnlyList<CSharpTestEntry> Tests { get; }

    /// <summary>Reads <paramref name="bytes"/> (a C# file's UTF-8 bytes) into its syntax tree, once.</summary>
    public static CSharpSyntax Read(byte[] bytes) => new(bytes ?? throw new ArgumentNullException(nameof(bytes)));

    /// <summary>The names <paramref name="node"/> calls (an invoked identifier, member or generic name), in the order first called.</summary>
    public static IReadOnlyList<string> CalledNames(SyntaxNode node)
    {
        ArgumentNullException.ThrowIfNull(node);
        var names = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        void Add(string name)
        {
            if (seen.Add(name))
            {
                names.Add(name);
            }
        }

        if (node is IdentifierNameSyntax group)
        {
            Add(group.Identifier.ValueText);                                 // a method group: the method it names
        }

        foreach (var invocation in node.DescendantNodesAndSelf().OfType<InvocationExpressionSyntax>())
        {
            if (Called(invocation.Expression) is { } name)
            {
                Add(name);
            }
        }

        return names;
    }

    private static string? Called(ExpressionSyntax expression) => expression switch
    {
        IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
        GenericNameSyntax generic => generic.Identifier.ValueText,
        MemberAccessExpressionSyntax member => member.Name.Identifier.ValueText,
        MemberBindingExpressionSyntax binding => binding.Name.Identifier.ValueText,
        _ => null,
    };

    /// <summary>
    /// The file with <paramref name="method"/>'s body replaced by one that throws <c>NotImplementedException</c> (its header and
    /// documentation kept; an async method's awaits once first, so it still compiles warning-free), as bytes.
    /// </summary>
    public byte[] WithoutBody(CSharpMethod method)
    {
        ArgumentNullException.ThrowIfNull(method);
        const string Throw = "throw new global::System.NotImplementedException()";
        string awaited = method.IsAsync ? "await global::System.Threading.Tasks.Task.CompletedTask; " : "";
        TextSpan span;
        string replacement;
        if (method.Node.Body is { } block)
        {
            span = TextSpan.FromBounds(block.OpenBraceToken.Span.End, block.CloseBraceToken.SpanStart);
            replacement = $" {awaited}{Throw}; ";
        }
        else
        {
            var arrow = method.Node.ExpressionBody!;
            var semicolon = method.Node.SemicolonToken;
            if (method.IsAsync)
            {
                span = TextSpan.FromBounds(arrow.SpanStart, semicolon.Span.End);       // "=> e;" becomes a block that awaits and throws
                replacement = $"{{ {awaited}{Throw}; }}";
            }
            else
            {
                span = arrow.Expression.Span;
                replacement = Throw;
            }
        }

        int from = ByteOffset(span.Start), to = ByteOffset(span.End);
        byte[] inserted = Encoding.UTF8.GetBytes(replacement);
        var result = new byte[Bytes.Length - (to - from) + inserted.Length];
        Bytes.AsSpan(0, from).CopyTo(result);
        inserted.CopyTo(result.AsSpan(from));
        Bytes.AsSpan(to).CopyTo(result.AsSpan(from + inserted.Length));
        return result;
    }

    // The byte of the file where a character of the text starts (the text is the file's UTF-8, its preamble left out).
    private int ByteOffset(int position) => _preamble + Encoding.UTF8.GetByteCount(_text.ToString(new TextSpan(0, position)));

    /// <summary>How much the body does: the characters of its tokens (its braces, arrow and semicolon left out).</summary>
    public static int BodySize(CSharpMethod method)
    {
        ArgumentNullException.ThrowIfNull(method);
        SyntaxNode body = (SyntaxNode?)method.Node.Body ?? method.Node.ExpressionBody!.Expression;
        int size = 0;
        foreach (var token in body.DescendantTokens())
        {
            size += token.Span.Length;
        }

        return method.Node.Body is null ? size : size - 2;
    }

    /// <summary>Whether the body already throws <c>NotImplementedException</c> (nothing to remove).</summary>
    public static bool Unimplemented(CSharpMethod method)
    {
        ArgumentNullException.ThrowIfNull(method);
        SyntaxNode body = (SyntaxNode?)method.Node.Body ?? method.Node.ExpressionBody!;
        return body.DescendantNodes().OfType<IdentifierNameSyntax>().Any(n => n.Identifier.ValueText == "NotImplementedException");
    }

    /// <summary>The method's <c>///</c> documentation, as written (empty when it has none).</summary>
    public static string Documentation(CSharpMethod method)
    {
        ArgumentNullException.ThrowIfNull(method);
        var lines = method.Node.GetLeadingTrivia()
            .Where(t => t.IsKind(SyntaxKind.SingleLineDocumentationCommentTrivia) || t.IsKind(SyntaxKind.MultiLineDocumentationCommentTrivia))
            .SelectMany(t => t.ToFullString().Split('\n'))
            .Select(l => l.Trim())
            .Where(l => l.Length > 0)
            .Select(l => l.StartsWith("///", StringComparison.Ordinal) ? l : "/// " + l);
        return string.Join("\n", lines);
    }

    // ("group: what", d => …) or ("group: what", Method): a named test entry.
    private static CSharpTestEntry? NamedEntry(TupleExpressionSyntax tuple)
    {
        if (tuple.Arguments.Count < 2 || tuple.Arguments[0].Expression is not LiteralExpressionSyntax { RawKind: (int)SyntaxKind.StringLiteralExpression } literal)
        {
            return null;
        }

        string name = literal.Token.ValueText;
        if (!name.Contains(": ", StringComparison.Ordinal))
        {
            return null;
        }

        return tuple.Arguments[1].Expression switch
        {
            LambdaExpressionSyntax lambda => new CSharpTestEntry(name, CSharpTestKind.Named, lambda.Body, null),
            IdentifierNameSyntax group => new CSharpTestEntry(name, CSharpTestKind.Named, group, group.Identifier.ValueText),
            _ => null,
        };
    }

    private static string AttributeName(AttributeSyntax attribute)
    {
        string name = attribute.Name switch
        {
            QualifiedNameSyntax qualified => qualified.Right.Identifier.ValueText,
            SimpleNameSyntax simple => simple.Identifier.ValueText,
            _ => attribute.Name.ToString(),
        };
        return name.EndsWith("Attribute", StringComparison.Ordinal) ? name[..^"Attribute".Length] : name;
    }

    private static string TypePath(SyntaxNode node) =>
        string.Join(".", node.Ancestors().OfType<BaseTypeDeclarationSyntax>().Reverse().Select(t => t.Identifier.ValueText));

    private static string NamespacePath(SyntaxNode node) =>
        string.Join(".", node.Ancestors().OfType<BaseNamespaceDeclarationSyntax>().Reverse().Select(n => n.Name.ToString()));

    /// <summary>The method's signature: its declaration without attributes, body or trivia, white space normalized.</summary>
    public static string Signature(MethodDeclarationSyntax method)
    {
        ArgumentNullException.ThrowIfNull(method);
        var bare = method.WithAttributeLists(default).WithBody(null).WithExpressionBody(null).WithSemicolonToken(default).WithoutTrivia();
        return bare.NormalizeWhitespace().ToString().Trim();
    }
}
