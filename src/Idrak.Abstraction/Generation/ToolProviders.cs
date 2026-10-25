// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// What a tool assembly offers a host that loads it (the idrak tool's <c>--tools</c>, <c>tools list</c>,
/// <c>tools test</c> and <c>mcp serve</c>): its tools. The assembly declares each provider with
/// <see cref="ToolProviderAttribute"/>, and <see cref="ToolProviders.Of"/> reads them once per assembly; a provider whose
/// tools are methods marked <see cref="ToolAttribute"/> returns <see cref="ToolProviders.FromMethods"/>.
/// </summary>
public interface IToolProvider
{
    /// <summary>The tools, in the order they are offered: any <see cref="ITool"/> (a <see cref="Tool"/> made from a function, or a type of its own).</summary>
    IReadOnlyList<ITool> Tools { get; }
}

/// <summary>
/// Names one of an assembly's tool providers: <c>[assembly: ToolProvider(typeof(MyTools))]</c>, where
/// <c>MyTools : IToolProvider</c> has a public parameterless constructor. An assembly may name several; their tools come
/// in the attributes' order.
/// </summary>
/// <param name="provider">The implementation of <see cref="IToolProvider"/>.</param>
[AttributeUsage(AttributeTargets.Assembly, AllowMultiple = true)]
public sealed class ToolProviderAttribute([DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor)] Type provider) : Attribute
{
    /// <summary>The implementation of <see cref="IToolProvider"/>.</summary>
    [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor)]
    public Type Provider { get; } = provider;
}

/// <summary>The tools an assembly declares (<see cref="ToolProviderAttribute"/>), and the tools of an object's <see cref="ToolAttribute"/> methods.</summary>
public static class ToolProviders
{
    // Each assembly's tools, its attributes read and its providers made once.
    private static readonly ConditionalWeakTable<Assembly, IReadOnlyList<ITool>> Declared = new();

    /// <summary>
    /// The tools <paramref name="assembly"/> declares: those of each provider its <see cref="ToolProviderAttribute"/>s
    /// name, in the attributes' order, each provider made once; the attributes are read once per assembly. Empty when it
    /// declares none.
    /// </summary>
    /// <exception cref="InvalidOperationException">An attribute names a type that is not an <see cref="IToolProvider"/>.</exception>
    public static IReadOnlyList<ITool> Of(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        return Declared.GetValue(assembly, static a => Read(a));
    }

    /// <summary>
    /// The public methods of <paramref name="instance"/>'s type marked <see cref="ToolAttribute"/>, static and instance
    /// ones (on <paramref name="instance"/>), as tools (<see cref="Tool.FromDelegate"/>), in the type's order: what a
    /// provider whose tools are methods returns from <see cref="IToolProvider.Tools"/>.
    /// </summary>
    [RequiresUnreferencedCode("See Tool.FromDelegate.")]
    [RequiresDynamicCode("See Tool.FromDelegate.")]
    public static IReadOnlyList<ITool> FromMethods(object instance)
    {
        ArgumentNullException.ThrowIfNull(instance);
        var tools = new List<ITool>();
        foreach (var method in instance.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static))
        {
            if (method.GetCustomAttribute<ToolAttribute>() is { } attribute && !method.IsGenericMethodDefinition)
            {
                tools.Add(Tool.FromMethod(attribute.Name, attribute.Description, method, method.IsStatic ? null : instance));
            }
        }

        return tools;
    }

    private static List<ITool> Read(Assembly assembly)
    {
        var tools = new List<ITool>();
        foreach (var declared in assembly.GetCustomAttributes<ToolProviderAttribute>())
        {
            var provider = Activator.CreateInstance(declared.Provider) as IToolProvider
                ?? throw new InvalidOperationException($"{assembly.GetName().Name}: [assembly: ToolProvider(typeof({declared.Provider.FullName}))] names a type that is not an "
                                                       + $"{nameof(IToolProvider)}.");
            tools.AddRange(provider.Tools);
        }

        return tools;
    }
}
