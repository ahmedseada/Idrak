// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;

namespace Idrak.Abstraction.Serving;

/// <summary>
/// Named models of any kind, reached through the contracts their kinds implement (<see cref="IPredictor{TIn, TOut}"/>,
/// <see cref="Generation.ITextModel"/>, <see cref="Generation.IChatModel"/>, <see cref="Retrieval.IEmbedder"/>, ...).
/// The inference engine implements it; bridges such as the MCP server take it, so they serve the models without the
/// engine's own type.
/// </summary>
public interface IModelCatalog
{
    /// <summary>The names of the models.</summary>
    IReadOnlyCollection<string> Names { get; }

    /// <summary>The kind of the model named <paramref name="name"/> (<see cref="EngineModel{TCopy}.Kind"/>: "predictor", "text", "chat", ...).</summary>
    /// <exception cref="KeyNotFoundException">There is no such model.</exception>
    string KindOf(string name);

    /// <summary>The model named <paramref name="name"/> as <typeparamref name="T"/>, a contract its kind implements.</summary>
    /// <exception cref="KeyNotFoundException">There is no such model.</exception>
    /// <exception cref="InvalidOperationException">The model's kind does not implement <typeparamref name="T"/>.</exception>
    T Model<T>(string name)
        where T : class;

    /// <summary>The model named <paramref name="name"/> as <typeparamref name="T"/>, if there is one and its kind implements <typeparamref name="T"/>.</summary>
    bool TryGetModel<T>(string name, [NotNullWhen(true)] out T? model)
        where T : class;
}
