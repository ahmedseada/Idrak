// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Types;
using Idrak.Types;

namespace Idrak;

/// <summary>
/// The library's own types, handed over as one immutable list: each defined by itself, in its own class (its description,
/// its name), presented through <see cref="IElementType"/>. This list holds no definition and takes no other type; a
/// package of its own types hands over a list of its own.
/// </summary>
public sealed class BuiltInTypes : ITypeProvider
{
    private BuiltInTypes()
    {
    }

    /// <summary>The library's types.</summary>
    public static BuiltInTypes All { get; } = new();

    /// <inheritdoc />
    public IReadOnlyList<IElementType> Types { get; } = Array.AsReadOnly<IElementType>(
    [
        F32.Type, BF16.Type, F16.Type, F64.Type, F8E4M3.Type, F8E5M2.Type, F4E2M1.Type, F8E8M0.Type,
        I8.Type, U8.Type, I16.Type, U16.Type, I32.Type, U32.Type, I64.Type, U64.Type, I4.Type, U2.Type, Bool.Type,
        Int2.Type, MXFP4.Type,
        Q8_0.Type, Q8_1.Type, Q4_0.Type, Q4_1.Type, Q5_0.Type, Q5_1.Type, IQ4_NL.Type,
        Q2_K.Type, Q3_K.Type, Q4_K.Type, Q5_K.Type, Q6_K.Type, Q8_K.Type,
        IQ2_XXS.Type, IQ2_XS.Type, IQ2_S.Type, IQ3_XXS.Type, IQ3_S.Type, IQ4_XS.Type, IQ1_S.Type, IQ1_M.Type,
        TQ1_0.Type, TQ2_0.Type,
    ]);
}