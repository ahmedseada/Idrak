namespace Idrak.Datasets;

/// <summary>
/// Rewrites a text by the consumer's rules (case, spacing, punctuation, a language's letter forms and digits …) so texts
/// that mean the same compare equal. The library has no rules of its own: implement this and apply it with
/// <see cref="Dataset.Normalize(ITextNormalizer, string[])"/> before the steps that compare values (<see cref="Dataset.Deduplicate"/>,
/// a split key). Implementations are called from one thread per enumeration; keep them free of shared mutable state.
/// </summary>
public interface ITextNormalizer
{
    /// <summary>The normalized form of <paramref name="text"/>.</summary>
    string Normalize(string text);
}
