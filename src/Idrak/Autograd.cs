namespace Idrak;

/// <summary>Global switches for automatic differentiation.</summary>
public static class Autograd
{
    [ThreadStatic]
    private static int t_disabledDepth;

    /// <summary>Whether operations on the current thread record the graph needed for <see cref="Tensor.Backward()"/>.</summary>
    public static bool IsEnabled => t_disabledDepth == 0;

    /// <summary>
    /// Turns gradient recording off on this thread until the returned scope is disposed.
    /// Use it for inference and evaluation: it is faster and allocates less.
    /// </summary>
    /// <example><c>using (Autograd.NoGrad()) { var prediction = model.Forward(x); }</c></example>
    public static NoGradScope NoGrad()
    {
        t_disabledDepth++;
        return new NoGradScope(true);
    }

    /// <summary>Restores gradient recording when disposed.</summary>
    public readonly struct NoGradScope : IDisposable
    {
        private readonly bool _active;

        internal NoGradScope(bool active) => _active = active;

        /// <inheritdoc />
        public void Dispose()
        {
            if (_active)
            {
                t_disabledDepth--;
            }
        }
    }
}
