// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak;

public sealed partial class Tensor
{
    // ---------------------------------------------------------------- writing in place
    //
    // For optimizers, initializers, caches and weight formats of one's own: these change the tensor's values without
    // making a new tensor, and autograd does not record them. They refuse a tensor computed by a recorded operation (a
    // change autograd would not see), and a backward step that read the values before the change refuses to run.

    /// <summary>
    /// Overwrites the elements with <paramref name="values"/> (row-major, as many as <see cref="Size"/>), in place.
    /// Not recorded by autograd (see the remarks on <see cref="Fill"/>).
    /// </summary>
    public void CopyFrom(ReadOnlySpan<float> values)
    {
        if (values.Length != Size)
        {
            throw new ArgumentException($"{values.Length} values cannot fill a tensor of shape {FormatShape(_shape)} ({Size} elements).", nameof(values));
        }

        BeginWrite(nameof(CopyFrom));
        Backend.Upload(values, Storage);
    }

    /// <summary>
    /// Overwrites the elements with <paramref name="source"/>'s (the same number of elements, row-major; any device), in
    /// place. Not recorded by autograd (see the remarks on <see cref="Fill"/>).
    /// </summary>
    public void CopyFrom(Tensor source)
    {
        ArgumentNullException.ThrowIfNull(source);
        source.ThrowIfDisposed();
        CheckSameSize(source, nameof(CopyFrom));
        BeginWrite(nameof(CopyFrom));
        if (source.Device == Device)
        {
            Backend.Copy(source.Storage, Storage, Size);
        }
        else
        {
            Backend.Upload(source.ToArray(), Storage);
        }
    }

    /// <summary>Sets every element to <paramref name="value"/>, in place.</summary>
    /// <remarks>
    /// Like the other in-place writes (<see cref="CopyFrom(ReadOnlySpan{float})"/>, <see cref="Scale"/>,
    /// <see cref="AddScaled"/>), it is not recorded by autograd. It is refused on a tensor computed by a recorded operation
    /// (one that is not <see cref="IsLeaf"/>); and if an operation recorded before the write reads this tensor in its
    /// backward step, <see cref="Backward()"/> throws instead of computing a gradient from the new values. Write after
    /// <see cref="Backward()"/>, as optimizers do.
    /// </remarks>
    public void Fill(float value)
    {
        BeginWrite(nameof(Fill));
        Backend.Fill(Storage, Size, value);
    }

    /// <summary>Multiplies every element by <paramref name="factor"/>, in place (see the remarks on <see cref="Fill"/>).</summary>
    public void Scale(float factor)
    {
        BeginWrite(nameof(Scale));
        Backend.Affine(Storage, Storage, Size, factor, 0f);
    }

    /// <summary>
    /// Adds <paramref name="scale"/> · <paramref name="other"/> (the same number of elements, on the same device) to this
    /// tensor, in place: <c>p.AddScaled(update, -learningRate)</c> is a gradient-descent step (see the remarks on
    /// <see cref="Fill"/>).
    /// </summary>
    public void AddScaled(Tensor other, float scale = 1f)
    {
        ArgumentNullException.ThrowIfNull(other);
        other.ThrowIfDisposed();
        CheckSameDevice(this, other);
        CheckSameSize(other, nameof(AddScaled));
        BeginWrite(nameof(AddScaled));
        Backend.Axpy(other.Storage, Storage, Size, scale);
    }

    // Checks the write is allowed (after the arguments were checked) and counts it.
    private void BeginWrite(string method)
    {
        ThrowIfDisposed();
        if (!IsLeaf)
        {
            throw new InvalidOperationException(
                $"{method} cannot change a tensor computed by a recorded operation: autograd would not see the change. "
                + "Write into a leaf tensor (a parameter, a buffer, a Clone()), or compute this one under Autograd.NoGrad().");
        }

        Interlocked.Increment(ref Storage.Version);
    }

    private void CheckSameSize(Tensor other, string method)
    {
        if (other.Size != Size)
        {
            throw new ArgumentException($"{method}: a tensor of shape {FormatShape(other._shape)} ({other.Size} elements) does not match {FormatShape(_shape)} ({Size} elements).");
        }
    }
}
