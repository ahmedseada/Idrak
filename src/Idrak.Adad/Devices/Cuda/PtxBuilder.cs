// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Adad.Devices.Cuda;

/// <summary>A kernel's text with numbered registers (ported from Adad's PtxBuilder): each R / RD / F / P a new register, declared in Finish.</summary>
internal sealed class PtxBuilder(string prefix = "")
{
    private readonly StringBuilder _body = new();
    private int _r, _rd, _f, _p, _h;

    // Registers are %{prefix}r0, %{prefix}rd0, …: a prefix keeps a builder's registers apart from those of a kernel written
    // by hand that its lines are emitted into (Take, Registers).
    private readonly string _prefix = prefix;

    public string R() => string.Create(CultureInfo.InvariantCulture, $"%{_prefix}r{_r++}");

    public string RD() => string.Create(CultureInfo.InvariantCulture, $"%{_prefix}rd{_rd++}");

    public string F() => string.Create(CultureInfo.InvariantCulture, $"%{_prefix}f{_f++}");

    public string P() => string.Create(CultureInfo.InvariantCulture, $"%{_prefix}p{_p++}");

    /// <summary>A 16-bit register (a half's bits, a byte to store).</summary>
    public string H() => string.Create(CultureInfo.InvariantCulture, $"%{_prefix}h{_h++}");

    /// <summary>The lines written since the last Take (for another kernel's body), and none kept.</summary>
    public string Take()
    {
        string lines = _body.ToString();
        _body.Clear();
        return lines;
    }

    /// <summary>The declarations of every register handed out so far (for another kernel's register section).</summary>
    public string Registers() => string.Create(CultureInfo.InvariantCulture,
        $"    .reg .pred %{_prefix}p<{Math.Max(1, _p)}>;\n    .reg .b32 %{_prefix}r<{Math.Max(1, _r)}>;\n    .reg .b64 %{_prefix}rd<{Math.Max(1, _rd)}>;\n    .reg .f32 %{_prefix}f<{Math.Max(1, _f)}>;\n    .reg .b16 %{_prefix}h<{Math.Max(1, _h)}>;\n");

    public void L(string line) => _body.Append("    ").Append(line).Append('\n');

    public void Label(string name) => _body.Append(name).Append(":\n");

    private readonly StringBuilder _declarations = new();

    /// <summary>Declarations of the kernel's own beyond the builder's registers (an epilogue's), emitted with them.</summary>
    public void Declare(string line) => _declarations.Append("    ").Append(line).Append('\n');

    /// <summary>
    /// The module: <paramref name="header"/>, the entry with its parameters and registers, shared memory (static up to 48
    /// KiB, else the module's dynamic block); <paramref name="directives"/> (Adad's: performance directives such as
    /// .minnctapersm) after the parameters, none by default (the copied kernels' text unchanged).
    /// </summary>
    public string Finish(string header, string entry, string parameters, int sharedBytes = 0, string directives = "")
    {
        var sb = new StringBuilder(header);
        if (sharedBytes > 48 * 1024)
        {
            sb.Append(".extern .shared .align 16 .b8 smem[];\n");
        }

        sb.Append(".visible .entry ").Append(entry).Append('(').Append(parameters).Append(')').Append(directives).Append("\n{\n");
        sb.Append(CultureInfo.InvariantCulture, $"    .reg .pred %p<{Math.Max(1, _p)}>;\n    .reg .b32 %r<{Math.Max(1, _r)}>;\n    .reg .b64 %rd<{Math.Max(1, _rd)}>;\n");
        sb.Append(CultureInfo.InvariantCulture, $"    .reg .f32 %f<{Math.Max(1, _f)}>;\n");
        if (_h > 0)
        {
            sb.Append(CultureInfo.InvariantCulture, $"    .reg .b16 %h<{_h}>;\n");
        }

        sb.Append(_declarations);
        if (sharedBytes is > 0 and <= 48 * 1024)
        {
            sb.Append(CultureInfo.InvariantCulture, $"    .shared .align 16 .b8 smem[{sharedBytes}];\n");
        }

        sb.Append(_body).Append("}\n");
        return sb.ToString();
    }
}
