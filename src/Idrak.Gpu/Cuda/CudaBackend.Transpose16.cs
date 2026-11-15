// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Gpu.Cuda;

// A block of 16-bit rows uploaded and transposed on the device (Backend.UploadTransposed16): a bfloat16 weight whose layout
// is its stored matrix's transpose (a tied head from its table) is made a block of rows at a time, with no host copy of the
// whole (BFloat16Weight.FromStoredRows). A 32 × 32 tile a block through shared memory, so reads and writes both coalesce.
internal sealed unsafe partial class CudaBackend
{
    private IntPtr? _transpose16;
    private bool _transpose16Tried;

    private const string Transpose16Name = "transpose16_rows";

    // dst[c · stride + first + r] = src[r · columns + c] for r < count, c < columns: a thread block per 32 × 32 tile
    // (32 × 8 threads, 4 rows each), the tile staged in shared memory with a column of padding (no bank conflicts).
    private const string Transpose16Ptx = """
        .version 7.0
        .target sm_80
        .address_size 64

        .visible .entry transpose16_rows(
            .param .u64 p_src, .param .u64 p_dst, .param .u32 p_count, .param .u32 p_columns, .param .u32 p_stride, .param .u32 p_first
        )
        .maxntid 256, 1, 1
        {
            .shared .align 2 .b16 tile[1056];
            .reg .pred %p<4>;
            .reg .b16 %h<2>;
            .reg .b32 %r<24>;
            .reg .b64 %rd<8>;
            ld.param.u64 %rd1, [p_src];
            ld.param.u64 %rd2, [p_dst];
            ld.param.u32 %r1, [p_count];
            ld.param.u32 %r2, [p_columns];
            ld.param.u32 %r3, [p_stride];
            ld.param.u32 %r4, [p_first];
            mov.u32 %r5, %tid.x;
            mov.u32 %r6, %tid.y;
            mov.u32 %r7, %ctaid.x;
            mov.u32 %r8, %ctaid.y;
            shl.b32 %r9, %r7, 5;
            shl.b32 %r10, %r8, 5;
            mov.u32 %r20, tile;
            mov.u32 %r11, 0;
        LOAD:
            setp.ge.u32 %p0, %r11, 32;
            @%p0 bra STORE;
            add.u32 %r12, %r6, %r11;
            add.u32 %r13, %r10, %r12;
            add.u32 %r14, %r9, %r5;
            setp.lt.u32 %p1, %r13, %r1;
            setp.lt.and.u32 %p1, %r14, %r2, %p1;
            mad.lo.u32 %r15, %r13, %r2, %r14;
            mul.wide.u32 %rd3, %r15, 2;
            add.u64 %rd3, %rd3, %rd1;
            mov.b16 %h0, 0;
            @%p1 ld.global.u16 %h0, [%rd3];
            mad.lo.u32 %r16, %r12, 33, %r5;
            shl.b32 %r16, %r16, 1;
            add.u32 %r16, %r16, %r20;
            st.shared.u16 [%r16], %h0;
            add.u32 %r11, %r11, 8;
            bra LOAD;
        STORE:
            bar.sync 0;
            mov.u32 %r11, 0;
        WRITE:
            setp.ge.u32 %p0, %r11, 32;
            @%p0 bra DONE;
            add.u32 %r12, %r6, %r11;
            add.u32 %r13, %r9, %r12;
            add.u32 %r14, %r10, %r5;
            setp.lt.u32 %p2, %r13, %r2;
            setp.lt.and.u32 %p2, %r14, %r1, %p2;
            mad.lo.u32 %r16, %r5, 33, %r12;
            shl.b32 %r16, %r16, 1;
            add.u32 %r16, %r16, %r20;
            ld.shared.u16 %h1, [%r16];
            add.u32 %r17, %r4, %r14;
            mul.wide.u32 %rd4, %r13, %r3;
            cvt.u64.u32 %rd5, %r17;
            add.u64 %rd4, %rd4, %rd5;
            shl.b64 %rd4, %rd4, 1;
            add.u64 %rd4, %rd4, %rd2;
            @%p2 st.global.u16 [%rd4], %h1;
            add.u32 %r11, %r11, 8;
            bra WRITE;
        DONE:
            ret;
        }
        """;

    public override bool UploadsTransposed16 => Transpose16() is not null;

    public override bool UploadTransposed16(ReadOnlySpan<ushort> block, int count, int columns, Storage destination, int stride, int first)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (Transpose16() is not { } kernel)
        {
            return false;
        }

        if (count == 0 || columns == 0)
        {
            return true;
        }

        if (block.Length < (long)count * columns || (long)(columns - 1) * stride + first + count > (long)destination.Length * 2)
        {
            throw new ArgumentOutOfRangeException(nameof(block), $"{count} rows of {columns} 16-bit values into a storage of {destination.Length} words at a stride of {stride} from {first}.");
        }

        // The block up as words (its last odd value padded), then transposed into place.
        int values = count * columns, words = (values + 1) / 2;
        var staged = Allocate(words, zeroed: false);
        ushort[] padded = ArrayPool<ushort>.Shared.Rent(words * 2);
        try
        {
            block[..values].CopyTo(padded);
            if (values % 2 != 0)
            {
                padded[values] = 0;
            }
            Upload(MemoryMarshal.Cast<ushort, float>(padded.AsSpan(0, words * 2)), staged);
            Launch(kernel, (uint)((columns + 31) / 32), (uint)((count + 31) / 32), 32, 8, P(staged), P(destination), U(count), U(columns), U(stride), U(first));
            return true;
        }
        finally
        {
            ArrayPool<ushort>.Shared.Return(padded);
            staged.Release();
        }
    }

    // The transpose kernel, loaded on first use (null where the driver refuses it: TypedUnavailableReason says why).
    private IntPtr? Transpose16()
    {
        if (!Volatile.Read(ref _transpose16Tried))
        {
            LoadTranspose16();
        }

        return _transpose16;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void LoadTranspose16()
    {
        lock (_typedLock)
        {
            if (_transpose16Tried)
            {
                return;
            }

            try
            {
                _transpose16 = LoadDynamicModule(Transpose16Name, [Transpose16Name], Transpose16Ptx, static _ => 0, out string? reason) is { } kernels ? kernels[Transpose16Name] : null;
                if (_transpose16 is null)
                {
                    TypedUnavailableReason = reason;
                }
            }
            finally
            {
                Volatile.Write(ref _transpose16Tried, true);
            }
        }
    }
}
