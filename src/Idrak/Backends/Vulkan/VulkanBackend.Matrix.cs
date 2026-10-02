// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using static Idrak.Backends.Vulkan.VulkanDriver;

namespace Idrak.Backends.Vulkan;

// Matrix units (tensor cores, XMX, WMMA) through cooperative matrices (VK_KHR_cooperative_matrix). Where the device
// reports the extension with its cooperativeMatrix feature for compute shaders, 16-bit floats in shaders (shaderFloat16)
// and subgroups in compute, and among its matrix shapes one with 16-bit float operands, 32-bit float sums and subgroup
// scope that the kernels tile (VulkanKernels.CoopShapeUsable), those features are enabled when the device is created and
// the cooperative-matrix products (VulkanKernels.Coop.cs, as precise as the float32 kernels) join the candidates of the
// float32 product and of the prompt-sized packed products: measured on the device like every other candidate, used only
// where faster. Of several such shapes, the one with the most multiply-adds per operation (ties: the device's order). A
// kernel the driver rejects is dropped. Devices without cooperative matrices, and IDRAK_VULKAN_MATRIX=0, keep every
// product as it was.
internal sealed unsafe partial class VulkanBackend
{
    /// <summary>What a device offers of cooperative matrices, as the products use them.</summary>
    /// <param name="Shape">The matrix shape the kernels are built for.</param>
    /// <param name="Float16Extension">16-bit floats enabled through VK_KHR_shader_float16_int8 (else as core Vulkan 1.2).</param>
    /// <param name="Emulated">Tests: the subgroup size of an emulation on a device without cooperative matrices (0: the device's own).</param>
    internal readonly record struct MatrixUnitSupport(VulkanKernels.CoopShape Shape, bool Float16Extension, int Emulated);

    /// <summary>Tests: false makes backends created afterwards leave cooperative matrices off; IDRAK_VULKAN_MATRIX=0 sets it for a process.</summary>
    internal static bool? MatrixUnitsOverride { get; set; } =
        Environment.GetEnvironmentVariable("IDRAK_VULKAN_MATRIX") is "0" or "false" ? false : null;

    /// <summary>
    /// Tests only: backends created afterwards on a device without cooperative matrices but with shaderFloat16 run the
    /// cooperative-matrix products emulated in this shape (VulkanKernels.Coop.cs: everything but the matrix operations as
    /// on a device with them), so the kernels can be checked against the CPU there. Null: off.
    /// </summary>
    internal static VulkanKernels.CoopShape? EmulatedMatrixUnits { get; set; }

    /// <summary>Tests only: the cooperative-matrix products' workgroup width (a power of two from 16 to MaxWidth) instead of
    /// four default subgroups, so they run with fewer subgroups than blocks (several rounds) or more (some idle).</summary>
    internal static int? CoopWidthOverride { get; set; }

    private readonly MatrixUnitSupport? _matrix;

    // The products' kernels by format (index 0 float32, else the packed format + 1), null where unusable; guarded by itself.
    private readonly VulkanKernel?[] _coopKernels = new VulkanKernel?[4];
    private readonly bool[] _coopChecked = new bool[4];

    /// <summary>The cooperative-matrix shape the products use here, or null without cooperative matrices.</summary>
    internal VulkanKernels.CoopShape? MatrixShape => _matrix?.Shape;

    /// <summary>Whether the cooperative-matrix products are emulated (tests).</summary>
    internal bool MatrixUnitsEmulated => _matrix is { Emulated: > 0 };

    /// <summary>
    /// Invocations per workgroup of the cooperative-matrix products: four subgroups of the device's default size (one per
    /// subgroup block), within MaxWidth and the device's workgroup limits.
    /// </summary>
    internal int CoopWidth
    {
        get
        {
            if (CoopWidthOverride is int forced)
            {
                return forced;
            }

            int width = (int)Math.Clamp(System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, 4 * Limits.SubgroupSize)), 16, VulkanKernels.MaxWidth);
            while (width > 16 && (width > Limits.MaxInvocations || width > Limits.MaxSizeX))
            {
                width /= 2;
            }

            return width;
        }
    }

    // What the device reports, read before the device is created.
    private static MatrixUnitSupport? ProbeMatrixUnits(PhysicalDevice physical)
    {
        if (MatrixUnitsOverride == false || !physical.Facts.ComputeSubgroups(VulkanDeviceFacts.SubgroupBasic))
        {
            return null;
        }

        uint instance = MakeVersion(1, 1);
        if (Exports(nameof(vkEnumerateInstanceVersion)) && vkEnumerateInstanceVersion(out uint loader) == Success)
        {
            instance = Math.Min(loader & ~0xFFFu, MakeVersion(1, 3));                // as the instance was created
        }

        bool float16Core = Math.Min(physical.Properties.ApiVersion, instance) >= MakeVersion(1, 2);
        bool float16Extension = !float16Core && HasExtension(physical.Handle, ShaderFloat16Int8Extension);
        bool cooperative = HasExtension(physical.Handle, CooperativeMatrixExtension);
        if (!float16Core && !float16Extension || !cooperative && EmulatedMatrixUnits is null)
        {
            return null;
        }

        var float16 = new VkPhysicalDeviceShaderFloat16Int8Features { SType = StructurePhysicalDeviceShaderFloat16Int8Features };
        var matrixFeatures = new VkPhysicalDeviceCooperativeMatrixFeatures { SType = StructurePhysicalDeviceCooperativeMatrixFeatures, PNext = &float16 };
        var features = new VkPhysicalDeviceFeatures2 { SType = StructurePhysicalDeviceFeatures2, PNext = cooperative ? &matrixFeatures : &float16 };
        vkGetPhysicalDeviceFeatures2(physical.Handle, &features);
        if (float16.ShaderFloat16 == 0)
        {
            return null;
        }

        if (!cooperative || matrixFeatures.CooperativeMatrix == 0)
        {
            // Tests: an emulation where the subgroup size divides the shape (else nothing).
            int subgroup = (int)physical.Facts.SubgroupSize;
            return EmulatedMatrixUnits is { } shape && subgroup > 0 && shape.M * shape.N % subgroup == 0 && VulkanKernels.CoopShapeUsable(shape)
                ? new MatrixUnitSupport(shape, float16Extension, subgroup) : null;
        }

        var stages = new VkPhysicalDeviceCooperativeMatrixProperties { SType = StructurePhysicalDeviceCooperativeMatrixProperties };
        var properties = new VkPhysicalDeviceProperties2 { SType = StructurePhysicalDeviceProperties2, PNext = &stages };
        vkGetPhysicalDeviceProperties2(physical.Handle, &properties);
        if ((stages.CooperativeMatrixSupportedStages & ShaderStageCompute) == 0)
        {
            return null;
        }

        var reported = ReportedMatrixShapes(physical.Handle);
        return ChooseMatrixShape(reported) is { } chosen ? new MatrixUnitSupport(chosen, float16Extension, 0) : null;
    }

    // The matrix shapes the device lists (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR), none when it lists none.
    private static VkCooperativeMatrixProperties[] ReportedMatrixShapes(IntPtr physical)
    {
        delegate* unmanaged<IntPtr, uint*, VkCooperativeMatrixProperties*, int> list;
        fixed (byte* name = "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\0"u8)
        {
            list = (delegate* unmanaged<IntPtr, uint*, VkCooperativeMatrixProperties*, int>)vkGetInstanceProcAddr(s_instance, name);
        }

        uint count = 0;
        if (list == null || list(physical, &count, null) != Success || count == 0)
        {
            return [];
        }

        var shapes = new VkCooperativeMatrixProperties[count];
        for (int i = 0; i < shapes.Length; i++)
        {
            shapes[i].SType = StructureCooperativeMatrixProperties;
        }

        fixed (VkCooperativeMatrixProperties* p = shapes)
        {
            if (list(physical, &count, p) is not (Success or Incomplete))
            {
                return [];
            }
        }

        return shapes[..(int)Math.Min(count, (uint)shapes.Length)];
    }

    /// <summary>
    /// The shape the products use among those a device reports: 16-bit float A and B, 32-bit float C and result,
    /// subgroup scope, one the kernels tile; of those the most multiply-adds per operation (M · N · K), ties to the first
    /// reported. Null when none qualifies.
    /// </summary>
    internal static VulkanKernels.CoopShape? ChooseMatrixShape(IReadOnlyList<VkCooperativeMatrixProperties> reported)
    {
        VulkanKernels.CoopShape? best = null;
        foreach (var p in reported)
        {
            var shape = new VulkanKernels.CoopShape((int)Math.Min(p.MSize, 1024), (int)Math.Min(p.NSize, 1024), (int)Math.Min(p.KSize, 1024));
            if (p.AType == ComponentFloat16 && p.BType == ComponentFloat16 && p.CType == ComponentFloat32 && p.ResultType == ComponentFloat32
                && p.Scope == ScopeSubgroup && VulkanKernels.CoopShapeUsable(shape)
                && (best is not { } b || shape.M * shape.N * shape.K > b.M * b.N * b.K))
            {
                best = shape;
            }
        }

        return best;
    }

    // The cooperative-matrix product of `format` (null: float32) for this device, built and its pipeline created on first
    // use (at the deepest staging step that fits); null without cooperative matrices, where its workgroup memory or
    // width exceeds the device's, or where the driver rejects it (it is then never a candidate).
    private VulkanKernel? CoopKernel(VulkanKernels.PackedFormat? format)
    {
        if (_matrix is not { } support)
        {
            return null;
        }

        int index = format is { } f ? (int)f + 1 : 0;
        lock (_coopKernels)
        {
            if (_coopChecked[index])
            {
                return _coopKernels[index];
            }

            _coopChecked[index] = true;
            int width = CoopWidth;
            if (width % Math.Max(1, support.Emulated) != 0)
            {
                return null;
            }

            // The deepest staging step whose workgroup memory the device has.
            SpirvKernel? built = null;
            foreach (int depth in VulkanKernels.CoopDepths)
            {
                if (depth >= support.Shape.K && VulkanKernels.Coop(new VulkanKernels.CoopSpec(format, support.Shape, width, support.Emulated, depth)) is { } candidate
                    && candidate.SharedBytes <= Limits.SharedBytes)
                {
                    built = candidate;
                    break;
                }
            }

            if (built is null || width > Limits.MaxInvocations || width > Limits.MaxSizeX)
            {
                return null;
            }

            // Its subgroups take the four blocks in turns, whatever their size: the pipeline keeps the device's default size.
            var kernel = new VulkanKernel(built.Words, built.Bindings, built.PushBytes, built.Name, built.Writes) { DefaultSubgroupSize = true };
            try
            {
                _ = PipelineOf(kernel);
            }
            catch (VulkanException)
            {
                return null;                                                   // the driver rejects it: never a candidate
            }

            _coopKernels[index] = kernel;
            return kernel;
        }
    }

    /// <summary>The cooperative matrices in one line, for benchmark headers.</summary>
    internal string DescribeMatrixUnits() => _matrix switch
    {
        null => "no cooperative matrices",
        { Emulated: > 0 } m => $"cooperative matrices emulated (tests), {m.Shape}",
        { } m => $"cooperative matrices {m.Shape} (float16 operands, float32 sums), workgroups of {CoopWidth}",
    };

    // The extensions and features to enable for cooperative matrices: appended to the device's extension names and
    // feature chain (the structures must outlive vkCreateDevice: the caller owns them).
    private void* MatrixFeatures(VkPhysicalDeviceCooperativeMatrixFeatures* matrix, VkPhysicalDeviceShaderFloat16Int8Features* float16, void* next)
    {
        if (_matrix is not { } support)
        {
            return next;
        }

        *float16 = new VkPhysicalDeviceShaderFloat16Int8Features { SType = StructurePhysicalDeviceShaderFloat16Int8Features, PNext = next, ShaderFloat16 = 1 };
        if (support.Emulated > 0)
        {
            return float16;
        }

        *matrix = new VkPhysicalDeviceCooperativeMatrixFeatures { SType = StructurePhysicalDeviceCooperativeMatrixFeatures, PNext = float16, CooperativeMatrix = 1 };
        return matrix;
    }

    private static readonly byte[] CooperativeMatrixName = Encoding.ASCII.GetBytes(CooperativeMatrixExtension + "\0");
    private static readonly byte[] ShaderFloat16Int8Name = Encoding.ASCII.GetBytes(ShaderFloat16Int8Extension + "\0");
}
