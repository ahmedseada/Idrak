// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Gpu.Vulkan;

// Cooperative matrices (VK_KHR_cooperative_matrix), 16-bit floats in shaders (shaderFloat16: Vulkan 1.2, or
// VK_KHR_shader_float16_int8) and bfloat16 in shaders (VK_KHR_shader_bfloat16): the features a device is asked for, and
// the matrix shapes it reports.
internal static unsafe partial class VulkanDriver
{
    public const uint StructurePhysicalDeviceShaderFloat16Int8Features = 1000082000,
        StructurePhysicalDeviceCooperativeMatrixFeatures = 1000506000,
        StructureCooperativeMatrixProperties = 1000506001,
        StructurePhysicalDeviceCooperativeMatrixProperties = 1000506002,
        StructurePhysicalDeviceShaderBfloat16Features = 1000141000;

    // VkComponentTypeKHR and VkScopeKHR values the products use (VK_COMPONENT_TYPE_BFLOAT16_KHR from VK_KHR_shader_bfloat16).
    public const uint ComponentFloat16 = 0, ComponentFloat32 = 1, ComponentBFloat16 = 1000141000, ScopeSubgroup = 3;

    /// <summary>VK_KHR_cooperative_matrix: matrix operations a subgroup does together (on matrix units where the device has them).</summary>
    public const string CooperativeMatrixExtension = "VK_KHR_cooperative_matrix";

    /// <summary>VK_KHR_shader_float16_int8: 16-bit floats in shaders (core in Vulkan 1.2).</summary>
    public const string ShaderFloat16Int8Extension = "VK_KHR_shader_float16_int8";

    /// <summary>VK_KHR_shader_bfloat16: bfloat16 values in shaders, and as cooperative matrix components.</summary>
    public const string ShaderBFloat16Extension = "VK_KHR_shader_bfloat16";

    [LibraryImport(Library)]
    public static partial IntPtr vkGetInstanceProcAddr(IntPtr instance, byte* name);
}

/// <summary>VkPhysicalDeviceCooperativeMatrixFeaturesKHR.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceCooperativeMatrixFeatures
{
    public uint SType;
    public void* PNext;
    public uint CooperativeMatrix;
    public uint CooperativeMatrixRobustBufferAccess;
}

/// <summary>VkPhysicalDeviceCooperativeMatrixPropertiesKHR: the shader stages with cooperative matrices.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceCooperativeMatrixProperties
{
    public uint SType;
    public void* PNext;
    public uint CooperativeMatrixSupportedStages;
}

/// <summary>VkCooperativeMatrixPropertiesKHR: one matrix shape and its component types (A: M × K, B: K × N, C and the result: M × N).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkCooperativeMatrixProperties
{
    public uint SType;
    public void* PNext;
    public uint MSize;
    public uint NSize;
    public uint KSize;
    public uint AType;
    public uint BType;
    public uint CType;
    public uint ResultType;
    public uint SaturatingAccumulation;
    public uint Scope;
}

/// <summary>VkPhysicalDeviceShaderFloat16Int8Features (Vulkan 1.2, or VK_KHR_shader_float16_int8).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceShaderFloat16Int8Features
{
    public uint SType;
    public void* PNext;
    public uint ShaderFloat16;
    public uint ShaderInt8;
}

/// <summary>VkPhysicalDeviceShaderBfloat16FeaturesKHR (VK_KHR_shader_bfloat16).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceShaderBfloat16Features
{
    public uint SType;
    public void* PNext;
    public uint ShaderBFloat16Type;
    public uint ShaderBFloat16DotProduct;
    public uint ShaderBFloat16CooperativeMatrix;
}
