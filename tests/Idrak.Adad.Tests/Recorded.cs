// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Adad.Tests;

// What the source (Idrak's build at de1f8a96, the parts Adad copied) gives, recorded by the reference harness
// (test-output\adad\piece1-q4k\reference\; its outputs reference-ptx.txt, reference-types.txt,
// reference-control-first-shape.txt, reference-control-measured.txt and its log, copied to plans\opened\40-piece1-q4k\).
internal static class Recorded
{
    /// <summary>q4_k's encoder over the seeded rows (the bytes the encoders gave as first written; the source's test records the same).</summary>
    public const string EncodeSeeded = "d416333347a37c3868f90949f929b55aa5fbea706cfa92bb7b698bcc12a7c190";

    /// <summary>The source's q4_k decoder (SuperBlock.Decode) over those blocks: the floats' bits.</summary>
    public const string DecodeSeeded = "087db33f3e105839b71b8af6642f6719aa798667daad7028db70f3df3c2d4963";

    /// <summary>The source's decoder over random valid blocks (64 rows × 2048, seed 5).</summary>
    public const string DecodeRandom = "d30c60952f1ea0596962a9f0df01b40aaadee7a43dc9253a6b34b275246f6799";

    /// <summary>The source's q4_k kernels' name part (super_widen_{it}_bf16).</summary>
    public const string SuperCode = "k256x144_ccb20da7";

    /// <summary>The native kernel's 9 shapes: entry and the SHA-256 of its module's text (line ends as LF).</summary>
    public static readonly (string Entry, string Sha)[] Native =
    [
        ("native_mma_bf16_r64_s3_n128", "69cbe287017945268dbd83c39ad722cadb922c27b4ebaea9fd11b08bfbf70e7a"),
        ("native_mma_bf16_r64_s4_n128", "9c35e5db70c3709f314c76b3abe066c844e08f0200246e0cd3c78c4c9cfe2f3a"),
        ("native_mma_bf16_r128_s2_n128", "e3fa7ccc97d02ef3e2a2a53022c2a5748892fa1f2438a125cab9ffea109d1d7f"),
        ("native_mma_bf16_r64_s3_n256", "7e1f052915d60904098636e7ecbf58d61c1ac63b8ad89661ee06168696aab638"),
        ("native_mma_bf16_r64_s3_n256_x", "d7c76c021facc297c870cc42a719dc1c1d6213c9a13ce6ab8b434d5937926c39"),
        ("native_mma_bf16_r64_s4_n256_x", "8ec69d32d06dc80043145c67c55ec3cf6c0eb7ebb36716a7208c3d6ab4dbc4d6"),
        ("native_mma_bf16_r128_s2_n256_x", "62b328e7b54a1109bc2e58b8df1a7339885526156aae73de9bfc6d45b1562e46"),
        ("native_mma_bf16_r128_s3_n128_x", "df907d9ad805700ca2894915b7569121b43d87c051ebed79dcc78a4be4dad80c"),
        ("native_mma_bf16_r64_s6_n128_x", "1de10af1eb586c9d80d2b6e83264f54c01cbc2477fb31598b042f67f3a6bed45"),
    ];

    /// <summary>The x pack's module text (the native lane's packing module up to its W kernel).</summary>
    public const string PackX = "f3eb7c39dde8d5809afb8d5c5b3f0441448c295cba08d3cdd71b5c4b7b3d1db0";

    /// <summary>The q4_k widening's module text: the native header, the pieces' table, the bfloat16 widening.</summary>
    public const string Widen = "aa302318bc8321f58f5f53ec42b328390fb6126ca3ca4edbcde7e64cd3dc0591";

    /// <summary>The splits' sum module text.</summary>
    public const string Sum = "53cbb6af75bc5d067b9f6e335acf108e0ab40960fda025acd9729ae1f88000f1";

    /// <summary>The source's control lane at each test shape, its native kernel's first shape unsplit (IDRAK_AUTOTUNE=0): y's bits.</summary>
    public static readonly (int M, int N, int K, string Sha)[] ControlFirstShape =
    [
        (128, 1024, 2048, "b8193ddc3222bc5d9a387c6f1ed669a1ff96961d3a2066b5425cc6c330d5594e"),
        (512, 2048, 2048, "744dc22283eb8acf6bf8eea42bcfc547de5a6b53c005937b3762a4d3c14704a7"),
        (512, 6144, 2048, "c5ae9a1b782ba0becbee91b8fd0d564e19325e63e1bde2831fa57a3bb6314155"),
        (512, 2048, 6144, "3bff4f7b367f31b7894a6a669c15e997d6f96f896794d1bfa0a3c9a3ab30ff75"),
        (128, 3072, 3072, "388d29f02d49c73ae567ea244b2edd6f3f2e590f22260bef6ccb04644f0e72fd"),
        (512, 1024, 3072, "53e30c4220a5e994383035d67d720c2fde40f6c0b57c9ee4b9527e6afdafa3ec"),
        (128, 3072, 8192, "bfefa45738937fd2342019b5993b79a68d898ea2e5b906342aed055fcf332b03"),
        (200, 300, 768, "06fe8c523775cbf6dc65911ae96170986ce69ed358439352ad156529504dd6f5"),
        (37, 70, 256, "93d649f8463c0fc1c6746ca10e6c84d8258c27fb6db89ec46ee2f8c2ba91eb89"),
    ];

    /// <summary>
    /// The source's control lane at each test shape at the shape and split it measured (a fresh tuning folder; the log's
    /// ProductRoute 9100 lines: shape · 16 + splits, the shape the index of NativeMmaConfigs, all 9 accepted): y's bits.
    /// </summary>
    public static readonly (int M, int N, int K, int Shape, int Splits, string Sha)[] ControlMeasured =
    [
        (128, 1024, 2048, 0, 8, "4f6f28ff72ca21ddee1382fdffc6d8a0871b8bdbe9a87e6378f479cd656efb49"),
        (512, 2048, 2048, 8, 1, "744dc22283eb8acf6bf8eea42bcfc547de5a6b53c005937b3762a4d3c14704a7"),
        (512, 6144, 2048, 8, 1, "c5ae9a1b782ba0becbee91b8fd0d564e19325e63e1bde2831fa57a3bb6314155"),
        (512, 2048, 6144, 8, 1, "3bff4f7b367f31b7894a6a669c15e997d6f96f896794d1bfa0a3c9a3ab30ff75"),
        (128, 3072, 3072, 8, 2, "26c1561d373c774da7e814c729f93550cffed4f621f70e4a117c02648ac97a18"),
        (512, 1024, 3072, 8, 2, "288acd22648426013ad238def0dbe0854a5ad14559519e0042c24e40ec3329c2"),
        (128, 3072, 8192, 7, 8, "3a3fb771fb1e505c024fab4e7d82adea7015d30c92165df37b3d7b20270724ce"),
        (200, 300, 768, 0, 8, "41a2ae6f1f58139bb5cd3d513c83dc6ff6e99548f5e9138397cd1c65b5c1064f"),
        (37, 70, 256, 7, 1, "93d649f8463c0fc1c6746ca10e6c84d8258c27fb6db89ec46ee2f8c2ba91eb89"),
    ];
}
