// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The library's benchmarks and stress tests, through its public API only, on every device IDRAK_DEVICES names (default:
// every GPU found, else the CPU). Each section prints a table and, with IDRAK_BENCH_OUT=file.json, writes its rows so two
// runs can be compared:
//
//   dotnet run -c Release --project benches/Idrak.Benches -- [section ...]
//   dotnet run -c Release --project benches/Idrak.Benches -- compare before.json after.json
//
// Sections (default: all):
//   ops            element-wise, reductions, softmax, layer norm: µs a call (IDRAK_BENCH_CALLS back to back, one sync),
//                  GB/s, launches; forward and backward
//   matmul         products and convolutions at several shapes: TFLOPS, ms a call
//   train-decoder  a decoder's training step (size by IDRAK_BENCH_SIZE): ms a step, launches a step, peak memory,
//                  activation memory with and without checkpointing
//   infer-decoder  a decoder decoding token by token: tokens a second, launches a token, the key/value cache's memory
//   train-speech   a speech synthesizer's adversarial training step (the plug-in's VITS; IDRAK_BENCH_MODEL names the
//                  model folder, default the tiny fixture under tests/; IDRAK_BENCH_CLIPS a folder of clips, default
//                  synthetic ones): ms a step, launches a step by operation, peak memory, at one and several clips
//   memory         stress: the pool over thousands of varied shapes (rule 77: device memory does not grow), long
//                  sequences, the largest batch the device takes before it runs out
//
// Settings: IDRAK_DEVICES (cpu, cuda:0, vulkan:0, ...), IDRAK_BENCH_SIZE (small, medium, large; default small on the
// CPU, medium on a GPU; a device that cannot hold a size's model or batch gets the next smaller batch, else the section
// is skipped with the reason), IDRAK_BENCH_REPEATS (timed rounds after the cold one, default 5), IDRAK_BENCH_WARMUP
// (untimed calls first, default 30), IDRAK_BENCH_CALLS (calls a timed round in the ops section, default 8: one device
// sync a round, so a driver's submission latency does not stand for a 10 µs kernel), IDRAK_BENCH_SECONDS (the most a
// stress section runs, default 60), IDRAK_BENCH_OUT, IDRAK_BENCH_MODEL, IDRAK_BENCH_CLIPS.

using Idrak.Benches;

if (args is ["compare", var before, var after])
{
    return Report.Compare(before, after);
}

if (args.Any(a => a is "-h" or "--help" or "help"))
{
    Console.WriteLine("sections: ops matmul train-decoder infer-decoder train-speech memory (default: all); compare before.json after.json");
    return 0;
}

var sections = args.Length == 0 ? Sections.All : args;
var unknown = sections.Where(s => !Sections.All.Contains(s, StringComparer.OrdinalIgnoreCase)).ToList();
if (unknown.Count > 0)
{
    Console.Error.WriteLine($"unknown section(s): {string.Join(", ", unknown)}; sections: {string.Join(" ", Sections.All)}");
    return 2;
}

var settings = Settings.FromEnvironment();
var report = new Report();
Console.WriteLine(settings.Describe());
foreach (string section in sections)
{
    Console.WriteLine();
    Console.WriteLine($"== {section}");
    foreach (var device in settings.Devices)
    {
        Console.WriteLine($"-- {device} ({device.Name})");
        try
        {
            Sections.Run(section, device, settings, report);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException or IOException or ResourceLimitExceededException)
        {
            Console.WriteLine($"   skipped: {e.Message}");
        }
    }
}

if (settings.Output is { } output)
{
    report.Write(output, settings);
    Console.WriteLine();
    Console.WriteLine($"{report.Rows.Count} rows written to {output} (compare with: -- compare before.json {output})");
}

return 0;
