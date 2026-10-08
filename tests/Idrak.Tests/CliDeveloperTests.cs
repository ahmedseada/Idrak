// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Cli;
using Idrak.Abstraction.Operations;
using Idrak.Cli.Commands.Developer;
using Idrak.Inference;
using Idrak.Layers;

// The idrak tool's developer commands (plans/idrak-cli.md, "Developers"): new, test, onnx import/export/check, kernels
// dump, trace, demo and shell, run in-process through CommandLine.Run with captured output.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliDeveloperGroup =
    [
        ("cli dev: every developer command has help with examples and environment, and no shared short forms", CliDevHelp),
        ("cli dev: new writes each template with its placeholders filled; the plug-in project builds and its tests pass (on the CPU)", CliDevNew),
        ("cli dev: onnx export of a small network round-trips through onnx import; onnx check matches the round trip, ONNX Runtime and reference outputs", CliDevOnnx),
        ("cli dev: kernels lists the kernel each operation runs on a device (registered, device, composed, host, none), as text and JSON, filtered by --source", CliDevKernelChain),
        ("cli dev: kernels dump writes non-empty PTX, SPIR-V and HIP files", CliDevKernels),
        ("cli dev: test finds the source checkout and fails clearly outside one", CliDevTest),
        ("cli dev: trace runs a command with telemetry to JSON Lines, live, and wrapped in one JSON document", CliDevTrace),
        ("cli dev: demo xor reaches its result on the device; spirals, shapes and gpt on the CPU", CliDevDemo),
        ("cli dev: shell runs typed commands with quotes, history and !N, and completes commands and options", CliDevShell),
    ];

    private static (int Code, string Output, string Error) DevRun(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = CommandLine.Run(args, output, error);
        return (code, output.ToString(), error.ToString());
    }

    private static JsonNode DevJson(params string[] args)
    {
        var (code, output, error) = DevRun(args);
        Check(code == 0, $"idrak {string.Join(' ', args)}: exit {code}: {error}");
        return JsonNode.Parse(output) ?? throw new Exception($"idrak {string.Join(' ', args)}: no JSON");
    }

    private static string DevTemp(string name)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-cli-dev-{name}-{Environment.ProcessId}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static void CliDevHelp(Device device)
    {
        _ = device;
        foreach (string name in new[] { "new", "test", "onnx import", "onnx export", "onnx check", "vlm check", "kernels", "kernels dump", "trace", "demo", "shell" })
        {
            var (code, output, _) = DevRun([.. name.Split(' '), "--help"]);
            Check(code == 0 && output.Contains("Examples:", StringComparison.Ordinal) && output.Contains("Environment (idrak help env for all):", StringComparison.Ordinal),
                $"help of {name}: exit {code}\n{output}");
            var command = CommandLine.Find(CommandTable.All, name.Split(' '), out _)!;
            Check(command.Name == name, $"{name} is found");
            Check(command.ShortForms.Keys.All(k => !CommandContext.CommonShortForms.ContainsKey(k)), $"{name} reuses a common short form");
            Check(command.ShortForms.Values.All(v => command.ValueOptions.Contains(v) || command.Flags.Contains(v)), $"{name}: every short form names an option");
        }

        Check(DevRun("new").Code == 2 && DevRun("new", "desktop", "X").Code == 2 && DevRun("new", "console", "9bad").Code == 2, "new: usage errors exit with 2");
        Check(DevRun("demo", "chess").Code == 2 && DevRun("kernels", "dump", "cubin").Code == 2, "demo, kernels dump: unknown kinds exit with 2");
    }

    private static void CliDevNew(Device device)
    {
        string folder = DevTemp("new");
        string root = TestCommand.FindRepositoryRoot(AppContext.BaseDirectory) ?? throw new Exception("the tests run from a source checkout");
        try
        {
            foreach (string kind in NewCommand.Kinds.Keys)
            {
                string target = Path.Combine(folder, kind);
                var json = DevJson("new", kind, "My-" + kind, "-o", target, "-j");
                var files = json["files"]!.AsArray().Select(f => (string)f!).ToList();
                Check(files.Count >= 2 && files.All(f => File.Exists(Path.Combine(target, f))), $"{kind}: files {string.Join(", ", files)}");
                Check(files.Any(f => f.EndsWith(".csproj", StringComparison.Ordinal)) && files.Any(f => f.EndsWith("Program.cs", StringComparison.Ordinal) || f.EndsWith("Plugin.cs", StringComparison.Ordinal)),
                    $"{kind}: a project and its code");
                foreach (string file in files)
                {
                    string text = File.ReadAllText(Path.Combine(target, file));
                    Check(!file.Contains("__", StringComparison.Ordinal) && !text.Contains("__NAME__", StringComparison.Ordinal) && !text.Contains("__REF(", StringComparison.Ordinal)
                          && !text.Contains("__NAMESPACE__", StringComparison.Ordinal) && !text.Contains("__FORMAT__", StringComparison.Ordinal), $"{kind}/{file}: a placeholder is left");
                }

                string project = File.ReadAllText(Path.Combine(target, files.First(f => f.EndsWith(".csproj", StringComparison.Ordinal) && !f.Contains("Tests", StringComparison.Ordinal))));
                Check(project.Contains($"<PackageReference Include=\"Idrak\" Version=\"{NewCommand.PackageVersion()}\" />", StringComparison.Ordinal), $"{kind}: the Idrak package\n{project}");
                Check(kind != "webapi" || project.Contains("Include=\"Idrak.AspNetCore\"", StringComparison.Ordinal), "webapi: the ASP.NET Core package");
                Check(kind != "rag" || project.Contains("Include=\"Idrak.Nlp\"", StringComparison.Ordinal), "rag: the language package");
            }

            Check(File.ReadAllText(Path.Combine(folder, "plugin", "src", "My-plugin", "RoundedWeight.cs")).Contains("FormatName = \"my-plugin\"", StringComparison.Ordinal)
                  && File.ReadAllText(Path.Combine(folder, "plugin", "src", "My-plugin", "Plugin.cs")).Contains("namespace My_plugin;", StringComparison.Ordinal), "the format name and namespace");
            var (code, _, error) = DevRun("new", "console", "Again", "-o", Path.Combine(folder, "console"));
            Check(code == 1 && error.Contains("--force", StringComparison.Ordinal), $"a folder that is not empty: {code} {error}");
            Check(DevRun("new", "console", "Again", "-o", Path.Combine(folder, "console"), "-f", "-q").Code == 0, "--force writes into it");

            // From the source checkout: project references, and the plug-in builds against the libraries already built
            // for this run and passes its own tests (once, on the CPU: it does not depend on the device).
            string plugin = Path.Combine(folder, "source");
            DevJson("new", "plugin", "Outside", "-o", plugin, "--source", root, "-j");
            string tests = Path.Combine(plugin, "tests", "Outside.Tests", "Outside.Tests.csproj");
            Check(File.ReadAllText(tests).Contains("src/Idrak/Idrak.csproj".Replace('/', Path.DirectorySeparatorChar), StringComparison.Ordinal), "a project reference into the checkout");
            Check(DevRun("new", "plugin", "Outside2", "-o", Path.Combine(folder, "bad"), "--source", folder).Code == 1, "--source that is not a checkout");
            if (device.IsGpu)
            {
                return;
            }

            string configuration = AppContext.BaseDirectory.Contains($"{Path.DirectorySeparatorChar}Debug{Path.DirectorySeparatorChar}", StringComparison.Ordinal) ? "Debug" : "Release";
            // Each project on its own, without rebuilding the libraries (the run already uses their build).
            foreach (string project in new[] { Path.Combine(plugin, "src", "Outside", "Outside.csproj"), tests })
            {
                var (buildCode, buildOutput) = DevProcess(plugin, "dotnet", "build", project, "-c", configuration, "-m:1", "-p:BuildProjectReferences=false", "-nologo", "-v:q");
                Check(buildCode == 0, $"the generated {Path.GetFileName(project)} builds:\n{buildOutput}");
            }

            string dll = Path.Combine(plugin, "tests", "Outside.Tests", "bin", configuration, "net10.0", "Outside.Tests.dll");
            var (runCode, runOutput) = DevProcess(plugin, "dotnet", dll, "cpu");
            Check(runCode == 0 && runOutput.Contains("All passed", StringComparison.Ordinal), $"the generated plug-in's tests:\n{runOutput}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static (int Code, string Output) DevProcess(string folder, string file, params string[] args)
    {
        var info = new ProcessStartInfo(file) { WorkingDirectory = folder, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (string arg in args)
        {
            info.ArgumentList.Add(arg);
        }

        info.Environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1";
        info.Environment["DOTNET_NOLOGO"] = "1";
        using var process = Process.Start(info)!;
        var error = process.StandardError.ReadToEndAsync();
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, output + error.Result);
    }

    // A trained-looking 4 → 8 → 3 network saved as a package, and its outputs for a batch.
    private static (string Package, float[] Inputs, float[] Outputs) DevPackage(string folder, Device device)
    {
        var builder = Network.Input(4).Seed(11).Linear(8).Tanh().Linear(3);
        using var model = builder.OnDevice(device).Build();
        string package = Path.Combine(folder, "small.ikm");
        ModelPackage.Create(package).Architecture(builder).Weights(model).Save();
        var inputs = Enumerable.Range(0, 5 * 4).Select(i => MathF.Sin(i * 0.7f)).ToArray();
        using var scope = new TensorScope();
        return (package, inputs, model.Predict(Tensor.From(inputs, [5, 4], device)).ToArray());
    }

    private static void CliDevOnnx(Device device)
    {
        string folder = DevTemp("onnx");
        try
        {
            var (package, inputs, outputs) = DevPackage(folder, device);
            string onnx = Path.Combine(folder, "small.onnx");
            var exported = DevJson("onnx", "export", package, "-o", onnx, "-d", device.ToString(), "-j");
            Check(new FileInfo(onnx).Length > 100 && (string)exported["weights"]! == "trained" && exported["inputShape"]!.AsArray().Count == 1, $"export: {exported}");
            Check(DevRun("onnx", "export", package, "-o", onnx).Code == 1 && DevRun("onnx", "export", package, "-o", onnx, "-f", "-q").Code == 0, "export: --force to overwrite");

            string back = Path.Combine(folder, "back.ikm");
            var imported = DevJson("onnx", "import", onnx, "-o", back, "-d", device.ToString(), "-j");
            Check((string)imported["kind"]! == "chain" && (long)imported["parameters"]! == 4 * 8 + 8 + 8 * 3 + 3, $"import: {imported}");
            using (var predictor = Predictor.Load(back, device).Build())
            {
                var again = Enumerable.Range(0, 5).SelectMany(i => predictor.Predict(inputs[(i * 4)..((i + 1) * 4)])).ToArray();
                AssertClose(outputs, again, 1e-4f, "the network after export and import");
            }

            // check: the round trip, ONNX Runtime (this runner references Idrak.Onnx.Runtime) and a reference file.
            var reference = new JsonObject
            {
                ["inputs"] = new JsonArray([.. Enumerable.Range(0, 5).Select(i => (JsonNode)new JsonArray([.. inputs[(i * 4)..((i + 1) * 4)].Select(v => (JsonNode)v)]))]),
                ["outputs"] = new JsonArray([.. Enumerable.Range(0, 5).Select(i => (JsonNode)new JsonArray([.. outputs[(i * 3)..((i + 1) * 3)].Select(v => (JsonNode)v)]))]),
            };
            File.WriteAllText(onnx + ".expected.json", reference.ToJsonString());
            var check = DevJson("onnx", "check", onnx, "-d", device.ToString(), "--tolerance", "1e-3", "-j");
            var statuses = check["checks"]!.AsArray().ToDictionary(c => (string)c!["name"]!, c => (string)c!["status"]!);
            Check((bool)check["ok"]! && statuses["Idrak export, imported again"] == "match" && statuses["ONNX Runtime"] == "match"
                  && statuses.Any(s => s.Key.StartsWith("reference outputs", StringComparison.Ordinal) && s.Value == "match"), $"check: {check}");
            Check(OnnxCheckCommand.RuntimeLoader() is not null, "ONNX Runtime is found when its assembly is present");

            // A wrong reference fails the check with exit code 1.
            reference["outputs"]![0]![0] = 100f;
            File.WriteAllText(onnx + ".expected.json", reference.ToJsonString());
            var (code, text, _) = DevRun("onnx", "check", onnx, "-d", device.ToString());
            Check(code == 1 && text.Contains("MISMATCH", StringComparison.Ordinal), $"a wrong reference: {code}\n{text}");

            // The network builder's JSON exports with initialized weights.
            string network = Path.Combine(folder, "network.json");
            File.WriteAllText(network, Network.Input(3).Linear(5).ReLU().Linear(2).ToJson().ToJsonString());
            var fromJson = DevJson("onnx", "export", network, "-j");
            Check((string)fromJson["weights"]! == "initialized" && File.Exists(Path.Combine(folder, "network.onnx")), $"export of a network JSON: {fromJson}");
            Check(DevRun("onnx", "check", Path.Combine(folder, "missing.onnx")).Code == 1 && DevRun("onnx", "import", onnx, "-o", Path.Combine(folder, "x.ikm"), "--shape", "a,b").Code == 2, "a missing file; a bad --shape");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliDevKernelChain(Device device)
    {
        string name = device.ToString();
        var json = DevJson("kernels", "-d", name, "-j");
        var operations = json["operations"]!.AsArray().Select(o => (Name: (string)o!["name"]!, Kernel: (string)o!["kernel"]!, Fallback: (string)o!["fallback"]!, Plugin: (bool)o!["plugin"]!)).ToList();
        var counts = json["counts"]!.AsObject();
        int all = Ops.All.Count + PluginOperations.All.Count;
        Check((string?)json["device"] == name && (string?)json["kind"] == device.Backend.Kind && operations.Count == all
              && counts.Sum(c => (int)c.Value!) == all, $"every operation once: {operations.Count} of {all}");
        var chain = Kernels.Chain(device.Backend);
        Check(operations.Select((o, i) => o.Name == chain[i].Operation.Name && o.Kernel == chain[i].Source.ToString().ToLowerInvariant()
                                         && o.Fallback == chain[i].Operation.Fallback.ToString().ToLowerInvariant() && o.Plugin == chain[i].Operation.IsPlugin).All(x => x), "the dispatcher's chain");
        Check(operations.Single(o => o.Name == "MatMulMany").Fallback == "composed" && operations.Single(o => o.Name == "Softmax").Fallback == "host"
              && operations.Single(o => o.Name == "GemmStrided").Fallback == "none", "fallbacks");
        if (device.Type == DeviceType.Cpu)
        {
            Check(operations.Single(o => o.Name == "Fill").Kernel == "device" && (int)counts["host"]! == 0, "the CPU runs its own kernels");
        }

        // A kernel registered for the device's kind shows; --source keeps only the operations of one kind.
        OperationKernels.Softmax softmax = (b, x, y, rows, cols, log) => b.SoftmaxKernel(x, y, rows, cols, log);
        using (Kernels.Register(Ops.Softmax, device.Backend.Kind, softmax))
        {
            var registered = DevJson("kernels", "-d", name, "--source", "registered", "-j")["operations"]!.AsArray();
            Check(registered.Count == 1 && (string?)registered[0]!["name"] == "Softmax" && (string?)registered[0]!["kernel"] == "registered", $"registered: {registered.ToJsonString()}");
            var (code, output, _) = DevRun("kernels", "-d", name);
            Check(code == 0 && output.StartsWith($"Kernels on {name} (", StringComparison.Ordinal) && output.Contains("1 registered", StringComparison.Ordinal)
                  && output.Split('\n').Any(l => l.StartsWith("Softmax ", StringComparison.Ordinal) && l.Contains("registered", StringComparison.Ordinal)), $"text listing:\n{output}");
        }

        Check(DevJson("kernels", "-d", name, "--source", "registered", "-j")["operations"]!.AsArray().Count == 0, "removed again");

        // The kernels of the outside plug-ins (tests/Idrak.PluginTests): "plug-in" rows, registered on the CPU and Vulkan.
        using (Idrak.PluginTests.PluginKernels.Install())
        {
            var rows = DevJson("kernels", "-d", name, "-j")["operations"]!.AsArray()
                .Where(o => (bool)o!["plugin"]!).ToDictionary(o => (string)o!["name"]!, o => (string)o!["kernel"]!);
            bool own = device.Backend.Kind is "cpu" or "vulkan";
            Check(rows.GetValueOrDefault("Outside.UnpackPairs") == (own ? "registered" : "host") && rows.GetValueOrDefault("Outside.ScaleRows") == (own ? "registered" : "composed")
                  && rows.GetValueOrDefault("Outside.Softplus") == (device.Type == DeviceType.Cpu ? "registered" : "composed"), $"plug-in rows: {string.Join(", ", rows)}");
            var (code, output, _) = DevRun("kernels", "-d", name, "--source", "registered");
            Check(code == 0 && output.Split('\n').Any(l => l.StartsWith("Outside.ScaleRows ", StringComparison.Ordinal) && l.TrimEnd().EndsWith("plug-in", StringComparison.Ordinal)) == own,
                $"text listing:\n{output}");
        }
        Check(DevRun("kernels", "-d", name, "--source", "gpu").Code == 2 && DevRun("kernels", "-d", name, "--source", "1").Code == 2
              && DevRun("kernels", "Softmax").Code == 2, "an unknown source or an argument exits with 2");
    }

    private static void CliDevKernels(Device device)
    {
        _ = device;
        string folder = DevTemp("kernels");
        try
        {
            var json = DevJson("kernels", "dump", "-o", folder, "-j");
            var files = json["files"]!.AsArray().Select(f => (Kind: (string)f!["kind"]!, Path: (string)f!["path"]!, Bytes: (long)f!["bytes"]!)).ToList();
            Check(files.All(f => File.Exists(f.Path) && new FileInfo(f.Path).Length == f.Bytes && f.Bytes > 0), "every file exists and is not empty");
            Check(files.Count(f => f.Kind == "ptx") == 2 && File.ReadAllText(Path.Combine(folder, "idrak.ptx")).Contains(".entry", StringComparison.Ordinal), "PTX with its entries");
            Check(files.Count(f => f.Path.EndsWith(".spv", StringComparison.Ordinal)) > 20 && File.Exists(Path.Combine(folder, "spirv", "kernels.txt")), "SPIR-V modules and their contracts");
            var spv = File.ReadAllBytes(files.First(f => f.Path.EndsWith(".spv", StringComparison.Ordinal)).Path);
            Check(BitConverter.ToUInt32(spv, 0) == 0x07230203, "a SPIR-V module starts with the magic number");
            Check(File.ReadAllText(Path.Combine(folder, "idrak_kernels.hip")).Contains("__global__", StringComparison.Ordinal)
                  && File.ReadAllText(Path.Combine(folder, "hip-kernels.txt")).Contains("add_f32(pointer, pointer, pointer, int)", StringComparison.Ordinal), "the HIP source and its parameters");
            string only = Path.Combine(folder, "only");
            var (code, output, _) = DevRun("kernels", "dump", "hip", "-o", only);
            Check(code == 0 && output.StartsWith("hip", StringComparison.Ordinal) && Directory.GetFiles(only).Length == 2, $"one kind: {output}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliDevTest(Device device)
    {
        _ = device;
        string folder = DevTemp("test");
        try
        {
            string? root = TestCommand.FindRepositoryRoot(AppContext.BaseDirectory);
            Check(root is not null && File.Exists(Path.Combine(root, "Idrak.slnx")), $"the checkout from {AppContext.BaseDirectory}");
            Check(TestCommand.FindRepositoryRoot(Path.Combine(root!, "src", "Idrak.Cli")) == root, "from a folder inside it");
            Check(TestCommand.FindRepositoryRoot(folder) is null, "no checkout around a temporary folder");
            TestCommand.StartFolder = folder;
            try
            {
                var (code, _, error) = DevRun("test", "--filter", "nothing");
                Check(code == 1 && error.Contains("not inside an Idrak source checkout", StringComparison.Ordinal) && error.Contains("git clone", StringComparison.Ordinal), $"outside a checkout: {code} {error}");
                Check(DevRun("test", "extra").Code == 2, "a positional argument is a usage error");
            }
            finally
            {
                TestCommand.StartFolder = null;
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliDevTrace(Device device)
    {
        string folder = DevTemp("trace");
        try
        {
            var (package, _, _) = DevPackage(folder, device);
            string onnx = Path.Combine(folder, "small.onnx");
            Check(DevRun("onnx", "export", package, "-o", onnx, "-q").Code == 0, "export");

            string log = Path.Combine(folder, "trace.jsonl");
            var (code, output, error) = DevRun("trace", "-o", log, "-d", device.ToString(), "--", "onnx", "check", onnx);
            Check(code == 0 && output.Contains("trace: onnx check exited with 0", StringComparison.Ordinal) && output.Contains("inference", StringComparison.Ordinal), $"trace to a file: {code}\n{output}\n{error}");
            var lines = File.ReadAllLines(log);
            Check(lines.Length > 0 && lines.All(l => JsonNode.Parse(l) is JsonObject), $"{lines.Length} JSON lines");

            (code, output, _) = DevRun("trace", "--levels", "layers,inference", "-d", device.ToString(), "--", "onnx", "check", onnx);
            Check(code == 0 && output.Contains("Linear", StringComparison.Ordinal) && output.Contains("Inference", StringComparison.Ordinal), $"printed live:\n{output}");

            var json = DevJson("trace", "-j", "-d", device.ToString(), "demo", "xor");
            Check((int)json["exitCode"]! == 0 && (string)json["result"]!["demo"]! == "xor" && (string)json["result"]!["device"]! == device.ToString(), $"one JSON document: {json}");
            Check(DevRun("trace").Code == 2 && DevRun("trace", "trace", "version").Code == 2 && DevRun("trace", "--levels", "kernels", "version").Code == 2
                  && DevRun("trace", "nosuch").Code == 2, "usage errors");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void CliDevDemo(Device device)
    {
        var xor = DevJson("demo", "xor", "-d", device.ToString(), "-j");
        Check((bool)xor["passed"]! && (double)xor["value"]! == 4 && (double)xor["samplesPerSecond"]! > 0 && (string)xor["device"]! == device.ToString(), $"xor: {xor}");
        if (device.IsGpu)
        {
            return;
        }

        foreach (string demo in new[] { "spirals", "shapes", "gpt" })
        {
            var json = DevJson("demo", demo, "-d", "cpu", "-j");
            Check((bool)json["passed"]! && (double)json["seconds"]! > 0, $"{demo}: {json}");
        }

        var (code, output, _) = DevRun("demo", "xor", "-d", "cpu");
        Check(code == 0 && output.Contains("Device: cpu", StringComparison.Ordinal) && output.Contains("samples/s", StringComparison.Ordinal), $"text:\n{output}");
    }

    private static void CliDevShell(Device device)
    {
        _ = device;
        Check(ShellCommand.Split("demo xor -d 'vulkan:0' \"a b\" c\\ d").SequenceEqual(["demo", "xor", "-d", "vulkan:0", "a b", "c d"]), "quotes and escapes");
        Check(ShellCommand.Complete("onnx ").SequenceEqual(["check", "export", "import"]), $"subcommands: {string.Join(",", ShellCommand.Complete("onnx "))}");
        Check(ShellCommand.Complete("kernels d").SequenceEqual(["dump"]) && ShellCommand.Complete("dem").Contains("demo"), "command names");
        Check(ShellCommand.Complete("onnx check --tol").SequenceEqual(["--tolerance"]) && ShellCommand.Complete("demo xor --j").SequenceEqual(["--json"]), "options");
        Check(ShellCommand.Complete("demo ").Count == 0, "nothing after a whole command");

        string folder = DevTemp("shell");
        try
        {
            string history = Path.Combine(folder, "history");
            ShellCommand.Input = new StringReader("version\n\nhistory\n!1\nshell\nnosuch\n'unclosed\nexit\nversion\n");
            try
            {
                var (code, output, error) = DevRun("shell", "--history", history);
                Check(code == 0, $"exit {code}: {error}");
                Check(output.Split('\n').Count(l => l.StartsWith("idrak> idrak ", StringComparison.Ordinal) || l.StartsWith("idrak ", StringComparison.Ordinal)) >= 2, $"version ran twice:\n{output}");
                Check(output.Contains("    1  version", StringComparison.Ordinal) && error.Contains("Already in the shell", StringComparison.Ordinal)
                      && error.Contains("Unknown command 'nosuch'", StringComparison.Ordinal) && error.Contains("Unclosed", StringComparison.Ordinal), $"history and errors:\n{output}\n{error}");
                Check(File.ReadAllLines(history).SequenceEqual(["version", "history", "version", "shell", "nosuch", "'unclosed"]), $"history file: {string.Join("|", File.ReadAllLines(history))}");
            }
            finally
            {
                ShellCommand.Input = null;
            }

            Check(DevRun("shell", "-j").Code == 2, "no JSON output");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
