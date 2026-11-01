// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins;
using Idrak.Cli;
using Idrak.Cli.Commands.Health;
using Idrak.Models.Abstractions;
using Idrak.Reinforcement.Plugins;
using Idrak.Text.Plugins;
using Idrak.Vision.Plugins;

// Plan 26, phase P1: the plug-in packages (Idrak.Text.Plugins, Idrak.Vision.Plugins, Idrak.Audio.Plugins, and plan 27's
// Idrak.Reinforcement.Plugins), the plug-in contract they register through (IIdrakPlugin, IdrakPlugin, IdrakPlugins), the
// tool registering them at startup and loading them with -P, their lack of internals, and the "not registered" messages
// naming them. Each family the later phases move in gets its own group (plugins: <family>).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] PluginPackageGroup =
    [
        ("plugins: packages: Idrak.Text.Plugins, Idrak.Vision.Plugins, Idrak.Audio.Plugins and Idrak.Reinforcement.Plugins load, each root class has Package (an IIdrakPluginPackage the assembly declares with IdrakPluginPackage, read once), All, RegisterAll and UnregisterAll and no RegisterIdrakPlugin, each family in a namespace of its own; RegisterAll twice leaves every registry the tool lists as once, and UnregisterAll leaves them as found", PluginPackagesRegisterAll),
        ("plugins: packages: an IdrakPlugin registers its parts once however often Register is called (from several threads too), Unregister leaves the registries as found, a failing part takes back the parts before it, Requires registers a needed plug-in only when missing; IdrakPlugins registers a plug-in once, replaces one of the same name (unregistering the earlier, which comes back when the new one fails) and names the packages when none is registered", PluginContract),
        ("plugins: packages: the tool registers the four packages at startup (text first, reinforcement last), and -P on each package's DLL loads it through the tool's loader (its IdrakPluginPackage attribute) adding nothing twice; -P on an assembly that declares no package is refused naming the attribute, and on a library assembly registers its built-ins", PluginPackagesCli),
        ("plugins: packages: IDRAK_PLUGIN_PACKAGES chooses the packages the tool registers at startup (all four when unset, none, or a list taken in the tool's order) and is refused naming the variable when it names no package", PluginPackagesSetting),
        ("plugins: packages: no library assembly nor the tool grants a plug-in package its internals, the plug-in packages grant none but the test runner (as every library package does), and they are not library defaults", PluginPackagesNoInternals),
        ("plugins: packages: LibraryDefaults loads only the libraries the program has (its files, read once, or loaded): one it lacks is skipped with no first-chance exception, and a host that has them all finds what loading every name finds", PluginPackagesLibrariesPresent),
        ("plugins: packages: each plug-in package declares the architectures it registers (IdrakPluginNames), read alike from its file's metadata and the loaded assembly; PluginPackageCatalogs' library catalog and an app's name the package, and the missing-architecture refusal names the package that declares the family asked for, or no package (never Vision for a text family)", PluginPackagesArchitectureMessage),
        ("plugins: packages: the not-registered messages of VisionFamilies, SpeechFamilies, SpeechSynthesisFamilies, VoiceConverters, GgufArchitectures, ToolCallFormats, EnvironmentKinds and RewardFunctions name the plug-in package to reference and its RegisterAll, each as a NotRegisteredException", PluginPackagesMessages),
    ];

    private sealed record PluginPackage(string Name, Type Root, IIdrakPluginPackage Package, IReadOnlyList<IIdrakPlugin> All, Action RegisterAll, Action UnregisterAll);

    private static readonly PluginPackage[] PluginPackages =
    [
        new(TextPlugins.Package.Name, typeof(TextPlugins), TextPlugins.Package, TextPlugins.All, TextPlugins.RegisterAll, TextPlugins.UnregisterAll),
        new(VisionPlugins.Package.Name, typeof(VisionPlugins), VisionPlugins.Package, VisionPlugins.All, VisionPlugins.RegisterAll, VisionPlugins.UnregisterAll),
        new(AudioPlugins.Package.Name, typeof(AudioPlugins), AudioPlugins.Package, AudioPlugins.All, AudioPlugins.RegisterAll, AudioPlugins.UnregisterAll),
        new(ReinforcementPlugins.Package.Name, typeof(ReinforcementPlugins), ReinforcementPlugins.Package, ReinforcementPlugins.All, ReinforcementPlugins.RegisterAll, ReinforcementPlugins.UnregisterAll),
    ];

    // What the tool lists (every registry by category) and the plug-ins recorded, as one comparable text.
    private static string RegistryState() =>
        string.Join("\n", Registries.Snapshot().Select(c => $"{c.Key}: {string.Join(", ", c.Names)}").Append($"IdrakPlugins: {string.Join(", ", IdrakPlugins.Names)}"));

    private static void PluginPackagesRegisterAll(Device device)
    {
        _ = device;
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var package in PluginPackages)
        {
            Check(package.Root.Assembly.GetName().Name == package.Name && package.Root.Namespace == package.Name, $"{package.Root.Name} is the root class of {package.Name}");
            var declared = IdrakPlugins.PackageOf(package.Root.Assembly);
            Check(declared is not null && declared.Name == package.Name && declared.All.SequenceEqual(package.All) && package.Package.All.SequenceEqual(package.All)
                  && ReferenceEquals(IdrakPlugins.PackageOf(package.Root.Assembly), declared),
                $"{package.Name}: the assembly declares its package (IdrakPluginPackage), read once, with the root class's name and plug-ins");
            Check(package.Root.GetProperty("Package", BindingFlags.Public | BindingFlags.Static)?.PropertyType == typeof(IIdrakPluginPackage),
                $"{package.Name}: the root class's Package is the IIdrakPluginPackage");
            Check(package.Root.Assembly.GetExportedTypes().All(t => t.GetMethod("RegisterIdrakPlugin", BindingFlags.Public | BindingFlags.Static, Type.EmptyTypes) is null),
                $"{package.Name}: no RegisterIdrakPlugin is left (-P reads the attribute)");
            foreach (var plugin in package.All)
            {
                string? ns = plugin.GetType().Namespace;
                Check(ns is not null && ns.StartsWith(package.Name + ".", StringComparison.Ordinal) && plugin.GetType().Assembly == package.Root.Assembly,
                    $"{package.Name}: the plug-in {plugin.Name} lives in a family namespace of the package ({ns})");
                Check(names.Add(plugin.Name), $"{package.Name}: the plug-in name {plugin.Name} is taken by another family");
            }
        }

        // Found registered (the tool registers the packages when a test runs it) or not; put back as found at the end.
        var registeredBefore = PluginPackages.SelectMany(p => p.All).Where(p => p.IsRegistered).ToList();
        try
        {
            foreach (var package in Enumerable.Reverse(PluginPackages))
            {
                package.UnregisterAll();
            }

            Check(PluginPackages.SelectMany(p => p.All).All(p => !p.IsRegistered && IdrakPlugins.Find(p.Name) is null), "UnregisterAll leaves no family registered");
            string found = RegistryState();
            foreach (var package in PluginPackages)
            {
                package.RegisterAll();
            }

            string once = RegistryState();
            Check(PluginPackages.SelectMany(p => p.All).All(p => p.IsRegistered && ReferenceEquals(IdrakPlugins.Find(p.Name), p)), "RegisterAll registers and records every family");
            foreach (var package in PluginPackages)
            {
                package.RegisterAll();
            }

            Check(RegistryState() == once, "RegisterAll twice leaves the registries as once");
            foreach (var package in Enumerable.Reverse(PluginPackages))
            {
                package.UnregisterAll();
            }

            Check(RegistryState() == found, $"UnregisterAll leaves the registries as RegisterAll found them:\n{found}\nnow:\n{RegistryState()}");
        }
        finally
        {
            foreach (var plugin in registeredBefore)
            {
                IdrakPlugins.Register(plugin);
            }
        }
    }

    // A plug-in for the contract tests: registers a token sampler under each name (the library's default one), counting
    // the parts it registers; Fail throws after them.
    private sealed class SamplerPlugin(string name, string[] samplers, bool fail = false, IIdrakPlugin? requires = null) : IdrakPlugin(name)
    {
        public int Parts;

        protected override void RegisterParts()
        {
            if (requires is not null)
            {
                Requires(requires);
            }

            foreach (string sampler in samplers)
            {
                Part(() =>
                {
                    Interlocked.Increment(ref Parts);
                    TokenSamplers.Register(TestKinds.Sampler(sampler, TokenSamplers.Default(TokenSamplers.DefaultName)!.Create));
                }, () => TokenSamplers.Unregister(sampler));
            }

            if (fail)
            {
                throw new InvalidOperationException("plugins test: a part failed");
            }
        }

        public void PartOutsideRegisterParts() => Part(() => { }, () => { });
    }

    private static void PluginContract(Device device)
    {
        _ = device;
        const string Name = "plugins-test";
        var before = TokenSamplers.Names.ToList();
        bool Samplers(params string[] added) => TokenSamplers.Names.SequenceEqual([.. before, .. added]);
        var first = new SamplerPlugin(Name, ["plugins-test-a", "plugins-test-b"]);
        var second = new SamplerPlugin(Name, ["plugins-test-c"]);
        var failing = new SamplerPlugin(Name, ["plugins-test-d"], fail: true);
        try
        {
            // IdrakPlugin alone.
            Check(!first.IsRegistered && first.Name == Name, "a new plug-in is not registered");
            first.Register();
            first.Register();
            Parallel.For(0, 8, _ => first.Register());
            Check(first.IsRegistered && first.Parts == 2 && Samplers("plugins-test-a", "plugins-test-b"), $"Register many times registers each part once: {first.Parts} parts");
            first.Unregister();
            first.Unregister();
            Check(!first.IsRegistered && Samplers(), "Unregister leaves the registries as found (twice too)");
            first.Register();
            Check(first.IsRegistered && first.Parts == 4 && Samplers("plugins-test-a", "plugins-test-b"), "Register after Unregister registers again");
            first.Unregister();
            Check(Samplers(), "and Unregister takes it back again");

            try
            {
                failing.Register();
                Check(false, "a failing part is reported");
            }
            catch (InvalidOperationException e) when (e.Message.Contains("a part failed", StringComparison.Ordinal))
            {
            }

            Check(!failing.IsRegistered && failing.Parts == 1 && Samplers(), "a failing registration takes back the parts before it");
            try
            {
                first.PartOutsideRegisterParts();
                Check(false, "Part outside RegisterParts is refused");
            }
            catch (InvalidOperationException e) when (e.Message.Contains("RegisterParts only", StringComparison.Ordinal))
            {
            }

            // IdrakPlugins.
            IdrakPlugins.Register(first);
            IdrakPlugins.Register(first);
            Check(IdrakPlugins.Names.Count(n => n == Name) == 1 && ReferenceEquals(IdrakPlugins.Get(Name), first) && first.Parts == 6 && Samplers("plugins-test-a", "plugins-test-b"),
                "IdrakPlugins.Register twice registers and records once");
            Check(IdrakPlugins.Origin(Name) == typeof(SamplerPlugin).Assembly.GetName().Name, $"the origin is the plug-in's assembly: {IdrakPlugins.Origin(Name)}");
            IdrakPlugins.Register(second);
            Check(!first.IsRegistered && second.IsRegistered && ReferenceEquals(IdrakPlugins.Find(Name), second) && Samplers("plugins-test-c"),
                "a plug-in of the same name replaces the earlier one, which is unregistered");
            Check(!IdrakPlugins.Unregister(first) && ReferenceEquals(IdrakPlugins.Find(Name), second), "unregistering the replaced plug-in leaves the one recorded");
            try
            {
                IdrakPlugins.Register(failing);
                Check(false, "a failing replacement is reported");
            }
            catch (InvalidOperationException)
            {
            }

            Check(second.IsRegistered && ReferenceEquals(IdrakPlugins.Find(Name), second) && Samplers("plugins-test-c"), "a failing replacement leaves the earlier plug-in registered");
            Check(IdrakPlugins.Unregister(Name) && !second.IsRegistered && IdrakPlugins.Find(Name) is null && Samplers(), "Unregister by name takes the plug-in back and forgets it");
            Check(!IdrakPlugins.Unregister(Name), "Unregister of a name not recorded is false");
            ThrowsNaming<NotSupportedException>(() => IdrakPlugins.Get(Name), "an unknown plug-in", $"'{Name}'", "IdrakPlugins.Register", "Idrak.Text.Plugins", "Idrak.Vision.Plugins", "Idrak.Audio.Plugins",
                "Idrak.Reinforcement.Plugins");

            // Requires: the needed plug-in is registered when missing and taken back with the one that needed it; one
            // registered already stays.
            var dependent = new SamplerPlugin("plugins-test-dependent", ["plugins-test-e"], requires: first);
            dependent.Register();
            Check(first.IsRegistered && ReferenceEquals(IdrakPlugins.Find(Name), first) && Samplers("plugins-test-a", "plugins-test-b", "plugins-test-e"), "Requires registers the missing plug-in");
            dependent.Unregister();
            Check(!first.IsRegistered && IdrakPlugins.Find(Name) is null && Samplers(), "and takes it back with the plug-in that needed it");
            IdrakPlugins.Register(first);
            dependent.Register();
            dependent.Unregister();
            Check(first.IsRegistered && ReferenceEquals(IdrakPlugins.Find(Name), first) && Samplers("plugins-test-a", "plugins-test-b"), "a needed plug-in registered already stays");
            IdrakPlugins.Unregister(first);
            Check(Samplers() && IdrakPlugins.Find(Name) is null, "the registries are left as found");
        }
        finally
        {
            IdrakPlugins.Unregister(Name);
            IdrakPlugins.Unregister("plugins-test-dependent");
            foreach (string sampler in (string[])["plugins-test-a", "plugins-test-b", "plugins-test-c", "plugins-test-d", "plugins-test-e"])
            {
                TokenSamplers.Unregister(sampler);
            }
        }
    }

    private static void PluginPackagesCli(Device device)
    {
        _ = device;
        Check(CommandLine.PluginPackages.Select(p => p.Package).SequenceEqual(PluginPackages.Select(p => p.Package)), "the tool registers Idrak.Text.Plugins, Idrak.Vision.Plugins, Idrak.Audio.Plugins and Idrak.Reinforcement.Plugins, in that order");
        var listed = CliJson("plugins", "list", "-j");
        Check(listed["plugins"] is JsonObject, $"plugins list has the plug-ins: {listed}");
        var recorded = listed["plugins"]!["names"]!.AsArray().Select(n => (string?)n).ToHashSet(StringComparer.Ordinal);
        Check(PluginPackages.SelectMany(p => p.All).All(p => p.IsRegistered && recorded.Contains(p.Name)), "every family of the four packages is registered by the tool, before any -P");
        Check(listed.AsObject().All(c => c.Value!["added"]!.AsArray().Count == 0), "the packages' families are the tool's own, not added by a plug-in");
        foreach (var package in PluginPackages)
        {
            string path = package.Root.Assembly.Location;
            var (exit, output, error) = Cli("plugins", "list", "-P", path, "--added", "-j");
            Check(exit == 0, $"-P {package.Name}.dll: exit {exit}: {error}");
            var added = JsonNode.Parse(output)!.AsObject();
            Check(added.All(c => c.Value!["added"]!.AsArray().Count == 0), $"-P {package.Name}.dll adds nothing the tool registered already: {output}");
            Check(PluginPackages.SelectMany(p => p.All).All(p => p.IsRegistered), $"-P {package.Name}.dll leaves every family registered");
        }

        // An assembly that declares no package: refused, naming the attribute (Idrak.AspNetCore declares neither a package
        // nor library registrations; Idrak.Mcp now declares library registrations).
        var (refusedExit, _, refused) = Cli("plugins", "list", "-P", typeof(Idrak.AspNetCore.ChatApiOptions).Assembly.Location);
        Check(refusedExit == 2 && refused.Contains("IdrakPluginPackage", StringComparison.Ordinal) && refused.Contains("IIdrakPluginPackage", StringComparison.Ordinal),
            $"-P on an assembly with no package: exit {refusedExit}: {refused}");
        Check(!IdrakPlugins.Load(typeof(Idrak.AspNetCore.ChatApiOptions).Assembly) && IdrakPlugins.PackageOf(typeof(Idrak.AspNetCore.ChatApiOptions).Assembly) is null,
            "IdrakPlugins.Load: nothing declared, false");

        // A library assembly (Idrak.Onnx.Runtime, shipped with this runner): -P registers its built-ins, once.
        var runtimes = OnnxRuntimeLoaders.Names.ToList();
        var (libraryExit, _, libraryError) = Cli("plugins", "list", "-P", typeof(Idrak.Onnx.Runtime.OnnxModule).Assembly.Location);
        Check(libraryExit == 0 && IdrakPlugins.Load(typeof(Idrak.Onnx.Runtime.OnnxModule).Assembly) && OnnxRuntimeLoaders.Names.SequenceEqual(runtimes)
              && runtimes.Contains("onnxruntime") && OnnxRuntimeLoaders.Origin("onnxruntime") == Overrides.Library,
            $"-P on a library assembly: exit {libraryExit} {libraryError}; runtimes {string.Join(", ", OnnxRuntimeLoaders.Names)}");
    }

    private static void PluginPackagesSetting(Device device)
    {
        _ = device;
        static string[] Chosen(string? value) => [.. CommandLine.ChosenPluginPackages(value).Select(p => p.Name)];
        Check(Chosen(null).SequenceEqual(["text", "vision", "audio", "reinforcement"]) && Chosen("").Length == 4 && Chosen(" All ").Length == 4, "all four when unset, empty or all");
        Check(Chosen("none").Length == 0 && Chosen(" NONE ").Length == 0, "none");
        Check(Chosen("audio, text").SequenceEqual(["text", "audio"]) && Chosen("Vision").SequenceEqual(["vision"]) && Chosen("reinforcement,text").SequenceEqual(["text", "reinforcement"]),
            "a list, in the tool's order (text first)");
        Check(CommandLine.ChosenPluginPackages("Reinforcement").Single().Package == ReinforcementPlugins.Package, "reinforcement is Idrak.Reinforcement.Plugins");
        Check(CommandLine.ChosenPluginPackages("vision").Single().Package == VisionPlugins.Package, "each name is its package");
        ThrowsNaming<UsageException>(() => CommandLine.ChosenPluginPackages("text,speech"), "an unknown package", ["IDRAK_PLUGIN_PACKAGES=text,speech", "'speech'", "text, vision, audio, reinforcement"]);
        Check(Idrak.Cli.Shared.EnvironmentVariables.All.Any(v => v.Name == CommandLine.PluginPackagesVariable), "the variable is in the tool's table");
        WithEnvironment(new Dictionary<string, string?> { [CommandLine.PluginPackagesVariable] = "speech" }, () =>
        {
            var (exit, _, error) = Cli("plugins", "list");
            Check(exit == 2 && error.Contains("IDRAK_PLUGIN_PACKAGES=speech", StringComparison.Ordinal), $"the tool refuses it naming the variable: {exit} {error}");
        });
        WithEnvironment(new Dictionary<string, string?> { [CommandLine.PluginPackagesVariable] = "none" }, () =>
            Check(Cli("plugins", "list", "-j").Exit == 0, "none: the tool runs with no package registered by it"));
    }

    private static void PluginPackagesNoInternals(Device device)
    {
        _ = device;
        var plugins = PluginPackages.Select(p => p.Name).ToHashSet(StringComparer.Ordinal);
        foreach (string name in ApiAssemblyNames.Append("Idrak.Cli").Append("Idrak.Cli.Commands"))
        {
            var friends = Assembly.Load(name).GetCustomAttributes<InternalsVisibleToAttribute>().Select(a => a.AssemblyName.Split(',')[0].Trim()).ToList();
            Check(!friends.Any(f => plugins.Contains(f) || f.EndsWith(".Plugins", StringComparison.Ordinal)),
                $"{name} grants its internals to {string.Join(", ", friends)}: plug-in packages use the public API only");
        }

        foreach (var package in PluginPackages)
        {
            var granted = package.Root.Assembly.GetCustomAttributes<InternalsVisibleToAttribute>().Select(a => a.AssemblyName.Split(',')[0].Trim()).ToList();
            Check(granted.All(a => a is "Idrak.Tests" or "Idrak.Tests.Library"), $"{package.Name} grants its internals to {string.Join(", ", granted)}: only the tests (the runner and its test library) may see them");
            Check(ApiAssemblyNames.Contains(package.Name) && LibraryAssemblyNames.Contains(package.Name), $"{package.Name} is in the API guard and the abstraction inventory");
        }

        var defaults = (string[])typeof(IdrakPlugins).Assembly.GetType("Idrak.Abstraction.LibraryDefaults", throwOnError: true)!
            .GetField("Assemblies", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!;
        Check(!defaults.Any(plugins.Contains), "the plug-in packages register nothing as library defaults (LibraryDefaults does not list them)");
        Check(defaults.Contains("Idrak.Reinforcement"), "Idrak.Reinforcement, a library package, is listed in LibraryDefaults (unlike its plug-in package)");
        Check(PluginPackages.All(p => p.Root.Assembly.GetCustomAttribute<LibraryRegistrationsAttribute>() is null), "the plug-in packages declare no library registrations");
    }

    private static void PluginPackagesMessages(Device device)
    {
        _ = device;
        const string None = "PluginsTestNobody";
        string[] text = ["Idrak.Text.Plugins", "TextPlugins.RegisterAll()"], vision = ["Idrak.Vision.Plugins", "VisionPlugins.RegisterAll()"],
            audio = ["Idrak.Audio.Plugins", "AudioPlugins.RegisterAll()"], reinforcement = ["Idrak.Reinforcement.Plugins", "ReinforcementPlugins.RegisterAll()"];
        ThrowsNaming<NotRegisteredException>(() => VisionFamilies.For(None, new JsonObject { ["vision_config"] = new JsonObject() }), "VisionFamilies",
            [$"'{None}' is not registered", "VisionFamilies.Register", .. vision]);
        ThrowsNaming<NotRegisteredException>(() => SpeechFamilies.Get(None), "SpeechFamilies", [$"'{None}' is not registered", "SpeechFamilies.Register", .. audio]);
        ThrowsNaming<NotRegisteredException>(() => SpeechSynthesisFamilies.Get(None), "SpeechSynthesisFamilies", [$"'{None}' is not registered", "SpeechSynthesisFamilies.Register", .. audio]);
        ThrowsNaming<NotRegisteredException>(() => VoiceConverters.Get(None), "VoiceConverters", [$"'{None}' is not registered", "VoiceConverters.Register", .. audio]);
        ThrowsNaming<NotRegisteredException>(() => GgufArchitectures.Get(None), "GgufArchitectures", [$"'{None}' is registered", "GgufArchitectures.Register", .. text]);
        ThrowsNaming<NotRegisteredException>(() => Idrak.Abstraction.Generation.ToolCallFormats.Create(None, new ToolCallContext(new ChatMLTemplate(), [], null)), "ToolCallFormats",
            [$"'{None}' is registered", "ToolCallFormats.Register", .. text]);
        ThrowsNaming<NotRegisteredException>(() => Idrak.Abstraction.Reinforcement.EnvironmentKinds.Get(None), "EnvironmentKinds",
            [$"'{None}' is not registered", "EnvironmentKinds.Register", .. reinforcement]);
        ThrowsNaming<NotRegisteredException>(() => Idrak.Abstraction.Reinforcement.RewardFunctions.Get(None), "RewardFunctions",
            [$"'{None}' is not registered", "RewardFunctions.Register", .. reinforcement]);
    }

    private static void PluginPackagesLibrariesPresent(Device device)
    {
        _ = device;
        const string Absent = "Idrak.NotShippedLibrary";
        var files = LibraryFiles.Present;
        Check(files is not null && ReferenceEquals(files, LibraryFiles.Present) && files.ByName.ContainsKey("Idrak") && files.ByName.ContainsKey("Idrak.Abstraction")
              && !files.ByName.ContainsKey(Absent),
            $"the runner's files are known, read once, and list Idrak and Idrak.Abstraction but not {Absent} ({files?.ByName.Count} files)");
        Check(files!.Program.Any(p => Path.GetFileNameWithoutExtension(p) == "Idrak.Abstraction")
              && files.Program.All(p => !string.Equals(Path.GetFileNameWithoutExtension(p), "System.Private.CoreLib", StringComparison.OrdinalIgnoreCase)
                                        || Path.GetDirectoryName(p) == Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory)),
            "the program's files hold the runner's own and not the shared framework's");
        Check(LibraryFiles.MayLoad("Idrak") && !LibraryFiles.MayLoad(Absent), $"Idrak may be loaded, {Absent} not");

        // Loading a set with a library the program lacks: nothing thrown for it, the others found.
        var thrown = new List<string>();
        void Seen(object? sender, System.Runtime.ExceptionServices.FirstChanceExceptionEventArgs e)
        {
            if (e.Exception is FileNotFoundException or FileLoadException && e.Exception.Message.Contains(Absent, StringComparison.Ordinal))
            {
                lock (thrown)
                {
                    thrown.Add($"{e.Exception.GetType().Name}: {e.Exception.Message}");
                }
            }
        }

        List<(Assembly, ILibraryRegistrations)> found;
        AppDomain.CurrentDomain.FirstChanceException += Seen;
        try
        {
            found = LibraryDefaults.Find(["Idrak", Absent, "Idrak.Text"]);
        }
        finally
        {
            AppDomain.CurrentDomain.FirstChanceException -= Seen;
        }

        Check(thrown.Count == 0, $"no first-chance exception for the library the program lacks: {string.Join("; ", thrown)}");
        Check(found.Select(f => f.Item1.GetName().Name).SequenceEqual(["Idrak", "Idrak.Text"]),
            $"the libraries present are found, in order: {string.Join(", ", found.Select(f => f.Item1.GetName().Name))}");

        // The same as loading every listed name (the way before): the runner ships most of them (rule 71).
        var listed = (string[])typeof(IdrakPlugins).Assembly.GetType("Idrak.Abstraction.LibraryDefaults", throwOnError: true)!
            .GetField("Assemblies", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!;
        var loadedEach = new List<string>();
        foreach (string name in listed)
        {
            try
            {
                if (Assembly.Load(new AssemblyName(name)).GetCustomAttribute<LibraryRegistrationsAttribute>() is not null)
                {
                    loadedEach.Add(name);
                }
            }
            catch (FileNotFoundException)
            {
                // a library this runner does not ship
            }
        }

        var foundAll = LibraryDefaults.Find(listed).Select(f => f.Item1.GetName().Name!).ToList();
        Check(foundAll.SequenceEqual(loadedEach) && foundAll.Count >= 10,
            $"every library the runner ships is found as loading each name finds it: {string.Join(", ", foundAll)} / {string.Join(", ", loadedEach)}");
    }

    private sealed class TestPackageCatalog(string package) : IPluginPackageCatalog
    {
        public string Name => "plugins-test";

        public string? PackageOf(string registry, string name) => registry == nameof(PretrainedArchitectures) && name == "PluginsTestArchitecture" ? package : null;
    }

    private static void PluginPackagesArchitectureMessage(Device device)
    {
        _ = device;
        const string None = "PluginsTestArchitecture", Registry = nameof(PretrainedArchitectures);
        var registeredBefore = PluginPackages.SelectMany(p => p.All).Where(p => p.IsRegistered).ToHashSet();
        try
        {
            foreach (var package in PluginPackages)
            {
                package.RegisterAll();
                var declared = package.Root.Assembly.GetCustomAttributes<IdrakPluginNamesAttribute>().Where(a => a.Registry == Registry).SelectMany(a => a.Names).ToList();
                var registered = PretrainedArchitectures.Names.Where(n => PretrainedArchitectures.Origin(n) == package.Name).ToList();
                Check(declared.Order(StringComparer.Ordinal).SequenceEqual(registered.Order(StringComparer.Ordinal), StringComparer.Ordinal),
                    $"{package.Name} declares the architectures it registers: declared [{string.Join(", ", declared)}], registered [{string.Join(", ", registered)}]");

                var read = new List<PresentPluginPackages.Declaration>();
                PresentPluginPackages.Read(package.Root.Assembly.Location, read);
                Check(read.Where(d => d.Registry == Registry).SelectMany(d => d.Names).SequenceEqual(declared) && read.All(d => d.Package == package.Name),
                    $"{package.Name}: its file's metadata holds the same declarations, without loading it: {string.Join("; ", read.Select(d => $"{d.Package} {d.Registry} [{string.Join(", ", d.Names)}]"))}");
                foreach (string name in declared)
                {
                    Check(PluginPackageCatalogs.PackageOf(Registry, name) == package.Name, $"the catalogs name {package.Name} for {name}: {PluginPackageCatalogs.PackageOf(Registry, name)}");
                }
            }

            var abstraction = new List<PresentPluginPackages.Declaration>();
            PresentPluginPackages.Read(typeof(IdrakPlugins).Assembly.Location, abstraction);
            Check(abstraction.Count == 0 && PluginPackageCatalogs.PackageOf(Registry, None) is null && PluginPackageCatalogs.PackageOf("VisionFamilies", "Qwen2ForCausalLM") is null,
                "the library declares no family, and a name nobody declares (or one asked of another registry) has no package");
            Check(PluginPackageCatalogs.Names.Contains(PluginPackageCatalogs.Present) && PluginPackageCatalogs.Origin(PluginPackageCatalogs.Present) == Overrides.Library
                  && PluginPackageCatalogs.Default(PluginPackageCatalogs.Present) is { Name: PluginPackageCatalogs.Present },
                $"the library's catalog is registered as a library default: {string.Join(", ", PluginPackageCatalogs.Names)}");
            ThrowsNaming<NotRegisteredException>(() => PluginPackageCatalogs.Get("plugins-test-nobody"), "PluginPackageCatalogs",
                ["'plugins-test-nobody' is registered", "PluginPackageCatalogs.Register"]);

            // The refusal of a text family nobody registered names the text package, never another.
            bool qwen = Idrak.Text.Plugins.Qwen.QwenPlugin.IsRegistered;
            Idrak.Text.Plugins.Qwen.QwenPlugin.Unregister();
            try
            {
                string message = Refusal(Idrak.Text.Plugins.Qwen.QwenPlugin.Qwen2ForCausalLM);
                Check(message.Contains("Idrak.Text.Plugins declares it", StringComparison.Ordinal) && message.Contains("RegisterAll()", StringComparison.Ordinal)
                      && !message.Contains("Idrak.Vision.Plugins", StringComparison.Ordinal) && !message.Contains("Idrak.Audio.Plugins", StringComparison.Ordinal),
                    $"the refusal of Qwen2ForCausalLM names Idrak.Text.Plugins alone: {message}");
            }
            finally
            {
                if (qwen)
                {
                    Idrak.Text.Plugins.Qwen.QwenPlugin.Register();
                }
            }

            string neutral = Refusal(None);
            Check(neutral.Contains("PretrainedArchitectures.Register", StringComparison.Ordinal) && neutral.Contains($"declares '{None}'", StringComparison.Ordinal)
                  && !neutral.Contains("Idrak.Vision.Plugins", StringComparison.Ordinal) && !neutral.Contains("Idrak.Text.Plugins", StringComparison.Ordinal),
                $"a family nobody declares is refused naming no package: {neutral}");

            // An app's catalog names a package the program does not ship.
            PluginPackageCatalogs.Register(new TestPackageCatalog("Contoso.Plugins"));
            try
            {
                Check(PluginPackageCatalogs.PackageOf(Registry, None) == "Contoso.Plugins" && Refusal(None).Contains("Contoso.Plugins declares it", StringComparison.Ordinal)
                      && PluginPackageCatalogs.Origin("plugins-test") == typeof(PluginPackageTestsMarker).Assembly.GetName().Name,
                    $"an app's catalog names its package: {Refusal(None)}");
            }
            finally
            {
                PluginPackageCatalogs.Unregister("plugins-test");
            }

            Check(PluginPackageCatalogs.PackageOf(Registry, None) is null && !PluginPackageCatalogs.Names.Contains("plugins-test"), "unregistered, the app's catalog is gone");
        }
        finally
        {
            foreach (var plugin in PluginPackages.SelectMany(p => p.All).Reverse())
            {
                if (!registeredBefore.Contains(plugin))
                {
                    IdrakPlugins.Unregister(plugin);
                }
            }
        }

        static string Refusal(string name)
        {
            try
            {
                PretrainedArchitectures.Get(name);
                return $"'{name}' was not refused";
            }
            catch (NotRegisteredException e)
            {
                return e.Registry == Registry && e.Name == name ? e.Message : $"refused by {e.Registry} for {e.Name}: {e.Message}";
            }
        }
    }

    // A type of this assembly, for the origin an app's registration gets.
    private sealed class PluginPackageTestsMarker;
}
