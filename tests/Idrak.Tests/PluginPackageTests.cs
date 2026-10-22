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
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] PluginPackageGroup =
    [
        ("plugins: packages: Idrak.Text.Plugins, Idrak.Vision.Plugins, Idrak.Audio.Plugins and Idrak.Reinforcement.Plugins load, each root class has All, RegisterAll, UnregisterAll and the -P entry RegisterIdrakPlugin, each family in a namespace of its own; RegisterAll twice leaves every registry the tool lists as once, and UnregisterAll leaves them as found", PluginPackagesRegisterAll),
        ("plugins: packages: an IdrakPlugin registers its parts once however often Register is called (from several threads too), Unregister leaves the registries as found, a failing part takes back the parts before it, Requires registers a needed plug-in only when missing; IdrakPlugins registers a plug-in once, replaces one of the same name (unregistering the earlier, which comes back when the new one fails) and names the packages when none is registered", PluginContract),
        ("plugins: packages: the tool registers the four packages at startup (text first, reinforcement last), and -P on each package's DLL loads it through the tool's loader (RegisterIdrakPlugin runs) adding nothing twice", PluginPackagesCli),
        ("plugins: packages: IDRAK_PLUGIN_PACKAGES chooses the packages the tool registers at startup (all four when unset, none, or a list taken in the tool's order) and is refused naming the variable when it names no package", PluginPackagesSetting),
        ("plugins: packages: no library assembly nor the tool grants a plug-in package its internals, the plug-in packages grant none but the test runner (as every library package does), and they are not library defaults", PluginPackagesNoInternals),
        ("plugins: packages: the not-registered messages of VisionFamilies, SpeechFamilies, SpeechSynthesisFamilies, VoiceConverters, GgufArchitectures, ToolCallFormats, EnvironmentKinds and RewardFunctions name the plug-in package to reference and its RegisterAll", PluginPackagesMessages),
    ];

    private sealed record PluginPackage(string Name, Type Root, IReadOnlyList<IIdrakPlugin> All, Action RegisterAll, Action UnregisterAll);

    private static readonly PluginPackage[] PluginPackages =
    [
        new(TextPlugins.Package, typeof(TextPlugins), TextPlugins.All, TextPlugins.RegisterAll, TextPlugins.UnregisterAll),
        new(VisionPlugins.Package, typeof(VisionPlugins), VisionPlugins.All, VisionPlugins.RegisterAll, VisionPlugins.UnregisterAll),
        new(AudioPlugins.Package, typeof(AudioPlugins), AudioPlugins.All, AudioPlugins.RegisterAll, AudioPlugins.UnregisterAll),
        new(ReinforcementPlugins.Package, typeof(ReinforcementPlugins), ReinforcementPlugins.All, ReinforcementPlugins.RegisterAll, ReinforcementPlugins.UnregisterAll),
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
            Check(package.Root.GetMethod("RegisterIdrakPlugin", BindingFlags.Public | BindingFlags.Static, Type.EmptyTypes) is { ReturnType: var r } && r == typeof(void),
                $"{package.Name}: a public static void RegisterIdrakPlugin() for -P");
            Check(package.Root.Assembly.GetExportedTypes().Count(t => t.GetMethod("RegisterIdrakPlugin", BindingFlags.Public | BindingFlags.Static, Type.EmptyTypes) is not null) == 1,
                $"{package.Name}: -P finds one RegisterIdrakPlugin (the root class's), so it registers the package once");
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
                    TokenSamplers.Register(sampler, TokenSamplers.Default(TokenSamplers.DefaultName)!);
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
        Check(CommandLine.PluginPackages.Select(p => p.Package).SequenceEqual(PluginPackages.Select(p => p.Name)), "the tool registers Idrak.Text.Plugins, Idrak.Vision.Plugins, Idrak.Audio.Plugins and Idrak.Reinforcement.Plugins, in that order");
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
        foreach (string name in ApiAssemblyNames.Append("Idrak.Cli"))
        {
            var friends = Assembly.Load(name).GetCustomAttributes<InternalsVisibleToAttribute>().Select(a => a.AssemblyName.Split(',')[0].Trim()).ToList();
            Check(!friends.Any(f => plugins.Contains(f) || f.EndsWith(".Plugins", StringComparison.Ordinal)),
                $"{name} grants its internals to {string.Join(", ", friends)}: plug-in packages use the public API only");
        }

        foreach (var package in PluginPackages)
        {
            var granted = package.Root.Assembly.GetCustomAttributes<InternalsVisibleToAttribute>().Select(a => a.AssemblyName.Split(',')[0].Trim()).ToList();
            Check(granted.All(a => a == "Idrak.Tests"), $"{package.Name} grants its internals to {string.Join(", ", granted)}: only the test runner may see them");
            Check(ApiAssemblyNames.Contains(package.Name) && LibraryAssemblyNames.Contains(package.Name), $"{package.Name} is in the API guard and the abstraction inventory");
        }

        var defaults = (ValueTuple<string, string>[])typeof(IdrakPlugins).Assembly.GetType("Idrak.Abstraction.LibraryDefaults", throwOnError: true)!
            .GetField("Assemblies", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!;
        Check(!defaults.Any(d => plugins.Contains(d.Item1)), "the plug-in packages register nothing as library defaults (LibraryDefaults does not list them)");
        Check(defaults.Any(d => d.Item1 == "Idrak.Reinforcement"), "Idrak.Reinforcement, a library package, is listed in LibraryDefaults (unlike its plug-in package)");
    }

    private static void PluginPackagesMessages(Device device)
    {
        _ = device;
        const string None = "PluginsTestNobody";
        string[] text = ["Idrak.Text.Plugins", "TextPlugins.RegisterAll()"], vision = ["Idrak.Vision.Plugins", "VisionPlugins.RegisterAll()"],
            audio = ["Idrak.Audio.Plugins", "AudioPlugins.RegisterAll()"], reinforcement = ["Idrak.Reinforcement.Plugins", "ReinforcementPlugins.RegisterAll()"];
        ThrowsNaming<NotSupportedException>(() => VisionFamilies.For(None, new JsonObject { ["vision_config"] = new JsonObject() }), "VisionFamilies",
            [$"'{None}' is not registered", "VisionFamilies.Register", .. vision]);
        ThrowsNaming<NotSupportedException>(() => SpeechFamilies.Get(None), "SpeechFamilies", [$"'{None}' is not registered", "SpeechFamilies.Register", .. audio]);
        ThrowsNaming<NotSupportedException>(() => SpeechSynthesisFamilies.Get(None), "SpeechSynthesisFamilies", [$"'{None}' is not registered", "SpeechSynthesisFamilies.Register", .. audio]);
        ThrowsNaming<NotSupportedException>(() => VoiceConverters.Get(None), "VoiceConverters", [$"'{None}' is not registered", "VoiceConverters.Register", .. audio]);
        ThrowsNaming<NotSupportedException>(() => GgufArchitectures.Get(None), "GgufArchitectures", [$"'{None}' is registered", "GgufArchitectures.Register", .. text]);
        ThrowsNaming<NotSupportedException>(() => Idrak.Abstraction.Generation.ToolCallFormats.Create(None, new ToolCallContext(new ChatMLTemplate(), [], null)), "ToolCallFormats",
            [$"'{None}' is registered", "ToolCallFormats.Register", .. text]);
        ThrowsNaming<NotSupportedException>(() => Idrak.Abstraction.Reinforcement.EnvironmentKinds.Get(None), "EnvironmentKinds",
            [$"'{None}' is not registered", "EnvironmentKinds.Register", .. reinforcement]);
        ThrowsNaming<NotSupportedException>(() => Idrak.Abstraction.Reinforcement.RewardFunctions.Get(None), "RewardFunctions",
            [$"'{None}' is not registered", "RewardFunctions.Register", .. reinforcement]);
    }
}
