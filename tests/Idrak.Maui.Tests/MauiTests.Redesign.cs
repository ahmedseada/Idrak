// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.RegularExpressions;
using Idrak.Diagnostics;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.Markdown;
using Idrak.Maui.Navigation;
using Idrak.Maui.Theme;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Controls.Shapes;
using Microsoft.Maui.Dispatching;

using DeviceInfo = Idrak.Diagnostics.DeviceInfo;
using Path = System.IO.Path;

namespace Idrak.Maui.Tests;

// Plan 34, phase R: the approved redesign. The navigation host (bottom bar or rail by the window's width), the sections,
// All labs, Home, the Markdown renderer, the kept conversations, the theme (both palettes, contrast, no literals), the icons,
// the source commit, the settings' controls and the devices' summary cards.
internal static partial class MauiTests
{
    // A device listing without starting a backend: a CPU, a GPU that started, one that did not, and a virtual one.
    private sealed class ListedDevices : IDeviceInformation
    {
        public int Reads { get; private set; }

        public DeviceReport Read(IReadOnlyList<string>? devices = null)
        {
            Reads++;
            var cpu = new DeviceInfo
            {
                Device = "cpu", Kind = "cpu", Backend = "CPU", Name = "Test CPU", Listed = true, MemoryBytes = 8L << 30, ComputeUnits = 8, SubgroupSize = 4,
                Facts = [new("instruction sets", "NEON, DotProd, RDM"), new("caches", "L1 64 KB")],
            };
            var gpu = new DeviceInfo
            {
                Device = "vulkan:0", Kind = "vulkan", Backend = "Vulkan", Name = "Test GPU", Listed = true, IsDefault = true, MemoryBytes = 4L << 30, KernelWidth = 64,
                SubgroupSize = 32, HardwareKind = "integrated", Driver = "1.3",
            };
            var failed = new DeviceInfo { Device = "vulkan:1", Kind = "vulkan", Backend = "Vulkan", Name = "", Listed = true, Error = "the driver refused" };
            var deferred = new DeviceInfo { Device = "deferred:0", Kind = "deferred", Backend = "deferred", Name = "deferred", Listed = false };
            var process = new ProcessFacts("Test OS", "Arm64", ".NET 10", 8, 8L << 30, 2L << 30, 100 << 20, "/cache", 8);
            return new DeviceReport([new(cpu, new MemoryUsage(0, 0, null), 0), new(gpu, null, 3), new(failed, null, 0), new(deferred, null, 0)],
                [new BackendInfo("vulkan", "Vulkan", 2, null), new BackendInfo("cuda", "CUDA", 0, "no CUDA driver")], process, "vulkan:0", TimeSpan.FromMilliseconds(12));
        }
    }

    // The UI thread's dispatcher for views made in a test (MAUI's bindings between elements ask for one): runs at once.
    private sealed class TestDispatcherProvider : IDispatcherProvider
    {
        private readonly InlineDispatcher _dispatcher = new();

        public IDispatcher GetForCurrentThread() => _dispatcher;
    }

    private static bool _hasDispatcher;

    // Set once, when the first test that builds views runs (the earlier tests run without one, as they always have).
    private static void EnsureDispatcher() => _hasDispatcher = _hasDispatcher || DispatcherProvider.SetCurrent(new TestDispatcherProvider()) || true;

    // The services an app's UseIdrak gives, without a platform: in-memory stores, the listing above, a scripted chat.
    private static ServiceProvider AppServices(IdrakMauiOptions? options = null, StartupRun? run = null)
    {
        EnsureDispatcher();
        options ??= new IdrakMauiOptions { Device = "cpu" };
        var models = new ModelService(new ManualMemoryPressure(), options);
        var settings = new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore());
        var store = new ListedStore(new StoredModel("Qwen/Qwen3-0.6B", "0123456789ab", Path.GetTempPath(), 734_000_000, 9, true, DateTimeOffset.UtcNow),
            new StoredModel("openai/whisper-tiny", "abcdef012345", Path.GetTempPath(), 151_000_000, 5, true, DateTimeOffset.UtcNow));
        var devices = new ListedDevices();
        var conversations = new MemoryConversationStore();
        var startup = new IdrakStartupReport(ComputeResources.CacheFolder, options.Device, [], Path.GetTempPath());
        var collection = new ServiceCollection()
            .AddSingleton(options).AddSingleton(settings).AddSingleton(startup).AddSingleton<ISettingsStore>(new MemorySettingsStore())
            .AddSingleton<IModelStore>(store).AddSingleton<IModelService>(models).AddSingleton<IDeviceInformation>(devices)
            .AddSingleton<IPermissionService>(new FakePermissions(true)).AddSingleton<IShareService>(new FakeShare()).AddSingleton<IConversationStore>(conversations)
            .AddSingleton(new ProgressCenter(options.ProgressInterval))
            .AddSingleton(new AppLayout(options.WideLayoutWidth))
            .AddSingleton(sp => new DeviceStatus(sp.GetRequiredService<IDeviceInformation>(), sp.GetService<IModelStore>()))
            .AddSingleton<IAppNavigator>(sp => new AppNavigator(sp))
            .AddSingleton(sp => new ChatViewModel(store, new OneChat(new ScriptedChat(3, "ab", Stats(3, 1))), options, new InlineDispatcher(), conversations: conversations))
            .AddTransient(sp => new SettingsViewModel(options, settings, startup));
        if (run is not null)
        {
            collection.AddSingleton(run);
        }

        return collection.BuildServiceProvider();
    }

    // A view on a page carrying Idrak's styles and the colours of a theme (as the app's resources carry them).
    private static ContentPage Themed(View view, AppTheme theme)
    {
        var page = new ContentPage { Content = view };
        page.Resources.MergedDictionaries.Add(new IdrakStyles());
        page.Resources.MergedDictionaries.Add(IdrakThemes.Colors(theme));
        return page;
    }

    private static IEnumerable<IVisualTreeElement> Tree(IVisualTreeElement root)
    {
        yield return root;
        foreach (var child in root.GetVisualChildren())
        {
            foreach (var e in Tree(child))
            {
                yield return e;
            }
        }
    }

    // Every label's text colour is one of the theme's colours (its keys resolved through the page's resources).
    private static int CheckLabels(VisualElement root, AppTheme theme, string what)
    {
        var colors = IdrakThemes.Colors(theme).Values.OfType<Color>().Select(c => c.ToArgbHex(true)).ToHashSet(StringComparer.OrdinalIgnoreCase);
        int labels = 0;
        foreach (var label in Tree(root).OfType<Label>())
        {
            labels++;
            Check(label.TextColor is { } color && colors.Contains(color.ToArgbHex(true)),
                $"{what} in {theme}: label '{label.Text}' has text colour {label.TextColor?.ToArgbHex(true) ?? "none"}, not one of the theme's");
        }

        return labels;
    }

    private static void RedesignViewsBind()
    {
        using var services = AppServices();
        foreach (var theme in new[] { AppTheme.Light, AppTheme.Dark })
        {
            int labels = 0;
            foreach (var section in AppSections.Visible(services.GetRequiredService<IdrakMauiOptions>()))
            {
                var view = section.CreateView(services);
                var page = Themed(view, theme);
                Check(page.BackgroundColor is not null && page.BackgroundColor.Equals(IdrakThemes.Colors(theme)["IdrakPageBackground"]), $"{section.Name} in {theme}: the page's ground");
                labels += CheckLabels(view, theme, section.Name);
                if (view is SegmentedSectionView segmented)
                {
                    foreach (var segment in segmented.ViewModel.Segments.ToList())
                    {
                        segmented.ShowFeature(segment.Name);
                        Check(segmented.Current is IHostedView && segmented.ViewModel.Selected == segment, $"{section.Name}: {segment.Name} shown");
                        labels += CheckLabels(segmented.Current!, theme, segment.Name);
                    }
                }
            }

            foreach (var feature in MobileFeatures.Enabled(services.GetRequiredService<IdrakMauiOptions>()))
            {
                var page = MobileFeatures.Open(feature.Name, services);
                page.Resources.MergedDictionaries.Add(new IdrakStyles());
                page.Resources.MergedDictionaries.Add(IdrakThemes.Colors(theme));
                Check(page is LabPage { View: not null } && page.Title == feature.Title, $"{feature.Name}: {page.GetType().Name} titled '{page.Title}'");
                labels += CheckLabels(page, theme, feature.Name + "'s page");
                foreach (bool wide in (bool[])[true, false])
                {
                    ((page as LabPage)?.View as AdaptiveView)?.Apply(wide);                       // both layouts lay out
                }
            }

            labels += CheckLabels(Themed(SettingsPages.CreateView(services), theme), theme, "settings");
            Check(labels > 200, $"{labels} labels checked in {theme}");
        }
    }

    private static void RedesignHostSwitches()
    {
        var options = new IdrakMauiOptions { Device = "cpu", WideLayoutWidth = 840 };
        using var services = AppServices(options);
        var host = new AppHostPage(services);
        var bar = host.FindByName<View>("Bar");
        var rail = host.FindByName<View>("Rail");
        Check(host.ViewModel.Tabs.Select(t => t.Name).SequenceEqual([AppSections.Home, AppSections.Chat, AppSections.Voice, AppSections.Vision])
              && host.ViewModel.Foot.Select(t => t.Name).SequenceEqual([AppSections.Labs, AppSections.Settings]),
            $"tabs {string.Join(", ", host.ViewModel.Tabs.Select(t => t.Name))}; foot {string.Join(", ", host.ViewModel.Foot.Select(t => t.Name))}");
        Check(host.Current is HomeView && host.ViewModel.Find(AppSections.Home)!.IsSelected, "Home first");

        host.ApplySize(412, 915);                                                        // a phone, upright
        Check(host.ViewModel.IsNarrow && bar.IsVisible && !rail.IsVisible, "a narrow window: the bottom bar");
        host.ApplySize(1280, 800);                                                       // a tablet, or the phone turned in a wide window
        Check(host.ViewModel.IsWide && !bar.IsVisible && rail.IsVisible, "a wide window: the rail");
        host.ApplySize(600, 900);                                                        // split screen
        Check(host.ViewModel.IsNarrow && bar.IsVisible && !rail.IsVisible, "narrow again: the bar again");

        // The breakpoint is a setting: the same window turns wide when it is lowered.
        services.GetRequiredService<AppLayout>().Breakpoint = 560;
        Check(host.ViewModel.IsWide && rail.IsVisible, "a lower breakpoint: the same 600 is wide");
        services.GetRequiredService<AppLayout>().Breakpoint = 840;

        // Sections and features: a tab shows its view (made once), a feature opens in the section that hosts it.
        Check(host.OpenSectionAsync(AppSections.Voice).GetAwaiter().GetResult() && host.Current is SegmentedSectionView voice
              && host.ViewModel.Selected == AppSections.Voice && host.ViewModel.Find(AppSections.Voice)!.IsSelected && !host.ViewModel.Find(AppSections.Home)!.IsSelected,
            $"voice shown: {host.Current?.GetType().Name}");
        var voiceView = host.Current;
        host.OpenFeatureAsync(MobileFeatures.Speak).GetAwaiter().GetResult();
        Check(ReferenceEquals(host.Current, voiceView) && ((SegmentedSectionView)voiceView!).Current is SpeakView, "Speak opens in Voice, the same view");
        host.OpenFeatureAsync(MobileFeatures.Image).GetAwaiter().GetResult();
        Check(host.ViewModel.Selected == AppSections.Vision && ((SegmentedSectionView)host.Current!).Current is ImageView, "Image opens in Vision");
        host.ViewModel.Find(AppSections.Chat)!.SelectCommand.Execute(null);
        Check(host.Current is ChatView, $"the chat tab: {host.Current?.GetType().Name}");

        // All labs: a sheet over the page on a narrow window, the rail's page on a wide one.
        host.OpenLabsAsync().GetAwaiter().GetResult();
        Check(host.ViewModel.IsSheetOpen && host.FindByName<ContentView>("SheetBody").Content is LabsView && host.Current is ChatView, "narrow: All labs as a sheet");
        host.ViewModel.CloseSheetCommand.Execute(null);
        host.ApplySize(1280, 800);
        host.OpenLabsAsync().GetAwaiter().GetResult();
        Check(!host.ViewModel.IsSheetOpen && host.Current is LabsView && host.ViewModel.Find(AppSections.Labs)!.IsSelected, "wide: All labs as the rail's page");
        Check(host.OpenSettingsAsync() is not null && host.OpenSectionAsync(AppSections.Settings).GetAwaiter().GetResult() && host.Current is SettingsView, "wide: the settings in the rail");

        // Features turned off take their section with them when they were all it showed; an unknown section is refused.
        options.DisabledFeatures.UnionWith([MobileFeatures.Ocr, MobileFeatures.Image, MobileFeatures.Generate]);
        host.ViewModel.Refresh();
        Check(host.ViewModel.Find(AppSections.Vision) is null && host.ViewModel.Tabs.Count == 3, $"tabs without vision: {string.Join(", ", host.ViewModel.Tabs.Select(t => t.Name))}");
        Check(!host.OpenSectionAsync("nowhere").GetAwaiter().GetResult(), "an unknown section opened");
        options.DisabledFeatures.Clear();

        // An app's section joins the tabs in its order, and the library's comes back when it is removed.
        var mine = new FeatureSection("lab-x", "Lab X", "spark", 25, MobileFeatures.Devices, "the app's own");
        try
        {
            AppSections.Register(mine);
            host.ViewModel.Refresh();
            Check(host.ViewModel.Tabs.Select(t => t.Name).SequenceEqual([AppSections.Home, AppSections.Chat, AppSections.Voice, "lab-x", AppSections.Vision])
                  && AppSections.Origin("lab-x") == typeof(MauiTests).Assembly.GetName().Name, $"tabs {string.Join(", ", host.ViewModel.Tabs.Select(t => t.Name))}");
            Check(host.OpenSectionAsync("lab-x").GetAwaiter().GetResult() && host.Current is DevicesView, "the app's section shows its feature's view");
        }
        finally
        {
            AppSections.Unregister("lab-x");
        }

        try
        {
            AppSections.Get("nope");
            Check(false, "an unknown section was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Message.Contains("home", StringComparison.Ordinal), e.Message);
        }
    }

    private static void RedesignLabs()
    {
        var options = new IdrakMauiOptions();
        using var services = AppServices(options);
        var status = services.GetRequiredService<DeviceStatus>();
        status.RefreshAsync().GetAwaiter().GetResult();
        var labs = new LabsViewModel(options, status);
        string Groups() => string.Join(" | ", labs.Groups.Select(g => $"{g.Title}: {string.Join(", ", g.Select(l => l.Name))}"));
        Check(labs.Groups.Select(g => g.Group).SequenceEqual([FeatureGroup.Language, FeatureGroup.Voice, FeatureGroup.Vision, FeatureGroup.Device]), Groups());
        Check(labs.Groups[0].Select(l => l.Name).SequenceEqual([MobileFeatures.Chat, MobileFeatures.Search])
              && labs.Groups[1].Select(l => l.Name).SequenceEqual([MobileFeatures.Transcribe, MobileFeatures.Speak, MobileFeatures.AudioProcess])
              && labs.Groups[2].Select(l => l.Name).SequenceEqual([MobileFeatures.Ocr, MobileFeatures.Image, MobileFeatures.Generate])
              && labs.Groups[3].Select(l => l.Name).SequenceEqual([MobileFeatures.Devices, MobileFeatures.Models, MobileFeatures.Terminal, AppSections.Settings]), Groups());
        var search = labs.All.Single(l => l.Name == MobileFeatures.Search);
        var chat = labs.All.Single(l => l.Name == MobileFeatures.Chat);
        Check(search is { IsPlanned: true, StateText: "● Planned", Icon: "search", Tone: "Sky" } && chat is { IsReady: true, Tagline: "Stream a model", Icon: "chat", Tone: "Indigo" },
            $"search {search.StateText} {search.Icon}; chat {chat.Tagline} {chat.Icon}");
        Check(labs.CountsText == "11 ready · 1 planned" && labs.All.Single(l => l.Name == MobileFeatures.AudioProcess).Title == "Studio"
              && labs.All.Single(l => l.Name == MobileFeatures.Ocr).Title == "Read text", labs.CountsText);
        Check(labs.All.Single(l => l.Name == MobileFeatures.Devices).Detail == "CPU · GPU ready" && labs.All.Single(l => l.Name == MobileFeatures.Models).Detail == $"2 pulled · {ProgressText.Bytes(885_000_000)}",
            $"{labs.All.Single(l => l.Name == MobileFeatures.Devices).Detail}; {labs.All.Single(l => l.Name == MobileFeatures.Models).Detail}");

        // The search: a lab by its words, a setting by its title (the Settings card), a model by its id (the Models card).
        labs.Query = "speech";
        Check(labs.Groups.SelectMany(g => g).Select(l => l.Name).Contains(MobileFeatures.Transcribe) && !labs.Groups.SelectMany(g => g).Any(l => l.Name == MobileFeatures.Image), Groups());
        labs.Query = "temperature";
        Check(labs.Groups.Single().Single().Name == AppSections.Settings, Groups());
        labs.Query = "whisper-tiny";
        Check(labs.Groups.Single().Single().Name == MobileFeatures.Models, Groups());
        labs.Query = "nothing like it";
        Check(labs.IsEmpty, Groups());
        labs.Query = "";

        // The filters.
        labs.Filters.Single(f => f.Filter == LabFilter.Planned).SelectCommand.Execute(null);
        Check(labs.Filter == LabFilter.Planned && labs.Groups.SelectMany(g => g).Select(l => l.Name).SequenceEqual([MobileFeatures.Search]) && labs.Filters.Single(f => f.IsSelected).Filter == LabFilter.Planned, Groups());
        labs.Filter = LabFilter.Microphone;
        Check(labs.Groups.SelectMany(g => g).Select(l => l.Name).SequenceEqual([MobileFeatures.Transcribe, MobileFeatures.AudioProcess]), Groups());
        labs.Filter = LabFilter.Camera;
        Check(labs.Groups.SelectMany(g => g).Select(l => l.Name).SequenceEqual([MobileFeatures.Ocr, MobileFeatures.Image]), Groups());
        labs.Filter = LabFilter.All;

        // A feature turned off is not listed; the app's planned Self-test joins this device's group with its own icon.
        options.DisabledFeatures.Add(MobileFeatures.Speak);
        var selfTest = new PlannedFeature("self-test", "Self-test", "✓", 90, "The library's tests, run on this phone", "A test group by filter",
            icon: "shield", tagline: "The library's own tests");
        try
        {
            MobileFeatures.Register(selfTest);
            labs.Refresh();
            Check(!labs.All.Any(l => l.Name == MobileFeatures.Speak) && labs.Groups[3].Select(l => l.Name).SequenceEqual([MobileFeatures.Devices, MobileFeatures.Models, MobileFeatures.Terminal, "self-test", AppSections.Settings])
                  && labs.All.Single(l => l.Name == "self-test") is { IsPlanned: true, Icon: "shield", Tone: "Neutral" }, Groups());
        }
        finally
        {
            MobileFeatures.Unregister("self-test");
        }
    }

    private static void RedesignMarkdown()
    {
        const string text = "Two things make it fit:\n1. **Small weights** — 0.6B parameters in int8.\n2. **The GPU** — `Vulkan` kernels, *measured*.\n\n"
            + "## Code\n```csharp\nvar model = Load(\"x\");\n```\n- one\n- two\n\nالنموذج **صغير** بما يكفي";
        var blocks = MarkdownDocument.Parse(text);
        string Kinds() => string.Join(", ", blocks.Select(b => b.GetType().Name));
        Check(blocks.Count == 6 && blocks[0] is MarkdownParagraph { Text: "Two things make it fit:", IsHeading: false } && blocks[1] is MarkdownList { Ordered: true, Items.Count: 2 }
              && blocks[2] is MarkdownParagraph { Level: 2, Text: "Code" } && blocks[3] is MarkdownCode { Language: "csharp", LanguageTitle: "C#", IsOpen: false }
              && blocks[4] is MarkdownList { Ordered: false } && blocks[5] is MarkdownParagraph, Kinds());
        var first = ((MarkdownList)blocks[1]).Items[0];
        Check(first.Marker == "1." && first.Runs.SequenceEqual([new MarkdownRun("Small weights", MarkdownRunStyle.Bold), new MarkdownRun(" — 0.6B parameters in int8.", MarkdownRunStyle.Plain)]),
            string.Join(" | ", first.Runs));
        var second = ((MarkdownList)blocks[1]).Items[1].Runs;
        Check(second.Any(r => r is { Text: "Vulkan", Style: MarkdownRunStyle.Code }) && second.Any(r => r is { Text: "measured", Style: MarkdownRunStyle.Italic }), string.Join(" | ", second));
        Check(((MarkdownCode)blocks[3]).Code == "var model = Load(\"x\");" && ((MarkdownList)blocks[4]).Items.Select(i => i.Marker + i.Text).SequenceEqual(["•one", "•two"]),
            ((MarkdownCode)blocks[3]).Code);

        // Right-to-left text is kept as written and its block says its direction.
        var arabic = (MarkdownParagraph)blocks[5];
        Check(arabic.RightToLeft && string.Concat(arabic.Runs.Select(r => r.Text)) == "النموذج صغير بما يكفي" && arabic.Runs[1] == new MarkdownRun("صغير", MarkdownRunStyle.Bold)
              && !((MarkdownParagraph)blocks[0]).RightToLeft, string.Join(" | ", arabic.Runs));

        // Streaming: an open bold holds to the end, an open fence is code to the end; the blocks above are kept, the last
        // changed in place.
        Check(MarkdownBlock.Inline("**Small wei").Single() == new MarkdownRun("Small wei", MarkdownRunStyle.Bold), "an open bold");
        Check(MarkdownDocument.Parse("```py\nprint(1)").Single() is MarkdownCode { IsOpen: true, Code: "print(1)", LanguageTitle: "Python" }, "an open fence");
        string copied = "";
        var document = new MarkdownDocument(code =>
        {
            copied = code;
            return Task.CompletedTask;
        });
        document.Update("Hello");
        var paragraph = document.Blocks.Single();
        int changes = 0;
        paragraph.PropertyChanged += (_, _) => changes++;
        document.Update("Hello there");
        Check(ReferenceEquals(document.Blocks.Single(), paragraph) && changes == 1 && ((MarkdownParagraph)paragraph).Text == "Hello there", $"{changes} changes in place");
        document.Update("Hello there\n\n```cs\nint a");
        var code = (MarkdownCode)document.Blocks[1];
        document.Update("Hello there\n\n```cs\nint a = 1;\n```");
        Check(ReferenceEquals(document.Blocks[0], paragraph) && ReferenceEquals(document.Blocks[1], code) && code is { Code: "int a = 1;", IsOpen: false } && document.Blocks.Count == 2,
            $"{document.Blocks.Count} blocks; code '{code.Code}'");
        code.CopyCommand!.Execute(code.Code);
        Check(copied == "int a = 1;", $"copied '{copied}'");
        document.Update("Other");
        Check(document.Blocks.Count == 1 && document.Changes >= 4, $"{document.Blocks.Count} blocks after a shorter text");

        // The view draws a block a child, a paragraph's runs as spans in the theme's fonts.
        var view = new MarkdownView { Document = new MarkdownDocument() };
        view.Document.Update(text);
        var page = Themed(view, AppTheme.Dark);
        var children = ((Layout)view.Content).Children;
        Check(children.Count == 6 && children[0] is Label { FormattedText.Spans.Count: 1 } && children[5] is Label { FlowDirection: FlowDirection.RightToLeft } && page is not null,
            $"{children.Count} children");
    }

    private static void RedesignConversations()
    {
        string folder = Temp("conversations");
        var store = new FileConversationStore(folder);
        var now = DateTimeOffset.UtcNow;
        var chat = new ChatConversation(ConversationText.NewId(now), ConversationText.Title("How can a phone run a model?\nMore text"), now, "Qwen/Qwen3-0.6B",
            [new ChatMessage("user", "How can a phone run a model?"), new ChatMessage("assistant", "Two things.\n1. **Small** weights", "I think"), new ChatMessage("user", "قلها بالعربية")]);
        store.Save(chat);
        var back = store.Load(chat.Id);
        Check(back is not null && back.Title == "How can a phone run a model?" && back.Model == chat.Model && back.Messages.SequenceEqual(chat.Messages)
              && back.Updated == chat.Updated, $"read back: {back?.Title}, {back?.Messages.Count} messages");
        var older = chat with { Id = ConversationText.NewId(now.AddDays(-1)), Title = "Earlier", Updated = now.AddDays(-1) };
        store.Save(older);
        var listed = store.List();
        Check(listed.Select(c => c.Id).SequenceEqual([chat.Id, older.Id]) && listed[0].Preview == "قلها بالعربية", string.Join(", ", listed.Select(c => c.Title)));
        Check(ConversationText.Group(now, now) == "Today" && ConversationText.Group(now.AddDays(-1), now) == "Yesterday" && ConversationText.Group(now.AddDays(-30), now) == "Earlier",
            "the day groups");
        File.WriteAllText(Path.Combine(folder, "broken.json"), "{ not json");
        Check(store.List().Count == 2, "a file that does not read is not listed");
        Check(store.Delete(older.Id) && !store.Delete(older.Id) && store.List().Count == 1, "deleted");
        foreach (string bad in (string[])["../x", "a/b", ""])
        {
            bool refused = false;
            try
            {
                store.Load(bad);
            }
            catch (ArgumentException)
            {
                refused = true;
            }

            Check(refused, $"'{bad}' was taken as an id");
        }

        // The chat keeps each answered chat, lists it today, opens a kept one and starts anew.
        string modelFolder = ChatFolder("chat-kept");
        var kept = new MemoryConversationStore();
        var page = new ChatViewModel(new ListedStore(Stored(modelFolder)), new OneChat(new ScriptedChat(3, "ab", Stats(3, 1))), new IdrakMauiOptions(), new InlineDispatcher(),
            conversations: kept);
        page.RefreshAsync().GetAwaiter().GetResult();
        page.SelectedModel = page.Models[0];
        page.Loading!.GetAwaiter().GetResult();
        Check(page.ShowSuggestions && page.Suggestions.Count == 3, "suggestions while empty");
        page.SuggestCommand.Execute(page.Suggestions[0]);
        SpinUntil(() => !page.IsStreaming && page.Messages.Count == 2);
        Check(!page.ShowSuggestions && kept.List().Single() is { } first && first.Title == page.Suggestions[0] && first.Preview == "ababab", $"kept: {kept.List().Count}");
        var bubble = page.Messages[^1];
        Check(bubble is { CanRegenerate: true, HasMeta: true, TokensText: "3 tokens", RateText: "3.0 tok/s" } && bubble.Document.Blocks.Count == 1, $"meta '{bubble.RateText}' '{bubble.TokensText}'");
        Check(page.Conversations.Single().Title == "Today" && page.Conversations[0].Single().IsCurrent, "listed today, current");
        string id = page.ConversationId;
        Check(page.RegenerateAsync().GetAwaiter().GetResult() && page.Messages.Count == 2 && kept.List().Count == 1, "asked again in place");
        page.NewChat();
        Check(page.Messages.Count == 0 && page.ConversationId != id && !page.Conversations[0].Single().IsCurrent, "a new chat");
        Check(page.OpenConversation(id) && page.Messages.Count == 2 && page.History.Count == 2 && page.Messages[1].Text == "ababab" && page.ConversationId == id, "opened again");
        page.ConversationQuery = "nothing like it";
        Check(page.Conversations.Count == 0, "the search hides it");
        page.ConversationQuery = "";
        Check(page.DeleteConversation(id) && kept.List().Count == 0 && page.Messages.Count == 0, "forgotten");
    }

    private static void SpinUntil(Func<bool> done)
    {
        var clock = Stopwatch.StartNew();
        while (!done())
        {
            Check(clock.Elapsed < TimeSpan.FromSeconds(20), "timed out waiting");
            Thread.Sleep(5);
        }
    }

    private static void RedesignTheme()
    {
        var light = IdrakThemes.Colors(AppTheme.Light);
        var dark = IdrakThemes.Colors(AppTheme.Dark);
        Check(light.Keys.ToHashSet().SetEquals(dark.Keys), $"only in light: {string.Join(", ", light.Keys.Except(dark.Keys))}; only in dark: {string.Join(", ", dark.Keys.Except(light.Keys))}");
        foreach (var (key, value) in light)
        {
            Check(dark[key].GetType() == value.GetType(), $"{key}: {value.GetType().Name} in light, {dark[key].GetType().Name} in dark");
        }

        // The design's dark palette (apps/Idrak.Mobile/Design/base.css).
        foreach (var (key, hex) in new[] { ("IdrakPageBackground", "060A14"), ("IdrakSurface", "0F1729"), ("IdrakSurface2", "141D33"), ("IdrakSurface3", "1A2440"),
                     ("IdrakText", "EAF0FA"), ("IdrakTextMuted", "8B98B4"), ("IdrakSky", "38BDF8"), ("IdrakIndigo", "6366F1"), ("IdrakAmber", "F59E0B"),
                     ("IdrakGreen", "34D399"), ("IdrakRed", "F43F5E"), ("IdrakViolet", "A78BFA") })
        {
            Check(dark[key] is Color c && c.ToArgbHex().EndsWith(hex, StringComparison.OrdinalIgnoreCase), $"{key}: {(dark[key] as Color)?.ToArgbHex()} for the design's {hex}");
        }

        // Every key a page, a view or a control names is in the styles, the icons and both palettes.
        var styles = new IdrakStyles();
        bool Has(string key, ResourceDictionary colors) => styles.ContainsKey(key) || styles.MergedDictionaries.Any(d => d.ContainsKey(key)) || colors.ContainsKey(key);
        var named = new HashSet<string>(StringComparer.Ordinal);
        foreach (string file in Xaml())
        {
            named.UnionWith(Regex.Matches(File.ReadAllText(file), @"DynamicResource (\w+)").Select(m => m.Groups[1].Value));
        }

        foreach (string file in Directory.EnumerateFiles(Path.Combine(RepositoryRoot(), "src", "Idrak.Maui"), "*.cs", SearchOption.AllDirectories))
        {
            named.UnionWith(Regex.Matches(File.ReadAllText(file), @"SetDynamicResource\([^;]*?""(Idrak\w+)""").Select(m => m.Groups[1].Value));
        }

        foreach (string tone in (string[])["Indigo", "Violet", "Green", "Rose", "Amber", "Sky", "Neutral"])
        {
            named.UnionWith([$"IdrakTone{tone}", $"IdrakTone{tone}Background"]);
        }

        named.UnionWith(HomeOptions.LibrarySlides.Select(s => $"IdrakBrushHero{s.Tone}"));
        foreach (var colors in new[] { light, dark })
        {
            var missing = named.Where(k => !Has(k, colors)).Order(StringComparer.Ordinal).ToList();
            Check(missing.Count == 0, $"named but not defined: {string.Join(", ", missing)}");
        }

        Check(named.Count > 150, $"{named.Count} keys named");

        // Text reads at 4.5:1 or more on the grounds it sits on, in both themes.
        foreach (var (theme, colors) in new[] { (AppTheme.Light, light), (AppTheme.Dark, dark) })
        {
            Color Get(string key) => colors[key] as Color ?? throw new CheckException($"{theme}: {key} is not a colour");
            foreach (string text in (string[])["IdrakText", "IdrakTextSoft", "IdrakTextMuted", "IdrakTextFaint", "IdrakAccentText", "IdrakError", "IdrakSuccess", "IdrakHighlight",
                         "IdrakPlannedText", "IdrakThinkText", "IdrakRecordText", "IdrakChipText", "IdrakToneIndigo", "IdrakToneViolet", "IdrakToneGreen", "IdrakToneRose",
                         "IdrakToneAmber", "IdrakToneSky"])
            {
                foreach (string ground in (string[])["IdrakPageBackground", "IdrakSurface", "IdrakSurface2"])
                {
                    double ratio = Contrast(Get(text), Get(ground));
                    Check(ratio >= 4.5, $"{theme}: {text} on {ground} reads at {ratio:F2}:1");
                }
            }

            foreach (string stop in (string[])["IdrakGradientStart", "IdrakGradientEnd"])
            {
                Check(Contrast(Get("IdrakOnGradient"), Get(stop)) >= 4.5, $"{theme}: text on the gradient's {stop}: {Contrast(Get("IdrakOnGradient"), Get(stop)):F2}:1");
            }

            Check(Contrast(Get("IdrakOnAccent"), Get("IdrakAccent")) >= 4.5, $"{theme}: text on the accent");
        }
    }

    // Every XAML file of the library's pages and views and of the app (the theme's dictionaries are where the literals are).
    private static IEnumerable<string> Xaml() =>
        Directory.EnumerateFiles(Path.Combine(RepositoryRoot(), "src", "Idrak.Maui"), "*.xaml", SearchOption.AllDirectories)
            .Concat(Directory.EnumerateFiles(Path.Combine(RepositoryRoot(), "apps", "Idrak.Mobile"), "*.xaml", SearchOption.AllDirectories))
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}Theme{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                        && !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                        && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal));

    private static void RedesignNoLiterals()
    {
        // No colour anywhere in a page but through a theme key, and no size of text or space but through one.
        var colour = new Regex("""\b(TextColor|BackgroundColor|BorderColor|Stroke|Fill|Color|Background|PlaceholderColor|ThumbColor|OnColor|MinimumTrackColor|MaximumTrackColor)="(?!\{)""");
        int files = 0;
        foreach (string file in Xaml())
        {
            files++;
            ThemeKeysOnly(Path.GetRelativePath(RepositoryRoot(), file));
            string xaml = File.ReadAllText(file);
            var found = colour.Matches(Regex.Replace(xaml, "<Setter[^>]*>", "")).Select(m => m.Value).ToList();
            Check(found.Count == 0, $"{Path.GetFileName(file)} sets a colour without a theme key: {string.Join(", ", found)}");
            Check(!Regex.IsMatch(Regex.Replace(xaml, "<!--.*?-->", "", RegexOptions.Singleline), "#[0-9A-Fa-f]{6}"), $"{Path.GetFileName(file)} names a colour");
        }

        Check(files >= 16, $"{files} XAML files scanned");
    }

    private static void RedesignSourceCommit()
    {
        var source = SourceInfo.Library;
        var attributes = typeof(SourceInfo).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), false)
            .Cast<System.Reflection.AssemblyMetadataAttribute>().Select(a => a.Key).ToHashSet();
        Check(attributes.Contains(SourceInfo.CommitKey) && attributes.Contains(SourceInfo.BranchKey), $"metadata: {string.Join(", ", attributes)}");
        Check(source.Commit == "unknown" || Regex.IsMatch(source.Commit, "^[0-9a-f]{7,40}$"), $"commit '{source.Commit}'");
        Check(source.IsKnown == (source.Commit != "unknown") && (source.Branch.Length == 0 ? source.Text == source.Commit : source.Text == $"{source.Branch} · {source.Commit}"), source.Text);
        Check(new SourceInfo("main", "628b6bb").Text == "main · 628b6bb" && new SourceInfo("", "628b6bb").Text == "628b6bb", "the text");
        Check(SourceInfo.Of(typeof(object).Assembly) is { Commit: "unknown", Branch: "" }, "an assembly without it");
        Console.WriteLine($"    built from {source.Text}");
    }

    private static void RedesignIcons()
    {
        var names = IdrakIcons.Names;
        foreach (string name in (string[])["home", "chat", "mic", "vision", "ocr", "grid", "gear", "cpu", "zap", "db", "speak", "wave", "image", "search", "shield", "down",
                     "arrow", "spark", "copy", "share", "redo", "pen", "plus", "up", "chev", "brain", "filter", "file", "sliders", "clock"])
        {
            Check(names.Contains(name), $"the design's icon '{name}' is missing");
        }

        foreach (string name in names)
        {
            var geometry = IdrakIcons.Get(name) as PathGeometry;
            Check(geometry is { Figures.Count: > 0 } && geometry.Figures.All(f => f.Segments.Count > 0), $"icon '{name}' did not parse into figures");
            var bounds = geometry!.Figures.SelectMany(f => f.Segments.OfType<LineSegment>().Select(s => s.Point).Prepend(f.StartPoint)).ToList();
            Check(bounds.All(p => p.X is >= -0.5 and <= 24.5 && p.Y is >= -0.5 and <= 24.5), $"icon '{name}' leaves the 24-unit view box");
        }

        Check(IdrakIcons.Get("nope") is null && IdrakIcons.Get("") is null && IdrakIcons.Key("home") == "IdrakIconHome", "unknown icons");
        var icon = new IconView { Glyph = "mic", Size = 30 };
        Check(icon.Geometry is PathGeometry && icon.WidthRequest == 30 && icon.ViewBoxScale is ScaleTransform { ScaleX: 1.25 }, "an icon view at 30 units");
        var badge = new IconBadge { Glyph = "chat", Tone = "Violet" };
        var page = Themed(badge, AppTheme.Dark);
        Check(badge.BackgroundColor.Equals(IdrakThemes.Colors(AppTheme.Dark)["IdrakToneVioletBackground"]) && page is not null, "a badge takes its tone's colours");
    }

    private static void RedesignHome()
    {
        var options = new IdrakMauiOptions { CarouselInterval = TimeSpan.FromSeconds(4) };
        options.Home.Slides = [.. HomeOptions.LibrarySlides.Take(2), new HomeSlide("Ours", "An app's ", "own {device} slide.", "Text on this {device}.", "Open", "devices", "Mine", "Violet", "spark")];
        using var services = AppServices(options);
        var status = services.GetRequiredService<DeviceStatus>();
        var home = new HomeViewModel(options, status, services.GetRequiredService<ChatViewModel>(), services.GetRequiredService<IAppNavigator>(), new SourceInfo("main", "628b6bb"));
        Check(home.SourceText == "main · 628b6bb" && home.Slides.Count == 3 && home.Slides[0].IsLogo && home.Slides[1].IsIcon && home.Slides[2].Emphasis == "own device slide."
              && home.Slides[0].Emphasis == "on this device.", $"{home.SourceText}; {string.Join(" | ", home.Slides.Select(s => s.Title + s.Emphasis))}");
        Check(home.SlideCaption == "1 of 3 · The library" && home.SlideInterval == TimeSpan.FromSeconds(4), home.SlideCaption);
        home.NextSlide();
        home.NextSlide();
        Check(home.Position == 2 && home.SlideCaption == "3 of 3 · Mine", home.SlideCaption);
        home.NextSlide();
        Check(home.Position == 0, "after the last, the first");
        Check(new HomeViewModel(new IdrakMauiOptions { CarouselInterval = null }, status).SlideInterval == TimeSpan.Zero, "no interval: it stays");

        // The tiles from a reading.
        home.RefreshAsync().GetAwaiter().GetResult();
        var summary = status.Summary!;
        Check(summary is { CpuTitle: "8 cores", CpuDetail: "NEON · DotProd", GpuTitle: "Vulkan", GpuDetail: "kernels 64 wide", GpuReady: true, Chip: "GPU ready" }
              && summary.MemoryTotal == ProgressText.Bytes(8L << 30) && Math.Abs(summary.MemoryUsed - 0.25) < 1e-9 && status.Models.Count == 2 && status.ModelsText == $"2 pulled · {ProgressText.Bytes(885_000_000)}",
            $"{summary}; {status.ModelsText}");
        var cpuOnly = DeviceStatus.Summarize(new DeviceReport([], [new BackendInfo("vulkan", "Vulkan", 0, "no Vulkan driver")],
            new ProcessFacts("os", "x64", ".NET", 4, 1L << 30, 1L << 29, 0, "/c", 4), "cpu", TimeSpan.Zero));
        Check(cpuOnly is { GpuTitle: "CPU only", GpuDetail: "no Vulkan driver", GpuReady: false, Chip: "CPU only", CpuTitle: "4 cores", CpuDetail: "4 threads" }, cpuOnly.ToString());
        Check(services.GetRequiredService<IDeviceInformation>() is ListedDevices { Reads: 1 }, "the device is read once");

        // The chat tile shows the last exchange and its speed once the chat has run.
        var chat = services.GetRequiredService<ChatViewModel>();
        chat.RefreshAsync().GetAwaiter().GetResult();
        chat.SelectedModel = chat.Models[0];
        chat.Loading!.GetAwaiter().GetResult();
        chat.Draft = "What runs here?";
        chat.SendAsync().GetAwaiter().GetResult();
        home.RefreshAsync().GetAwaiter().GetResult();
        Check(home.ChatQuestion == "What runs here?" && home.ChatAnswer == "ababab" && home.HasChatRate && home.ChatRate.StartsWith("3.0 tok/s on cpu", StringComparison.Ordinal)
              && home.ChatModel == "Qwen3-0.6B · float32", $"'{home.ChatQuestion}' '{home.ChatAnswer}' '{home.ChatRate}' '{home.ChatModel}'");

        // The layout settings are kept and read back.
        var settings = new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore());
        var saved = new IdrakMauiOptions { WideLayoutWidth = 700, CarouselInterval = null };
        settings.SaveAsync(saved).GetAwaiter().GetResult();
        var read = new IdrakMauiOptions();
        settings.LoadIntoAsync(read).GetAwaiter().GetResult();
        Check(read.WideLayoutWidth == 700 && read.CarouselInterval is null, $"read {read.WideLayoutWidth}, {read.CarouselInterval}");
        var setting = IdrakMauiSettings.Definitions.Single(s => s.Key == "wide_layout_dp");
        bool refused = false;
        try
        {
            setting.Set(read, "100");
        }
        catch (FormatException)
        {
            refused = true;
        }

        Check(refused, "a breakpoint under 320 was taken");
    }

    private static void RedesignSettingsAndDevices()
    {
        string storage = Temp("settings-groups");
        try
        {
            var options = new IdrakMauiOptions();
            var startup = IdrakMauiStartup.Apply(options, storage);
            var page = new SettingsViewModel(options, new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore()), startup);
            Check(page.Groups.Select(g => g.Title).SequenceEqual(SettingsViewModel.GroupOrder) && page.Groups.Sum(g => g.Count) == IdrakMauiSettings.Definitions.Count,
                string.Join(", ", page.Groups.Select(g => $"{g.Title} {g.Count}")));
            var theme = page.Entries.Single(e => e.Setting.Key == "theme");
            Check(theme is { IsSegmented: true, IsPicker: false, IsField: false } && theme.Segments.Select(s => s.Text).SequenceEqual(["system", "light", "dark"])
                  && theme.Segments[0].IsSelected, "the theme as segments");
            theme.Segments[2].SelectCommand.Execute(null);
            Check(theme.Text == "dark" && theme.Segments[2].IsSelected && !theme.Segments[0].IsSelected, $"chose '{theme.Text}'");
            var vectors = page.Entries.Single(e => e.Setting.Key == "cpu_vectors");
            Check(vectors is { IsSegmented: true, Group: "General", Text.Length: 0 } && vectors.Segments.Select(s => s.Label).SequenceEqual(["Automatic", "Native", "Portable", "Scalar"])
                  && vectors.Segments[0].IsSelected && theme.Segments[0].Label == "System", $"the vector paths as segments: {string.Join(", ", vectors.Segments.Select(s => s.Label))}");
            vectors.Segments[2].SelectCommand.Execute(null);
            Check(vectors.Text == "portable" && vectors.Segments[2].IsSelected, $"chose '{vectors.Text}'");
            vectors.Segments[0].SelectCommand.Execute(null);
            Check(vectors.Text.Length == 0, "automatic is the library's own");
            var weights = page.Entries.Single(e => e.Setting.Key == "chat_weights");
            Check(weights.IsSegmented == weights.Choices.Count <= 4 && weights.IsPicker == !weights.IsSegmented && page.Entries.Single(e => e.Setting.Key == "device") is { IsPicker: true, IsSegmented: false }
              && page.Entries.Single(e => e.Setting.Key == "lookup_tables") is { IsSegmented: true, Segments.Count: 4 }, $"weights ({weights.Choices.Count} choices) and the device");
            var temperature = page.Entries.Single(e => e.Setting.Key == "chat_temperature");
            Check(temperature is { IsSlider: true, IsField: false, Minimum: 0, Maximum: 2 } && temperature.NumberText.StartsWith("default", StringComparison.Ordinal), temperature.NumberText);
            temperature.Number = 0.737;
            Check(temperature.Text == "0.75" && temperature.NumberText == "0.75", $"rounded to the step: '{temperature.Text}'");
            temperature.ResetCommand.Execute(null);
            Check(temperature.Text.Length == 0, "reset to the library's");
            Check(page.Entries.Single(e => e.Setting.Key == "hf_token") is { IsField: true, IsPassword: true } && page.Entries.Single(e => e.Setting.Key == "cpu_threads").IsField,
                "the token and a number without a range as fields");
            page.Query = "packed weights";
            var packedFound = page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key).ToList();
            Check(packedFound.Contains("packed_cache") && packedFound.Contains("packed_cache_folder") && page.Entries.Single(e => e.Setting.Key == "packed_cache") is { IsSegmented: true, Group: "Storage" },
                $"the packed-weight cache's settings found by the search, in Storage: {string.Join(", ", packedFound)}");
            page.Query = "";
        }
        finally
        {
            Restore(storage);
        }

        // Devices: a summary card each, its state and key facts, every fact folded; virtual devices apart.
        var cards = DevicesViewModel.Summarize(new ListedDevices().Read());
        Check(cards.Select(c => c.Title).SequenceEqual(["This phone", "CPU", "Vulkan", "Vulkan", "deferred", "CUDA"]), string.Join(", ", cards.Select(c => c.Title)));
        var gpu = cards[2];
        Check(gpu is { IsReady: true, IsDefault: true, Badge: "Default", IsVirtual: false, Device: "vulkan:0" } && gpu.Facts.Select(f => f.Name).SequenceEqual(["memory", "kernel width", "lanes", "driver"])
              && gpu.Details.Count > gpu.Facts.Count && !gpu.IsExpanded, $"gpu: {string.Join(", ", gpu.Facts.Select(f => f.Name))}");
        gpu.ToggleCommand.Execute(null);
        Check(gpu.IsExpanded && gpu.ToggleText == "Less", gpu.ToggleText);
        Check(cards[3] is { IsFailed: true, Badge: "Failed", HasProblem: true } && cards[4] is { IsVirtual: true } && cards[5] is { IsNone: true, Badge: "None" }
              && cards[1].Facts.Any(f => f.Name == "instruction sets"), "failed, virtual and none");
        var vm = new DevicesViewModel(new ListedDevices());
        vm.RefreshAsync().GetAwaiter().GetResult();
        Check(vm.Devices.Count == 5 && vm.VirtualDevices.Single().Device == "deferred:0" && vm.HasVirtualDevices, $"{vm.Devices.Count} devices, {vm.VirtualDevices.Count} virtual");
    }

    private static void RedesignLabParts()
    {
        // Studio: the pipeline as steps, each taking itself out.
        using var lab = new ProcessLab();
        var studio = lab.Vm;
        studio.PipelineText = "highpass=80, denoise, agc";
        Check(studio.Steps.Select(s => s.Name).SequenceEqual(["highpass", "denoise", "agc"]) && studio.Steps[0] is { Value: "80", HasArrow: false } && studio.Steps[1].HasArrow,
            string.Join(" › ", studio.Steps.Select(s => s.Name)));
        studio.Steps[1].RemoveCommand.Execute(null);
        Check(studio.PipelineText == "highpass=80, agc" && studio.Steps.Count == 2, studio.PipelineText);

        // Speak: the measures' words before a speech.
        using var speak = new SpeakLab(new ScriptedSynthesizer());
        Check(speak.Vm.FirstSoundText == "–" && speak.Vm.RealTimeText == "–" && speak.Vm.Pieces.Count == 0 && !speak.Vm.HasMeasures && speak.Vm.ModelChip == "voice",
            $"nothing measured yet; chip '{speak.Vm.ModelChip}'");

        // The caret, the record button and the waveform: nothing animates off screen (no handler here), the waveform keeps its last levels.
        var caret = new CaretView { IsBlinking = true };
        var record = new RecordButton { IsRecording = true };
        Check(caret.IsVisible && !caret.AnimationIsRunning("IdrakCaret") && !record.AnimationIsRunning("IdrakPulse"), "no animation off screen");
        var wave = new WaveformView(8) { IsLive = true };
        foreach (double level in (double[])[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5])
        {
            wave.Level = level;
        }

        Check(wave.Count == 8 && wave.Levels().SequenceEqual([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), string.Join(", ", wave.Levels()));
        wave.IsLive = false;
        wave.Level = 0.05;
        Check(wave.Count == 8, "not live: a level is not pushed");
    }

    // A phone's CPU as phase F lists it (Mono: the runtime's figure is its heap budget, the system's figure the phone's
    // memory), and a Vulkan device whose driver refused two kernels.
    private static DeviceReport PhoneReading(params KeyValuePair<string, string>[] more)
    {
        KeyValuePair<string, string>[] facts =
        [
            new("processor", "Test SoC (8 cores: 4 × part 0x41:0xd05 up to 2.00 GHz, 3 × part 0x41:0xd0d up to 2.75 GHz, 1 × part 0x41:0xd0e up to 3.20 GHz)"),
            new("core groups", "8 cores: 4 × part 0x41:0xd05 up to 2.00 GHz, 3 × part 0x41:0xd0d up to 2.75 GHz, 1 × part 0x41:0xd0e up to 3.20 GHz"),
            new("threads", "8 threads, 8 cores"),
            new("instruction sets", "NEON (portable vectors: the runtime reports no NEON intrinsics), FMA"),
            new("vector paths", "automatic (IDRAK_CPU_VECTORS); tiled products: 8×8 in 128-bit vectors (portable vectors: the runtime reports no NEON intrinsics); few-row products: register-tiled; int4 lane shifts: multiplies by powers of two (the same bits)"),
            new("memory", "11.0 GiB the process may use, 4.2 GiB of it free now (as the system reports them)"),
            new("system memory", "11.0 GiB, 4.2 GiB available (/proc/meminfo)"),
            new("runtime memory", "143.0 MiB: Mono's heap budget now, not a limit (it grows as the heap does)"),
            .. more,
        ];
        var cpu = new DeviceInfo { Device = "cpu", Kind = "cpu", Backend = "CPU", Name = "Test SoC", Listed = true, MemoryBytes = 11L << 30, ComputeUnits = 8, SubgroupSize = 4, Facts = facts };
        var gpu = new DeviceInfo
        {
            Device = "vulkan:0", Kind = "vulkan", Backend = "Vulkan", Name = "Test GPU", Listed = true, IsDefault = true, KernelWidth = 64, HardwareKind = "integrated",
            Facts =
            [
                new("refused kernels", "int8_gemv at width 64 (vkCreateComputePipelines with subgroups of 64: VK_ERROR_UNKNOWN; VK_ERROR_UNKNOWN); instead: Int8MatMul: the host; "
                    + "softmax at width 128 (VK_ERROR_UNKNOWN); instead: nothing has needed it since"),
                new("subgroup-size requests dropped", "none"),
            ],
        };
        var process = new ProcessFacts("Android", "Arm64", ".NET 10 (Mono)", 8, 150L << 20, 400L << 20, 50 << 20, "/cache", 8);
        return new DeviceReport([new(cpu, null, 0), new(gpu, null, 0)], [], process, "vulkan:0", TimeSpan.Zero);
    }

    private static void RedesignPhoneFacts()
    {
        const long GiB = 1L << 30;
        long total = 11 * GiB, free = (long)Math.Round(4.2 * GiB);
        var report = PhoneReading();
        var cpuFacts = report.Devices[0].Info.Facts;

        // The memory from the system's figure, held to a memory limit; the runtime's figure where the system reports none.
        Check(DeviceFacts.SystemMemory(cpuFacts) is { } memory && memory.Total == total && memory.Free == free && Math.Abs(memory.Used - (double)(total - free) / total) < 1e-9
              && memory.Text == $"{ProgressText.Bytes(free)} free of {ProgressText.Bytes(total)}", $"{DeviceFacts.SystemMemory(cpuFacts)}");
        var limited = DeviceFacts.SystemMemory(PhoneReading(new KeyValuePair<string, string>("memory limit", "6.0 GiB, 5.0 GiB in use (cgroup v2 /sys/fs/cgroup)")).Devices[0].Info.Facts);
        Check(limited is { Total: 6 * GiB, Free: 1 * GiB }, $"held to the limit: {limited}");
        var runtime = DeviceFacts.SystemMemory([new("memory", "16.0 GiB the process may use, 9.5 GiB of it free now (as the runtime reports them: the system reports none here)")]);
        Check(runtime is { Total: 16 * GiB } r && r.Free == (long)(9.5 * GiB) && DeviceFacts.SystemMemory([]) is null && DeviceFacts.SystemMemory(null) is null, $"the runtime's figures: {runtime}");

        // The core groups and the vector paths in a line; the refused kernels one each.
        string groups = DeviceFacts.CoreGroups(DeviceFacts.Fact(cpuFacts, "core groups"));
        Check(groups == "4 × 2.0 GHz · 3 × 2.75 GHz · 1 × 3.2 GHz" && DeviceFacts.CoreGroupCount(DeviceFacts.Fact(cpuFacts, "core groups")) == 3, groups);
        Check(DeviceFacts.CoreGroups("4 cores: 4 × part 0x41:0xd03") == "4 × 0x41:0xd03" && DeviceFacts.CoreGroups(null).Length == 0, DeviceFacts.CoreGroups("4 cores: 4 × part 0x41:0xd03"));
        string vectors = DeviceFacts.VectorPaths(DeviceFacts.Fact(cpuFacts, "vector paths"));
        Check(vectors == "automatic · 8×8 in 128-bit vectors (portable vectors)" && DeviceFacts.VectorPaths("something else") == "something else", vectors);
        var refused = DeviceFacts.RefusedKernels(DeviceFacts.Fact(report.Devices[1].Info.Facts, "refused kernels"));
        Check(refused.Count == 2 && refused[0].StartsWith("int8_gemv at width 64 (", StringComparison.Ordinal) && refused[0].EndsWith("; instead: Int8MatMul: the host", StringComparison.Ordinal)
              && refused[1].StartsWith("softmax at width 128", StringComparison.Ordinal) && DeviceFacts.RefusedKernels("none").Count == 0 && DeviceFacts.RefusedKernels(null).Count == 0,
            string.Join(" | ", refused));

        // Home's tiles: the memory and the core groups.
        var summary = DeviceStatus.Summarize(report);
        Check(summary.MemoryFree == ProgressText.Bytes(free) && summary.MemoryTotal == ProgressText.Bytes(total) && Math.Abs(summary.MemoryUsed - (double)(total - free) / total) < 1e-9
              && summary.CpuDetail == "3 core groups" && summary.GpuReady, summary.ToString());

        // The Devices cards: the CPU's summary, the runtime's figure labelled in the details; the GPU's badge, its list folded.
        var cards = DevicesViewModel.Summarize(report);
        var cpu = cards.Single(c => c.Title == "CPU");
        Check(cpu.Facts.Select(f => f.Name).SequenceEqual(["memory", "core groups", "vector paths", "instruction sets"]) && cpu.Facts[0].Value == $"{ProgressText.Bytes(free)} free of {ProgressText.Bytes(total)}"
              && cpu.Facts[1].Value == groups && cpu.Facts[2].Value == vectors && !cpu.HasRefusedKernels, string.Join("; ", cpu.Facts.Select(f => $"{f.Name}: {f.Value}")));
        Check(cpu.Details.Any(d => d.Name == "the runtime's memory figure" && d.Value.StartsWith("143.0 MiB: Mono's heap budget", StringComparison.Ordinal))
              && cpu.Details.All(d => d.Name != "runtime memory") && cpu.Details.Any(d => d.Name == "system memory"), string.Join(", ", cpu.Details.Select(d => d.Name)));
        var gpu = cards.Single(c => c.Title == "Vulkan");
        Check(gpu is { HasRefusedKernels: true, RefusedText: "2 kernels on fallback", IsRefusedExpanded: false } && gpu.RefusedKernels.SequenceEqual(refused), gpu.RefusedText);
        gpu.ToggleRefusedCommand.Execute(null);
        Check(gpu.IsRefusedExpanded, "the refused kernels unfold");
        cpu.IsRefusedExpanded = true;
        Check(!cpu.IsRefusedExpanded, "nothing to unfold without a refused kernel");
        var one = DevicesViewModel.Summarize(new DeviceReport([new(new DeviceInfo
        {
            Device = "vulkan:0", Kind = "vulkan", Backend = "Vulkan", Name = "g", Listed = true, Facts = [new("refused kernels", "rms_norm at width 64 (VK_ERROR_UNKNOWN); instead: the host")],
        }, null, 0)], [], report.Process, "cpu", TimeSpan.Zero));
        Check(one.Single(c => c.Title == "Vulkan").RefusedText == "1 kernel on fallback", "one kernel");

        // The Devices view binds the badge in both themes, its labels in the theme's colours.
        EnsureDispatcher();
        var vm = new DevicesViewModel(new PhoneDevices(report));
        vm.RefreshAsync().GetAwaiter().GetResult();
        foreach (var theme in (AppTheme[])[AppTheme.Light, AppTheme.Dark])
        {
            var page = Themed(new DevicesView(vm), theme);
            Check(CheckLabels(page, theme, "devices") > 0 && Tree(page).OfType<Label>().Any(l => l.Text == "2 kernels on fallback"), $"the devices view in {theme}");
        }
    }

    private sealed class PhoneDevices(DeviceReport report) : IDeviceInformation
    {
        public DeviceReport Read(IReadOnlyList<string>? devices = null) => report;
    }

    private static void RedesignIndicator()
    {
        // The design's dots: 6 units, 6 apart, the page shown a pill of 22 (24 on a tablet) in the accent.
        var dots = new PageIndicator { Count = 4 };
        Check(dots.WidthRequest == 3 * 12 + 22 && dots.HeightRequest == 6 && dots.Shown == 0, $"{dots.WidthRequest} x {dots.HeightRequest}");
        Check(dots.WidthOf(0, 0) == 22 && dots.WidthOf(1, 0) == 6 && dots.WidthOf(3, 0) == 6, "the first page's pill");
        dots.Position = 2;                                                       // off screen: there at once
        Check(dots.Shown == 2 && dots.WidthOf(2, dots.Shown) == 22 && dots.WidthOf(0, dots.Shown) == 6 && dots.WidthOf(1, 1.5) == 14 && dots.WidthOf(2, 1.5) == 14,
            $"at {dots.Shown}; halfway {dots.WidthOf(1, 1.5)}");
        dots.ActiveWidth = 24;
        Check(dots.WidthRequest == 3 * 12 + 24, $"a tablet's pill: {dots.WidthRequest}");
        dots.Count = 0;
        Check(dots.WidthRequest == 0, "no pages, no dots");

        // Bound to Home's carousel: the slides' count and its place.
        var source = new IndicatorSource();
        var bound = new PageIndicator { BindingContext = source };
        bound.SetBinding(PageIndicator.CountProperty, new Binding(nameof(IndicatorSource.Count)));
        bound.SetBinding(PageIndicator.PositionProperty, new Binding(nameof(IndicatorSource.Position)));
        source.Position = 2;
        Check(bound.Count == 3 && bound.Position == 2 && bound.Shown == 2, $"bound: {bound.Count} pages at {bound.Position}");
    }

    private sealed class IndicatorSource : ViewModel
    {
        private int _position;

        public int Count => 3;

        public int Position
        {
            get => _position;
            set => SetProperty(ref _position, value);
        }
    }
}
