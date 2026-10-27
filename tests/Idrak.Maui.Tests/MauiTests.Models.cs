// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Tests;

// The Models lab: the model store's pull through a Hugging Face served in the process (no network), and its page's view model.
internal static partial class MauiTests
{
    // A Hugging Face in the process: a repository's revision, its file tree and its files (ranges honoured), a gated
    // repository answering 401 without a token and 403 with another one, 404 for what it does not have. Every request is
    // recorded, and moves the test's clock by a second.
    private sealed class FakeHub : HttpMessageHandler
    {
        public const string Commit = "c0ffee0123456789abcdef0123456789abcdef01";

        private readonly Dictionary<string, Dictionary<string, byte[]>> _repos = new(StringComparer.Ordinal);

        public sealed record Seen(string Path, string? Authorization, string? Range);

        public ConcurrentQueue<Seen> Requests { get; } = new();

        public HashSet<string> Gated { get; } = new(StringComparer.Ordinal);

        public string? AcceptedToken { get; set; }

        public Exception? Unreachable { get; set; }

        public ManualTime? Time { get; set; }

        // Called with a file's path when its download is asked for (before its first byte).
        public Action<string>? OnFile { get; set; }

        // Runs once while a file is served, when this many of its bytes are out.
        public (string File, long At, Action Run)? Interrupt { get; set; }

        public static string ShortCommit => Commit[..12];

        public FakeHub Add(string repo, string path, byte[] bytes)
        {
            if (!_repos.TryGetValue(repo, out var files))
            {
                _repos[repo] = files = new(StringComparer.Ordinal);
            }

            files[path] = bytes;
            return this;
        }

        // The downloads of a file, of one repository when it is named.
        public IEnumerable<Seen> Downloads(string file, string? repo = null) => Requests.Where(r => r.Path.Contains("/resolve/", StringComparison.Ordinal)
            && r.Path.EndsWith("/" + file, StringComparison.Ordinal) && (repo is null || r.Path.StartsWith($"/{repo}/", StringComparison.Ordinal)));

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            string path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);
            string? auth = request.Headers.Authorization?.ToString();
            Requests.Enqueue(new Seen(path, auth, request.Headers.Range?.ToString()));
            Time?.Advance(TimeSpan.FromSeconds(1));
            if (Unreachable is { } unreachable)
            {
                throw unreachable;
            }

            string[] p = path.Trim('/').Split('/');
            bool api = p[0] == "api";
            string repo = api ? $"{p[2]}/{p[3]}" : $"{p[0]}/{p[1]}";
            if (!_repos.TryGetValue(repo, out var files))
            {
                return Answer(HttpStatusCode.NotFound, "Repository not found");
            }

            if (Gated.Contains(repo) && (auth is null || auth != "Bearer " + AcceptedToken))
            {
                return Answer(auth is null ? HttpStatusCode.Unauthorized : HttpStatusCode.Forbidden, "Access to this model is restricted");
            }

            if (api && p[4] == "revision")
            {
                return Answer(HttpStatusCode.OK, $"{{\"sha\":\"{Commit}\"}}");
            }

            if (api && p[4] == "tree")
            {
                var tree = new JsonArray();
                foreach (var (name, bytes) in files)
                {
                    tree.Add(new JsonObject { ["type"] = "file", ["path"] = name, ["size"] = bytes.Length });
                }

                return Answer(HttpStatusCode.OK, tree.ToJsonString());
            }

            string file = string.Join('/', p[4..]);
            if (p[2] != "resolve" || !files.TryGetValue(file, out var content))
            {
                return Answer(HttpStatusCode.NotFound, "Entry not found");
            }

            OnFile?.Invoke(file);
            cancellationToken.ThrowIfCancellationRequested();
            long from = request.Headers.Range?.Ranges.First().From ?? 0;
            bool here = Interrupt is { } i && i.File == file;
            var stream = new ServedStream(content, (int)from, here ? Interrupt!.Value.At : long.MaxValue, here ? Interrupt!.Value.Run : null);
            return Task.FromResult(new HttpResponseMessage(from > 0 ? HttpStatusCode.PartialContent : HttpStatusCode.OK) { Content = new StreamContent(stream) });
        }

        private static Task<HttpResponseMessage> Answer(HttpStatusCode status, string body) =>
            Task.FromResult(new HttpResponseMessage(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") });
    }

    // A file served in pieces of 16 KiB from where a range starts, honouring cancellation between pieces.
    // (Its own copy, not MemoryStream's: a derived MemoryStream's span read falls back to the array read it overrides.)
    private sealed class ServedStream : MemoryStream
    {
        private readonly byte[] _bytes;
        private readonly long _interruptAt;
        private Action? _interrupt;

        public ServedStream(byte[] bytes, int from, long interruptAt, Action? interrupt)
            : base(bytes, writable: false)
        {
            _bytes = bytes;
            Position = from;
            _interruptAt = interruptAt;
            _interrupt = interrupt;
        }

        public override int Read(byte[] buffer, int offset, int count) => Serve(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer) => Serve(buffer);

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return new(Serve(buffer.Span));
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(Serve(buffer.AsSpan(offset, count)));
        }

        private int Serve(Span<byte> buffer)
        {
            if (_interrupt is { } run && Position >= _interruptAt)
            {
                _interrupt = null;
                run();
            }

            int count = (int)Math.Min(Math.Min(buffer.Length, 16 << 10), _bytes.Length - Position);
            _bytes.AsSpan((int)Position, count).CopyTo(buffer);
            Position += count;
            return count;
        }
    }

    private static byte[] Bytes(int count, int seed)
    {
        var bytes = new byte[count];
        new Random(seed).NextBytes(bytes);
        return bytes;
    }

    // A model the library reads (its config, tokenizer and safetensors weights), with files it does not take beside them.
    private static FakeHub TinyModel(FakeHub hub, string repo, int weights = 200_000) => hub
        .Add(repo, "config.json", Bytes(100, 1))
        .Add(repo, "tokenizer.json", Bytes(300, 2))
        .Add(repo, "model.safetensors", Bytes(weights, 3))
        .Add(repo, "README.md", Bytes(50, 4))
        .Add(repo, "onnx/model.onnx", Bytes(70, 5));

    // Runs a test with its own cache folder and no token from the process (HF_TOKEN, a saved login), then puts both back.
    private static void InStore(string name, Action<string> test)
    {
        string cache = Temp(name);
        string?[] saved = [.. TokenVariables.Select(Environment.GetEnvironmentVariable)];
        try
        {
            ComputeResources.CacheFolder = cache;
            Environment.SetEnvironmentVariable("HF_TOKEN", null);
            Environment.SetEnvironmentVariable("HUGGING_FACE_HUB_TOKEN", null);
            Environment.SetEnvironmentVariable("HF_TOKEN_PATH", Path.Combine(cache, "no-token-here"));
            test(cache);
        }
        finally
        {
            for (int i = 0; i < TokenVariables.Length; i++)
            {
                Environment.SetEnvironmentVariable(TokenVariables[i], saved[i]);
            }

            ComputeResources.CacheFolder = null;
        }
    }

    private static readonly string[] TokenVariables = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"];

    private static string RevisionFolder(ModelStore store, string repo) => Path.Combine(store.Folder, repo, FakeHub.ShortCommit);

    private static T Throws<T>(Func<Task> run, string what) where T : Exception
    {
        try
        {
            run().GetAwaiter().GetResult();
        }
        catch (T e)
        {
            return e;
        }

        throw new CheckException($"{what}: no {typeof(T).Name}");
    }

    private static void ModelStorePulled()
    {
        InStore("pull", _ =>
        {
            // The token kept in the secret store reaches the options as UseIdrak loads them, and goes with every request.
            var hub = TinyModel(new FakeHub(), "lab/gated");
            hub.Gated.Add("lab/gated");
            hub.AcceptedToken = "hf_good";
            var secrets = new MemorySecretStore();
            var settings = new IdrakMauiSettings(new MemorySettingsStore(), secrets);
            settings.SaveAsync(new IdrakMauiOptions { HuggingFaceToken = "hf_good" }).GetAwaiter().GetResult();
            var options = new IdrakMauiOptions();
            settings.LoadIntoAsync(options).GetAwaiter().GetResult();
            var store = new ModelStore(options, new HttpClient(hub));
            Check(store.HasToken, "the token from the secret store");
            var stored = store.PullAsync("lab/gated").GetAwaiter().GetResult();
            Check(hub.Requests.All(r => r.Authorization == "Bearer hf_good"), $"requests without the token: {string.Join(", ", hub.Requests.Where(r => r.Authorization != "Bearer hf_good").Select(r => r.Path))}");
            var taken = hub.Requests.Where(r => r.Path.Contains("/resolve/", StringComparison.Ordinal)).Select(r => r.Path.Split('/')[^1]).Order(StringComparer.Ordinal).ToList();
            Check(taken.SequenceEqual(["config.json", "model.safetensors", "tokenizer.json"]), $"files taken: {string.Join(", ", taken)}");
            Check(stored.Id == "lab/gated" && stored.Revision == FakeHub.ShortCommit && stored.Files == 3 && stored.Bytes == 200_400 && stored.Complete
                  && !File.Exists(Path.Combine(stored.Folder, ModelStore.PullMarker)), $"stored: {stored}");
            Check(File.ReadAllBytes(Path.Combine(stored.Folder, "model.safetensors")).AsSpan().SequenceEqual(Bytes(200_000, 3)), "the weights' bytes");

            // Pulled again: nothing is downloaded (every file is in).
            int before = hub.Requests.Count(r => r.Path.Contains("/resolve/", StringComparison.Ordinal));
            store.PullAsync("lab/gated").GetAwaiter().GetResult();
            Check(hub.Requests.Count(r => r.Path.Contains("/resolve/", StringComparison.Ordinal)) == before, "a pulled model downloaded again");

            // Without the token: 401; with another one: 403.
            var none = Throws<HttpRequestException>(() => new ModelStore(new IdrakMauiOptions(), new HttpClient(hub)).PullAsync("lab/gated"), "no token");
            var other = Throws<HttpRequestException>(() => new ModelStore(new IdrakMauiOptions { HuggingFaceToken = "hf_other" }, new HttpClient(hub)).PullAsync("lab/gated"), "another token");
            Check(none.StatusCode == HttpStatusCode.Unauthorized && other.StatusCode == HttpStatusCode.Forbidden, $"{none.StatusCode}, {other.StatusCode}");

            // One .gguf file of a repository, at a revision.
            hub.Add("lab/gguf", "a.gguf", Bytes(1000, 6)).Add("lab/gguf", "sub/b.gguf", Bytes(2000, 7));
            var gguf = new ModelStore(new IdrakMauiOptions(), new HttpClient(hub)).PullAsync("lab/gguf/sub/b.gguf", "v1").GetAwaiter().GetResult();
            Check(gguf.Files == 1 && gguf.Bytes == 2000 && File.Exists(Path.Combine(gguf.Folder, "sub", "b.gguf")) && !hub.Downloads("a.gguf").Any(), $"gguf: {gguf}");
            Check(hub.Requests.Any(r => r.Path == "/api/models/lab/gguf/revision/v1"), "the revision asked for");
            Throws<ArgumentException>(() => store.PullAsync("lab/gguf", "a b"), "a revision with a space");

            // A partial file resumes where it stopped, with a range, to the same bytes.
            TinyModel(hub, "lab/resume");
            var plain = new ModelStore(new IdrakMauiOptions(), new HttpClient(hub));
            string folder = RevisionFolder(plain, "lab/resume");
            Directory.CreateDirectory(folder);
            File.WriteAllBytes(Path.Combine(folder, "model.safetensors.part"), Bytes(200_000, 3)[..50_000]);
            var resumed = plain.PullAsync("lab/resume").GetAwaiter().GetResult();
            Check(hub.Downloads("model.safetensors", "lab/resume").Single().Range == "bytes=50000-", $"range: {hub.Downloads("model.safetensors", "lab/resume").Single().Range}");
            Check(resumed.Complete && resumed.Bytes == 200_400 && File.ReadAllBytes(Path.Combine(folder, "model.safetensors")).AsSpan().SequenceEqual(Bytes(200_000, 3)),
                "the resumed file's bytes");
            Check(plain.List().Select(m => m.Id).Order(StringComparer.Ordinal).SequenceEqual(["lab/gated", "lab/gguf", "lab/resume"]), "listed");
        });
    }

    private static void ModelStoreCancelled()
    {
        InStore("cancel", _ =>
        {
            var hub = TinyModel(new FakeHub(), "lab/cancel");
            var store = new ModelStore(new IdrakMauiOptions(), new HttpClient(hub));
            var page = new ModelsViewModel(store, new IdrakMauiOptions(), new ProgressCenter(TimeSpan.FromMilliseconds(250))) { ModelId = "lab/cancel" };
            string folder = RevisionFolder(store, "lab/cancel");

            // Cancelled between two files: the two small ones are in, the weights not started; only the marker says so.
            hub.OnFile = file =>
            {
                if (file == "model.safetensors")
                {
                    page.Cancel();
                }
            };
            Check(!page.PullAsync().GetAwaiter().GetResult(), "a cancelled pull succeeded");
            Check(page.Status.StartsWith("Cancelled", StringComparison.Ordinal) && !page.HasError, $"status '{page.Status}', error '{page.Error}'");
            var first = page.Models.Single();
            Check(!first.IsComplete && first.Model.Files == 2 && first.StateText.StartsWith("unfinished", StringComparison.Ordinal)
                  && !File.Exists(Path.Combine(folder, "model.safetensors.part")) && File.Exists(Path.Combine(folder, ModelStore.PullMarker)),
                $"after a cancel between files: {first.Model}");
            Check(store.List().Single() is { Complete: false }, "listed as complete");

            // Cancelled inside the weights: a partial file stays.
            hub.OnFile = null;
            hub.Interrupt = ("model.safetensors", 64 << 10, page.Cancel);
            Check(!page.PullAsync().GetAwaiter().GetResult(), "a cancelled pull succeeded");
            long partial = new FileInfo(Path.Combine(folder, "model.safetensors.part")).Length;
            Check(partial >= 64 << 10 && partial < 200_000 && !page.Models.Single().IsComplete, $"partial {partial}");

            // Pulled again: it resumes where it stopped and is complete, the same bytes.
            hub.Interrupt = null;
            Check(page.PullAsync().GetAwaiter().GetResult(), $"not resumed: {page.Error}");
            Check(hub.Downloads("model.safetensors").Last().Range == $"bytes={partial}-", $"range {hub.Downloads("model.safetensors").Last().Range}");
            Check(page.Models.Single() is { IsComplete: true } done && done.Model.Bytes == 200_400 && !File.Exists(Path.Combine(folder, ModelStore.PullMarker))
                  && File.ReadAllBytes(Path.Combine(folder, "model.safetensors")).AsSpan().SequenceEqual(Bytes(200_000, 3)), "the resumed model");
        });
    }

    private static void ModelStoreSpace()
    {
        InStore("space", cache =>
        {
            var hub = TinyModel(new FakeHub(), "lab/space");
            var options = new IdrakMauiOptions();
            var store = new ModelStore(options, new HttpClient(hub), folder => new StorageSpace(cache, 100_000, 1_000_000));
            var refused = Throws<NotEnoughSpaceException>(() => store.PullAsync("lab/space"), "a pull too large for the free space");
            Check(refused.Needed == 200_400 && refused.Free == 100_000 && refused.Kept == 0 && refused.Message.Contains("Not enough space for lab/space", StringComparison.Ordinal),
                refused.Message);
            Check(!hub.Requests.Any(r => r.Path.Contains("/resolve/", StringComparison.Ordinal)) && !Directory.Exists(RevisionFolder(store, "lab/space")), "a byte downloaded before the refusal");

            // What a partial file holds is not needed again; the headroom kept counts.
            string folder = RevisionFolder(store, "lab/space");
            Directory.CreateDirectory(folder);
            File.WriteAllBytes(Path.Combine(folder, "model.safetensors.part"), Bytes(200_000, 3)[..150_000]);
            options.StorageHeadroomMiB = 1;
            var kept = Throws<NotEnoughSpaceException>(() => store.PullAsync("lab/space"), "a pull leaving less than the headroom");
            Check(kept.Needed == 50_400 && kept.Kept == 1 << 20 && kept.Message.Contains("keeping", StringComparison.Ordinal), kept.Message);
            options.StorageHeadroomMiB = null;
            Check(store.PullAsync("lab/space").GetAwaiter().GetResult().Complete, "a pull that fits");

            // The real measurement: the nearest folder that exists, its file system's space.
            var measured = ModelStore.Measure(Path.Combine(cache, "not", "yet"));
            Check(measured is { } m && m.Folder == Path.GetFullPath(cache) && m.FreeBytes > 0 && m.TotalBytes >= m.FreeBytes, $"measured {measured}");
            Check(new ModelStore(new IdrakMauiOptions()).FreeSpace() is { FreeBytes: > 0 }, "the store's free space");
        });
    }

    private static void ModelsPageProgress()
    {
        InStore("page", _ =>
        {
            var time = new ManualTime();
            var hub = TinyModel(new FakeHub { Time = time }, "lab/page");
            var store = new ModelStore(new IdrakMauiOptions(), new HttpClient(hub));
            var options = new IdrakMauiOptions { SuggestedModels = ["lab/page", " lab/page ", "lab/gguf/model.gguf", ""] };
            var center = new ProgressCenter(TimeSpan.FromMilliseconds(250), time: time);
            var page = new ModelsViewModel(store, options, center);
            Check(page.HasSuggestions && page.Suggestions.Select(s => s.Model).SequenceEqual(["lab/page", "lab/gguf/model.gguf"]), "suggestions, each once");
            page.Suggestions[0].Command.Execute(null);
            Check(page.ModelId == "lab/page", "a suggestion fills the id");
            Check(page.HasNoModels, "an empty store says so in the header");

            var seen = new List<long>();
            page.PropertyChanged += (_, e) =>
            {
                if (e.PropertyName == nameof(ModelsViewModel.Pull) && page.Pull is { } item)
                {
                    item.PropertyChanged += (_, p) =>
                    {
                        if (p.PropertyName == nameof(ProgressItem.Done))
                        {
                            seen.Add(item.Done);
                        }
                    };
                }
            };
            Check(page.PullAsync().GetAwaiter().GetResult(), $"not pulled: {page.Error}");
            var pull = page.Pull!;
            Check(center.Items.Contains(pull) && pull.IsFinished && pull.Done == 200_400 && pull.Total == 200_400 && pull.Fraction == 1 && pull.Unit == ProgressUnit.Bytes,
                $"the item: done {pull.Done} of {pull.Total}, finished {pull.IsFinished}");
            Check(seen.Any(d => d > 0 && d < 200_400) && seen.SequenceEqual(seen.Order()), $"the amounts shown on the way: {string.Join(", ", seen)}");
            Check(pull.Rate > 0 && pull.RateText.Length > 0 && pull.AmountText == $"{ProgressText.Bytes(200_400)} / {ProgressText.Bytes(200_400)}"
                  && pull.Message.Contains("lab/page", StringComparison.Ordinal), $"rate '{pull.RateText}', amount '{pull.AmountText}', message '{pull.Message}'");
            Check(pull.Updates < 20, $"{pull.Updates} UI updates for one pull");
            Check(page.Status.Contains("lab/page is in", StringComparison.Ordinal) && !page.IsPulling, page.Status);

            // The list after it, with sizes; the storage and the token.
            var entry = page.Models.Single();
            Check(!page.HasNoModels, "the empty line hidden once a model is kept");
            Check(entry.Title == "lab/page" && entry.IsComplete && entry.Details.StartsWith(ProgressText.Bytes(200_400) + " in 3 files", StringComparison.Ordinal)
                  && entry.Folder == RevisionFolder(store, "lab/page"), $"entry {entry.Details}");
            Check(page.ModelsText.StartsWith("1 model, ", StringComparison.Ordinal) && page.StorageText.Contains("free of", StringComparison.Ordinal)
                  && page.TokenText.StartsWith("No Hugging Face token", StringComparison.Ordinal), $"'{page.ModelsText}', '{page.StorageText}', '{page.TokenText}'");

            // Removing asks first: nothing without a confirmation, nothing when it is refused, then it goes.
            Check(!page.RemoveAsync(entry.Model).GetAwaiter().GetResult() && page.HasError && page.Models.Count == 1, "removed without a way to confirm");
            var asked = new List<string>();
            page.Confirm = (title, message) =>
            {
                asked.Add(message);
                return Task.FromResult(asked.Count > 1);
            };
            Check(!page.RemoveAsync(entry.Model).GetAwaiter().GetResult() && page.Models.Count == 1, "removed though refused");
            Check(page.RemoveAsync(entry.Model).GetAwaiter().GetResult() && page.Models.Count == 0 && page.Status.StartsWith("Removed lab/page@", StringComparison.Ordinal)
                  && page.ModelsText == "No models pulled yet." && asked[0].Contains(ProgressText.Bytes(200_400), StringComparison.Ordinal), $"'{page.Status}'; asked '{asked[0]}'");
        });
    }

    private static void ModelsPageErrors()
    {
        InStore("errors", cache =>
        {
            var hub = TinyModel(new FakeHub(), "lab/gate");
            hub.Gated.Add("lab/gate");
            hub.AcceptedToken = "hf_right";
            var options = new IdrakMauiOptions();
            var page = new ModelsViewModel(new ModelStore(options, new HttpClient(hub)), options, new ProgressCenter(TimeSpan.FromMilliseconds(250)));

            string Pull(string model)
            {
                page.ModelId = model;
                Check(!page.PullAsync().GetAwaiter().GetResult(), $"{model} was pulled");
                return page.Error;
            }

            Check(Pull("").StartsWith("Type a Hugging Face id", StringComparison.Ordinal), page.Error);
            Check(Pull("not a model").Contains("is not a Hugging Face id", StringComparison.Ordinal), page.Error);
            Check(Pull("lab/gate").StartsWith("lab/gate needs a Hugging Face token (401)", StringComparison.Ordinal), page.Error);
            Check(page.Pull is { IsFinished: true } item && item.Message.Contains("not pulled", StringComparison.Ordinal), "the progress item says it ended");
            options.HuggingFaceToken = "hf_wrong";
            Check(Pull("lab/gate").StartsWith("No access to lab/gate (403): it is gated. Accept its terms", StringComparison.Ordinal), page.Error);
            Check(Pull("lab/missing").StartsWith("lab/missing was not found on Hugging Face (404)", StringComparison.Ordinal), page.Error);
            options.HuggingFaceToken = null;
            hub.Unreachable = new HttpRequestException("Name or service not known (huggingface.co:443)");
            Check(Pull("lab/other").StartsWith("No network: Hugging Face could not be reached (Name or service not known", StringComparison.Ordinal), page.Error);

            // The causes no local hub gives easily, through the same wording.
            Check(ModelsViewModel.Explain(new HttpRequestException("x", null, HttpStatusCode.Unauthorized), "m/x", tokenSet: true).StartsWith("m/x: the Hugging Face token was refused (401)", StringComparison.Ordinal),
                "a refused token");
            Check(ModelsViewModel.Explain(new HttpRequestException("x", null, HttpStatusCode.Forbidden), "m/x", tokenSet: false).Contains("Set a Hugging Face token in Settings", StringComparison.Ordinal),
                "gated without a token");
            var space = new NotEnoughSpaceException("m/x", 3L << 30, 1L << 30, 0, cache);
            Check(ModelsViewModel.Explain(space, "m/x", false) == space.Message && space.Message.Contains(ProgressText.Bytes(3L << 30), StringComparison.Ordinal), space.Message);
            var full = new IOException("No space left on device", OperatingSystem.IsWindows() ? unchecked((int)0x80070070) : 28);
            Check(ModelsViewModel.Explain(full, "m/x", false).StartsWith("The storage is full", StringComparison.Ordinal), "a full storage");
            Check(!ModelsViewModel.Explain(new IOException("Directory not empty", 39), "m/x", false).StartsWith("The storage is full", StringComparison.Ordinal) || OperatingSystem.IsWindows(),
                "another file system error read as a full storage");
            Check(ModelsViewModel.Explain(new OperationCanceledException(), "m/x", false).StartsWith("Cancelled: what m/x downloaded so far is kept", StringComparison.Ordinal), "cancelled");
            Check(ModelsViewModel.Explain(new HttpRequestException("x", null, HttpStatusCode.BadGateway), "m/x", false).StartsWith("m/x: Hugging Face answered 502", StringComparison.Ordinal),
                "a server error");
        });
    }
}
