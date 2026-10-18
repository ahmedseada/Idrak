// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Cluster;
using Idrak.Cli;
using Idrak.Cli.Commands.Cluster;
using Idrak.Cli.Commands.Serve;
using Idrak.Cluster;
using Idrak.Nlp;
using EventLog = Idrak.Cluster.EventLog;

// Plan 23, phase C9: the cluster dashboard. Each test starts the real server on 127.0.0.1 (a port the system picks) and
// calls it as the page does, over HTTP; every wait is bounded, and every server, run and process is ended in finally.
internal static partial class Tests
{
    // A file another handle is still writing (the pretrainer's log): Windows refuses a reader that does not share writing.
    private static List<string> SharedLines(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        using var reader = new StreamReader(stream);
        var lines = new List<string>();
        while (reader.ReadLine() is { } line)
        {
            lines.Add(line);
        }

        return lines;
    }

    private static readonly (string Name, Action<Device> Run)[] ClusterDashboardGroup =
    [
        ("cluster: ui the builder API asks cluster init's questions (each with its description, default and suggestion, a refused answer saying why) and writes the file cluster init --yes writes for the same answers; the inspection's phases add up to each prediction", ClusterUiBuilder),
        ("cluster: ui a sim run started from the API pauses at one step boundary on every rank, resumes, and stops with a checkpoint after the step in progress, its losses and weights equal to an uninterrupted run's; the event stream carries every rank's steps, the collectives and the run's changes in order, and nothing is measured while no page listens", ClusterUiSteeredRun),
        ("cluster: ui a fault injected live is drawn at the next step and written to the event log marked live with its step; a replay with --seed S --step N of the steered run reproduces its faults and clocks bit for bit; localhost and followed runs refuse faults naming why", ClusterUiLiveFault),
        ("cluster: ui the dashboard binds to 127.0.0.1 by default; a request from another machine is refused without a key and taken with a set or stored one; another site's page is refused", ClusterUiAccess),
        ("cluster: ui the page names no external address, every call it makes is a route the server answers and every route is called; it is served in the tool's language, right to left in Arabic", ClusterUiPage),
        ("cluster: ui a run folder written by cluster start is followed read-only: rank 0's log and the ranks' lines stream as events, and pause, resume, stop and faults are refused naming why", ClusterUiFollow),
        ("cluster: ui a sharded run (Shard = parameters) pauses on every rank at one step and its stop ends every rank at one step boundary with a checkpoint joining every rank's share: weights, optimizer state and losses as an uninterrupted run's", ClusterUiShardedStop),
    ];

    // A dashboard on 127.0.0.1 and a client of it.
    private static (DashboardServer Server, HttpClient Http) UiServer(string? topology)
    {
        var server = new DashboardServer(new DashboardSettings("127.0.0.1", 0, null, [], topology));
        server.Start();
        return (server, new HttpClient { BaseAddress = new Uri(server.Url!), Timeout = TimeSpan.FromMinutes(2) });
    }

    private static (HttpStatusCode Status, JsonNode? Body) UiCall(HttpClient http, string method, string path, JsonNode? body = null)
    {
        using var request = new HttpRequestMessage(new HttpMethod(method), path);
        if (body is not null)
        {
            request.Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json");
        }

        using var response = http.Send(request);
        string text = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        return (response.StatusCode, text.Length == 0 ? null : JsonNode.Parse(text));
    }

    private static JsonNode UiOk(HttpClient http, string method, string path, JsonNode? body = null)
    {
        var (status, answer) = UiCall(http, method, path, body);
        Check(status == HttpStatusCode.OK && answer is not null, $"{method} {path}: {(int)status} {answer?.ToJsonString()}");
        return answer!;
    }

    // Waits up to `seconds` for `done`, polling.
    private static bool UiWait(Func<bool> done, double seconds)
    {
        var watch = Stopwatch.StartNew();
        while (watch.Elapsed.TotalSeconds < seconds)
        {
            if (done())
            {
                return true;
            }

            Thread.Sleep(50);
        }

        return done();
    }

    // A page listening to a run's events (server-sent), its records kept in order.
    private sealed class UiListener : IDisposable
    {
        private readonly CancellationTokenSource _stop = new(TimeSpan.FromMinutes(10));
        private readonly List<JsonObject> _records = [];
        private readonly HttpClient _http;
        private readonly Task _reading;

        public UiListener(string url, string runId)
        {
            _http = new HttpClient { BaseAddress = new Uri(url), Timeout = Timeout.InfiniteTimeSpan };
            var response = _http.GetAsync($"/api/runs/{runId}/events", HttpCompletionOption.ResponseHeadersRead, _stop.Token).GetAwaiter().GetResult();
            Check(response.IsSuccessStatusCode && response.Content.Headers.ContentType?.MediaType == "text/event-stream", $"the events of {runId}: {(int)response.StatusCode}");
            _reading = Task.Run(async () =>
            {
                try
                {
                    using var reader = new StreamReader(await response.Content.ReadAsStreamAsync(_stop.Token));
                    while (await reader.ReadLineAsync(_stop.Token) is { } line)
                    {
                        if (line.StartsWith("data: ", StringComparison.Ordinal) && JsonNode.Parse(line["data: ".Length..]) is JsonObject record)
                        {
                            lock (_records)
                            {
                                _records.Add(record);
                            }
                        }
                    }
                }
                catch (Exception e) when (e is OperationCanceledException or IOException or HttpRequestException)
                {
                    // Closed by the test.
                }
                finally
                {
                    response.Dispose();
                }
            });
        }

        public List<JsonObject> Records
        {
            get
            {
                lock (_records)
                {
                    return [.. _records];
                }
            }
        }

        public bool WaitFor(Func<JsonObject, bool> match, double seconds = 120) => UiWait(() => Records.Any(match), seconds);

        public void Dispose()
        {
            _stop.Cancel();
            _reading.Wait(TimeSpan.FromSeconds(10));
            _http.Dispose();
            _stop.Dispose();
        }
    }

    private static bool Is(JsonObject record, string name) => (string?)record["event"] == name;

    // The tiny model's pretraining on the test tokens: plain SGD at a constant rate, so a run's first steps do not depend on its length.
    private static string[] UiPretrain(string run, int steps) =>
    [
        "pretrain", TinyLlama, "--tokens", PretrainData("tokens.bin"), "--steps", steps.ToString(CultureInfo.InvariantCulture), "--max-length", "32", "--batch-tokens", "64",
        "--optimizer", "sgd", "--lr", "0.2", "--schedule", "constant", "--warmup", "0", "--no-graphs", "-o", run,
    ];

    // Equal shares, so two runs of one command split and sum alike (restored by disposing).
    private static IDisposable UiEnvironment()
    {
        var old = new Dictionary<string, string?>();
        foreach (var (name, value) in new[] { ("IDRAK_CLUSTER_REBALANCE_STEPS", "0") })
        {
            old[name] = Environment.GetEnvironmentVariable(name);
            Environment.SetEnvironmentVariable(name, value);
        }

        return new Undo(() =>
        {
            foreach (var (name, value) in old)
            {
                Environment.SetEnvironmentVariable(name, value);
            }
        });
    }

    private static JsonArray UiArgs(string[] args) => new([.. args.Select(a => (JsonNode)a)]);

    private static string UiState(HttpClient http, string id) => (string)UiOk(http, "GET", "/api/runs")["runs"]!.AsArray().First(r => (string?)r!["id"] == id)!["state"]!;

    private static void ClusterUiBuilder(Device device)
    {
        if (!FirstRun(nameof(ClusterUiBuilder)))
        {
            return;
        }

        string folder = TempFolder("cluster-ui-builder");
        var (server, http) = UiServer(null);
        try
        {
            var given = new Dictionary<string, string>
            {
                ["nodes"] = "2", ["node.1.name"] = "desk", ["node.1.address"] = "10.0.0.1:29500", ["node.2.name"] = "mini", ["node.1.devices"] = "cpu,cpu",
                ["node.1.device.1.memory"] = "64MB", ["node.1.device.2.memory"] = "64MB", ["node.2.devices"] = "cpu", ["node.2.device.1.memory"] = "64MB",
                ["inter-node"] = "ethernet-1g",
            };
            var asked = new List<string>();
            var answers = new JsonObject();
            var step = UiOk(http, "POST", "/api/topology/questions", new JsonObject { ["answers"] = new JsonObject() });
            var refused = UiOk(http, "POST", "/api/topology/questions", new JsonObject { ["answers"] = new JsonObject(), ["answer"] = "zero" });
            Check((string?)refused["refused"] is { } why && why.Contains("a number of machines", StringComparison.Ordinal) && (string?)refused["question"]!["key"] == "nodes",
                $"a refused answer says why and asks again: {refused.ToJsonString()}");
            while (step["done"]!.GetValue<bool>() == false)
            {
                var question = step["question"]!;
                string key = (string)question["key"]!;
                Check(((string?)question["prompt"])?.Length > 0 && ((string?)question["description"])?.Length > 0 && question.AsObject().ContainsKey("default") && question.AsObject().ContainsKey("suggestion"),
                    $"question {key} has its prompt, description, default and suggestion: {question.ToJsonString()}");
                asked.Add(key);
                step = UiOk(http, "POST", "/api/topology/questions", new JsonObject { ["answers"] = step["answers"]!.DeepClone(), ["answer"] = given.GetValueOrDefault(key, "") });
                Check((string?)step["refused"] is null, $"{key} answered: {step.ToJsonString()}");
                Check(asked.Count < 100, "the builder ends");
            }

            answers = step["answers"]!.AsObject();
            // The same questions in the same order as the builder cluster init asks with.
            var direct = new TopologyBuilder(MachineFacts.Gather());
            var directKeys = new List<string>();
            while (direct.Next is { } q)
            {
                directKeys.Add(q.Key);
                Check(direct.Answer(given.GetValueOrDefault(q.Key, "")) is null, $"the builder takes {q.Key}");
            }

            Check(asked.SequenceEqual(directKeys), $"the API asks cluster init's questions in its order: {string.Join(", ", asked)} / {string.Join(", ", directKeys)}");
            string viaApi = Path.Combine(folder, "api.json"), viaInit = Path.Combine(folder, "init.json");
            var saved = UiOk(http, "POST", "/api/topology", new JsonObject { ["answers"] = answers.DeepClone(), ["save"] = true, ["path"] = viaApi });
            Check(saved["saved"]!.GetValue<bool>() && (int)saved["ranks"]! == 3 && File.Exists(viaApi), $"the API saved the topology: {saved.ToJsonString()}");
            using (var output = new StringWriter(CultureInfo.InvariantCulture))
            using (var error = new StringWriter(CultureInfo.InvariantCulture))
            {
                int code = CommandLine.Run(["cluster", "init", "-y", "-o", viaInit, "--nodes", "2", "--names", "desk,mini", "--address", "10.0.0.1:29500", "--devices", "cpu,cpu;cpu",
                    "--memory", "64MB,64MB;64MB", "--inter", "ethernet-1g"], output, error);
                Check(code == ExitCodes.Ok, $"cluster init -y: {output}{error}");
            }

            Check(File.ReadAllText(viaApi) == File.ReadAllText(viaInit), "the API writes the file cluster init --yes writes for the same answers, byte for byte");
            var loaded = UiOk(http, "GET", "/api/topology?path=" + Uri.EscapeDataString(viaApi));
            Check(JsonNode.DeepEquals(loaded["answers"], answers), $"the saved file reads back as the same answers: {loaded["answers"]?.ToJsonString()} / {answers.ToJsonString()}");
            var bad = UiCall(http, "POST", "/api/topology", new JsonObject { ["answers"] = new JsonObject { ["nodes"] = "1", ["node.1.name"] = "desk", ["node.1.devices"] = "cpu", ["node.1.device.1.memory"] = "64" } });
            Check(bad.Status == HttpStatusCode.BadRequest && ((string?)bad.Body?["error"])?.Contains("node.1.device.1.memory", StringComparison.Ordinal) == true,
                $"a refused answer is a 400 naming its key: {bad.Body?.ToJsonString()}");

            var inspection = UiOk(http, "GET", "/api/topology/inspect?path=" + Uri.EscapeDataString(viaApi) + "&bytes=1MB");
            Check((int)inspection["world"]! == 3 && inspection["pairs"]!.AsArray().Count == 3 && (string?)inspection["default_collective"] == "hierarchical"
                  && inspection["pairs"]!.AsArray().Count(p => (string?)p!["kind"] == "same-device") == 1, $"the inspection's pairs and default: {inspection.ToJsonString()}");
            foreach (var prediction in inspection["predictions"]!.AsArray())
            {
                double total = (double)prediction!["seconds"]!;
                double phases = prediction["phases"]!.AsArray().Sum(p => (double)p!["seconds"]!);
                int count = prediction["phases"]!.AsArray().Count;
                // Each time is a TimeSpan, a whole number of 100 ns ticks: the sum may differ from the total by a tick a phase.
                Check(total > 0 && Math.Abs(phases - total) <= 1e-7 * (count + 1) && prediction["phases"]!.AsArray().All(p => p!["links"]!.AsArray().Count > 0),
                    $"{prediction["algorithm"]}: its phases ({phases}) add up to the prediction ({total}) and name their links");
            }

            var hierarchical = inspection["predictions"]!.AsArray().First(p => (string?)p!["algorithm"] == "hierarchical")!["phases"]!.AsArray();
            Check(hierarchical.Count == 3 && ((string)hierarchical[1]!["name"]!).Contains("across the nodes", StringComparison.Ordinal)
                  && hierarchical[1]!["links"]!.AsArray().Any(l => (string?)l == "ethernet-1g"), $"hierarchical's three phases, the middle one across the nodes on ethernet-1g: {hierarchical.ToJsonString()}");
            var draft = UiOk(http, "POST", "/api/topology/inspect", new JsonObject { ["answers"] = answers.DeepClone(), ["bytes"] = "1MB" });
            Check(JsonNode.DeepEquals(draft["predictions"], inspection["predictions"]), "a draft from the form's answers inspects as the saved file");
        }
        finally
        {
            http.Dispose();
            server.Dispose();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterUiSteeredRun(Device device)
    {
        if (!FirstRun(nameof(ClusterUiSteeredRun)))
        {
            return;
        }

        string folder = TempFolder("cluster-ui-steered");
        using var environment = UiEnvironment();
        string topology = Path.Combine(folder, "two.json");
        File.WriteAllText(topology, DpTopology(device, 2).ToJson());
        var (server, http) = UiServer(topology);
        string steered = Path.Combine(folder, "steered"), plain = Path.Combine(folder, "plain");
        long last;
        try
        {
            var started = UiOk(http, "POST", "/api/runs", new JsonObject { ["command"] = UiArgs(UiPretrain(steered, 1000)), ["mode"] = "in-process", ["paused"] = true });
            string id = (string)started["id"]!;
            var run = server.Runs.Find(id)!;
            var control = run.Control!;
            Check(UiWait(() => UiState(http, id) == "paused", 120), $"the run waits before step 0 as asked: {UiState(http, id)}");
            Check(control is { PauseAt: 0, LastStep: -1, Observed: false } && server.Runs.Listening == 0, "paused before step 0, no rank began it; nobody observes");
            UiOk(http, "POST", $"/api/runs/{id}/resume");
            string log = Path.Combine(steered, "log.jsonl");
            Check(UiWait(() => File.Exists(log) && SharedLines(log).Count(l => l.Contains("\"event\":\"batch\"", StringComparison.Ordinal)) >= 3, 180),
                "three steps trained with no page listening");
            // Paused before the page connects, so what it gets first is what was kept while nobody listened.
            long early = (long)UiOk(http, "POST", $"/api/runs/{id}/pause")["at"]!;
            Check(UiWait(() => UiState(http, id) == "paused", 120) && control.LastStep == early - 1, $"paused before step {early} with no page listening");
            using (var page = new UiListener(server.Url!, id))
            {
                Check(page.WaitFor(r => Is(r, "run") && (string?)r["state"] == "paused" && (long)r["step"]! == early, 30), "the kept records arrive");
                Thread.Sleep(200);
                var kept = page.Records;
                Check(kept.Count > 0 && !kept.Any(r => Is(r, "batch") || Is(r, "rank_step") || Is(r, "gradient_sync") || Is(r, "collectives") || Is(r, "links")),
                    $"nothing was measured while no page listened: {string.Join(", ", kept.Select(r => (string?)r["event"]).Distinct())}");
                Check(server.Runs.Listening == 1 && control.Observed, "a page listens: the steps are observed");
                UiOk(http, "POST", $"/api/runs/{id}/resume");
                Check(page.WaitFor(r => Is(r, "rank_step") && (long)r["step"]! >= early + 2, 120), "the ranks' steps stream while the page listens");

                long pausedAt = (long)UiOk(http, "POST", $"/api/runs/{id}/pause")["at"]!;
                Check(page.WaitFor(r => Is(r, "run") && (string?)r["state"] == "paused" && (long)r["step"]! == pausedAt, 120), "the pause is reached");
                Check(control.LastStep == pausedAt - 1, $"every rank waits before step {pausedAt}: the highest step begun is {control.LastStep}");
                var atPause = page.Records;
                int pauseIndex = atPause.FindIndex(r => Is(r, "run") && (string?)r["state"] == "paused" && (long)r["step"]! == pausedAt);
                Check(atPause.Where(r => Is(r, "rank_step")).All(r => (long)r["step"]! < pausedAt), $"no rank began step {pausedAt} while paused");
                Thread.Sleep(300);
                Check(!page.Records.Skip(pauseIndex + 1).Any(r => Is(r, "rank_step") || Is(r, "batch") && (long)r["step"]! > pausedAt), "nothing trains while paused");
                UiOk(http, "POST", $"/api/runs/{id}/resume");
                Check(page.WaitFor(r => Is(r, "rank_step") && (long)r["step"]! >= pausedAt + 2 && (int)r["rank"]! == 1, 120), "the run goes on after the resume");
                last = (long)UiOk(http, "POST", $"/api/runs/{id}/stop")["at"]!;
                Check(page.WaitFor(r => Is(r, "run") && (string?)r["state"] is "stopped" or "failed", 180), "the run ends after the stop");
                Check(UiState(http, id) == "stopped", $"the run stopped as asked: {run.Detail}");
                string checkpoint = Path.Combine(steered, $"checkpoint-{last + 1}");
                Check(Directory.Exists(checkpoint), $"the stop wrote the checkpoint of the last finished step ({checkpoint})");

                // The stream, in order: every rank's steps go forward; a rank's sync of a step comes before it begins the next;
                // every rank's step comes before the step's batch (rank 0 cannot finish a step the others have not begun), and
                // rank 0's sync before its batch.
                var records = page.Records;
                for (int r = 0; r < 2; r++)
                {
                    var mine = records.Select((x, i) => (Record: x, At: i)).Where(x => (int?)x.Record["rank"] == r).ToList();
                    var steps = mine.Where(x => Is(x.Record, "rank_step")).ToList();
                    Check(steps.Count > 3 && steps.Zip(steps.Skip(1)).All(p => (long)p.Second.Record["step"]! == (long)p.First.Record["step"]! + 1),
                        $"rank {r}'s steps in order: {string.Join(", ", steps.Select(x => x.Record["step"]))}");
                    foreach (var (begun, next) in steps.Zip(steps.Skip(1)))
                    {
                        long step = (long)begun.Record["step"]!;
                        Check(mine.Count(x => Is(x.Record, "gradient_sync") && (long)x.Record["step"]! == step && x.At > begun.At && x.At < next.At) == 1,
                            $"rank {r}'s sync of step {step} comes after it began the step and before it began the next");
                    }
                }

                // From the first step both ranks began while the page listened.
                long observed = Enumerable.Range(0, 2).Max(r => records.Where(x => Is(x, "rank_step") && (int)x["rank"]! == r).Min(x => (long)x["step"]!));
                foreach (var batch in records.Where(x => Is(x, "batch") && (long)x["step"]! - 1 >= observed))
                {
                    long done = (long)batch["step"]! - 1;
                    int at = records.IndexOf(batch);
                    int ranksBefore = records.Take(at).Count(x => Is(x, "rank_step") && (long)x["step"]! == done);
                    bool syncBefore = records.Take(at).Any(x => Is(x, "gradient_sync") && (long)x["step"]! == done && (int)x["rank"]! == 0);
                    Check(ranksBefore == 2 && syncBefore && Is(records[at - 1], "collectives") && Is(records[at - 2], "links"),
                        $"step {done}: both ranks' steps and rank 0's sync, then the step's links and collectives, before its batch ({ranksBefore}, {syncBefore})");
                }

                Check(records.Any(x => Is(x, "collectives") && x["ranks"]!.AsArray().Count == 2 && x["ranks"]!.AsArray().All(c => (long)c!["count"]! > 0)),
                    "the collectives of every rank are in the stream");
            }

            Check(UiWait(() => server.Runs.Listening == 0, 30) && !control.Observed, "the page left: the telemetry is unsubscribed and nothing is observed");

            // The run uninterrupted for as many steps: the same losses and weights, bit for bit.
            using (var output = new StringWriter(CultureInfo.InvariantCulture))
            using (var error = new StringWriter(CultureInfo.InvariantCulture))
            {
                int steps = (int)last + 1;
                int code = CommandLine.Run(["cluster", "sim", "--topology", topology, "--", .. UiPretrain(plain, steps)], output, error);
                Check(code == ExitCodes.Ok, $"the uninterrupted run: {output}{error}");
                var a = Pretrainer.ReadCheckpoint(Path.Combine(steered, $"checkpoint-{steps}"));
                var b = Pretrainer.ReadCheckpoint(Path.Combine(plain, $"checkpoint-{steps}"));
                Check(a.Losses.Count == steps && a.Losses.SequenceEqual(b.Losses), $"the steered run's {steps} losses are the uninterrupted run's: {string.Join(", ", a.Losses)} / {string.Join(", ", b.Losses)}");
                Check(File.ReadAllBytes(Path.Combine(steered, $"checkpoint-{steps}", "model.safetensors")).AsSpan()
                    .SequenceEqual(File.ReadAllBytes(Path.Combine(plain, $"checkpoint-{steps}", "model.safetensors"))), "and its weights, byte for byte");
                Console.WriteLine($"    steered: paused, resumed and stopped after {steps} steps; losses and weights as the uninterrupted run's");
            }
        }
        finally
        {
            http.Dispose();
            server.Dispose();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterUiLiveFault(Device device)
    {
        if (!FirstRun(nameof(ClusterUiLiveFault)))
        {
            return;
        }

        string folder = TempFolder("cluster-ui-fault");
        using var environment = UiEnvironment();
        string topology = Path.Combine(folder, "three.json");
        var shape = DpTopology(device, 3);
        File.WriteAllText(topology, shape.ToJson());
        var (server, http) = UiServer(topology);
        string steered = Path.Combine(folder, "steered");
        const int Seed = 7;
        const int Steps = 400;                         // far more than the test takes: the stop always finds the run going
        try
        {
            string id = (string)UiOk(http, "POST", "/api/runs", new JsonObject { ["command"] = UiArgs(UiPretrain(steered, Steps)), ["seed"] = Seed, ["paused"] = true })["id"]!;
            Check(UiWait(() => UiState(http, id) == "paused", 120), "paused before step 0");
            UiOk(http, "POST", $"/api/runs/{id}/resume");
            var run = server.Runs.Find(id)!;
            Check(UiWait(() => run.Control!.LastStep >= 2, 120), $"two steps begun: the run is {run.State} ({run.Detail}), its control {run.Control!.State} at step {run.Control.LastStep}");
            long at = (long)UiOk(http, "POST", $"/api/runs/{id}/pause")["at"]!;
            Check(UiWait(() => UiState(http, id) == "paused", 120), "paused again");
            var bad = UiCall(http, "POST", $"/api/runs/{id}/faults", new JsonObject { ["kind"] = "SlowWorker", ["ranks"] = new JsonArray(9) });
            Check(bad.Status == HttpStatusCode.BadRequest && ((string?)bad.Body?["error"])?.Contains("outside 0 to 2", StringComparison.Ordinal) == true, $"a fault on no rank is refused: {bad.Body?.ToJsonString()}");
            var queued = UiOk(http, "POST", $"/api/runs/{id}/faults", new JsonObject { ["kind"] = "slow-worker", ["ranks"] = new JsonArray(1), ["factor"] = 8 });
            Check((long)queued["step"]! == at && (string?)queued["kind"] == "SlowWorker", $"queued for step {at}: {queued.ToJsonString()}");
            UiOk(http, "POST", $"/api/runs/{id}/resume");
            Check(UiWait(() => run.Control!.LastStep >= at + 2, 120), "two steps after the fault");
            Check(UiState(http, id) is "running" or "paused", $"the run still goes before its stop: {run.State} ({run.Detail}) at step {run.Control!.LastStep}");
            long last = (long)UiOk(http, "POST", $"/api/runs/{id}/stop")["at"]!;
            Check(UiWait(() => UiState(http, id) == "stopped", 180), $"stopped: {run.Detail}");

            var log = EventLog.Load(Path.Combine(steered, ClusterSimCommand.EventLogFile));
            Check(log.LiveFaults.Count == 1 && log.LiveFaults[0] is { Kind: FaultKind.SlowWorker, Factor: 8 } live && live.Step == at && live.Ranks.SequenceEqual([1])
                  && log.Faults.Contains(live) && log.Seed == Seed, $"the live fault is in the event log with its step {at}, marked live: {string.Join("; ", log.Faults.Select(f => $"{f.Kind}@{f.Step}"))}");

            // The replay of the steered run through the step after the fault: the same faults and clocks, bit for bit.
            long through = at + 1;
            Check(through <= last, $"the run passed step {through}");
            // The steered run's ranks measured for the cluster advisor (the dashboard shows suggestions): its two slice
            // collectives are on the clock, so the replay measures alike (cluster replay does, through the command line).
            var options = ClusterOptions.FromEnvironment() with { Suggest = true };
            ClusterRunResult Replay(string name, EventLog? steering)
            {
                void Work(ClusterRank rank)
                {
                    ClusterSession.Current = rank;
                    try
                    {
                        using var output = new StringWriter(CultureInfo.InvariantCulture);
                        int code = CommandLine.Run(UiPretrain(Path.Combine(folder, name), Steps), rank.Rank == 0 ? output : TextWriter.Null, output);
                        Check(code == ExitCodes.Ok, $"rank {rank.Rank} of the replay: {output}");
                    }
                    finally
                    {
                        ClusterSession.Current = null;
                    }
                }

                var result = steering is null ? ClusterReplay.Run(shape, options, Seed, through, Work) : ClusterReplay.Run(shape, options, Seed, through, Work, steering!);
                result.ThrowIfFailed();
                return result;
            }

            var replay = Replay("replay", log);
            Check(replay.Log!.LiveFaults.Count == 1 && replay.Log.LiveFaults[0].Step == at && replay.Log.Faults.SequenceEqual(log.Faults.Where(f => f.Step <= through), FaultComparer.Instance),
                "the replay injects the live fault at its step, and the run's own faults, as the steered run did");
            for (int r = 0; r < 3; r++)
            {
                for (long s = 0; s <= through; s++)
                {
                    Check(replay.Log.ClockAt(r, s) is { } c && c == log.ClockAt(r, s), $"rank {r}'s clock as step {s} began: {replay.Log.ClockAt(r, s)} against {log.ClockAt(r, s)}");
                }
            }

            // The same through idrak cluster replay: the live fault from the run's event log, the rerun's clocks bit for bit.
            var cli = ClusterCli(["cluster", "replay", "--topology", topology, "--step", through.ToString(CultureInfo.InvariantCulture), "--from",
                Path.Combine(steered, ClusterSimCommand.EventLogFile), "--", .. UiPretrain(Path.Combine(folder, "cli-replay"), Steps)]);
            Check(cli.Code == ExitCodes.Ok && cli.Output.Contains("live faults from the log (1)", StringComparison.Ordinal) && cli.Output.Contains("matches", StringComparison.Ordinal),
                $"idrak cluster replay --from the steered run's log reproduces it: {cli.Code} {cli.Error}\n{cli.Output}");
            var unsteered = Replay("unsteered", null);
            Check(Enumerable.Range(0, 3).Any(r => unsteered.Log!.ClockAt(r, through) != log.ClockAt(r, through)),
                "without the live fault the clocks differ: the replay reproduced the fault, not a run without it");
            Console.WriteLine($"    live slow worker at step {at}: replay through step {through} equal to the steered run's clocks; rank 1's clock {log.ClockAt(1, through):G6} s against {unsteered.Log!.ClockAt(1, through):G6} s unsteered");

            // Faults belong to the simulated transport: a localhost run and a followed folder refuse them, naming why.
            var localhost = server.Runs.Attach(RunKind.Localhost, topology, UiPretrain(Path.Combine(folder, "local"), 2), options, shape);
            Check(localhost.Refusal("faults") is { } why && why.Contains("simulated transport", StringComparison.Ordinal) && localhost.Refusal("pause") is null,
                $"a localhost run takes pause but refuses faults: {localhost.Refusal("faults")}");
            localhost.Finish("failed", "never started (a test's)");
            string followed = (string)UiOk(http, "POST", "/api/runs", new JsonObject { ["follow"] = steered })["id"]!;
            var refused = UiCall(http, "POST", $"/api/runs/{followed}/faults", new JsonObject { ["kind"] = "SlowWorker", ["ranks"] = new JsonArray(0) });
            Check(refused.Status == HttpStatusCode.Conflict && ((string?)refused.Body?["error"])?.Contains("read-only", StringComparison.Ordinal) == true, $"a followed run refuses faults: {refused.Body?.ToJsonString()}");
        }
        finally
        {
            http.Dispose();
            server.Dispose();
            Directory.Delete(folder, recursive: true);
        }
    }

    // Faults compared by value (their ranks are lists).
    private sealed class FaultComparer : IEqualityComparer<Fault>
    {
        public static readonly FaultComparer Instance = new();

        public bool Equals(Fault a, Fault b) => a.Kind == b.Kind && a.Step == b.Step && a.Ranks.SequenceEqual(b.Ranks) && a.Duration == b.Duration && a.Factor.Equals(b.Factor);

        public int GetHashCode(Fault f) => HashCode.Combine(f.Kind, f.Step);
    }

    private static void ClusterUiAccess(Device device)
    {
        if (!FirstRun(nameof(ClusterUiAccess)))
        {
            return;
        }

        // The command's default: 127.0.0.1 and a port the system picks, this machine only.
        using (var output = new StringWriter(CultureInfo.InvariantCulture))
        using (var error = new StringWriter(CultureInfo.InvariantCulture))
        {
            string? host = Environment.GetEnvironmentVariable(DashboardSettings.HostVariable), port = Environment.GetEnvironmentVariable(DashboardSettings.PortVariable);
            Environment.SetEnvironmentVariable(DashboardSettings.HostVariable, null);
            Environment.SetEnvironmentVariable(DashboardSettings.PortVariable, null);
            try
            {
                var watch = Stopwatch.StartNew();
                int code = CommandLine.Run(["cluster", "ui", "--no-browser", "--timeout", "2s"], output, error);
                Check(code == ExitCodes.Ok && watch.Elapsed < TimeSpan.FromSeconds(60), $"cluster ui ran until its timeout: {code}: {output}{error}");
                Check(Regex.IsMatch(output.ToString(), @"dashboard http://127\.0\.0\.1:\d+/ \(this machine only\)") && output.ToString().Contains("dashboard stopped", StringComparison.Ordinal),
                    $"it listened on 127.0.0.1 by default, at a port the system picked: {output}");
            }
            finally
            {
                Environment.SetEnvironmentVariable(DashboardSettings.HostVariable, host);
                Environment.SetEnvironmentVariable(DashboardSettings.PortVariable, port);
            }
        }

        var lan = IPAddress.Parse("192.168.1.20");
        var open = new DashboardSettings("0.0.0.0", 0, null, [], null);
        Check(DashboardServer.Refusal(IPAddress.Loopback, null, open) is null && DashboardServer.Refusal(IPAddress.IPv6Loopback, null, open) is null, "this machine's requests need no key");
        Check(DashboardServer.Refusal(lan, null, open) is { } none && none.Contains("idrak server keys add", StringComparison.Ordinal),
            $"another machine is refused when no key is set, naming how to add one: {DashboardServer.Refusal(lan, null, open)}");
        var keyed = new DashboardSettings("0.0.0.0", 0, "a set key of the owner", [ServerKeys.Hash("a stored key")], null);
        Check(DashboardServer.Refusal(lan, null, keyed) is { } missing && missing.Contains("API key", StringComparison.Ordinal), "without a key: refused");
        Check(DashboardServer.Refusal(lan, "a wrong key", keyed) is not null, "a wrong key: refused");
        Check(DashboardServer.Refusal(lan, "a set key of the owner", keyed) is null && DashboardServer.Refusal(lan, "a stored key", keyed) is null, "the set key and a stored key are taken");

        var (server, http) = UiServer(null);
        try
        {
            Check(server.Loopback && new Uri(server.Url!).Host == "127.0.0.1", $"the server listens on loopback: {server.Url}");
            Check(UiCall(http, "GET", "/api/machine").Status == HttpStatusCode.OK, "this machine's request is answered without a key");
            using var foreign = new HttpRequestMessage(HttpMethod.Post, "/api/runs") { Content = new StringContent("{}", Encoding.UTF8, "application/json") };
            foreign.Headers.Add("Origin", "http://another.example");
            using var answer = http.Send(foreign);
            Check(answer.StatusCode == HttpStatusCode.Forbidden, $"another site's page is refused: {(int)answer.StatusCode}");
            using var named = new HttpRequestMessage(HttpMethod.Get, "/api/machine");
            named.Headers.Host = "rebound.example";
            using var rebound = http.Send(named);
            Check(rebound.StatusCode == HttpStatusCode.Forbidden, $"a name that is not this machine's is refused on a loopback dashboard: {(int)rebound.StatusCode}");
        }
        finally
        {
            http.Dispose();
            server.Dispose();
        }
    }

    private static void ClusterUiPage(Device device)
    {
        if (!FirstRun(nameof(ClusterUiPage)))
        {
            return;
        }

        string page = DashboardPage.Html;
        Check(page.Length > 10_000 && !Regex.IsMatch(page, @"https?:", RegexOptions.IgnoreCase) && !Regex.IsMatch(page, @"(src|href|action)\s*=\s*[""']?\s*//"),
            "the page names no address outside its server (no http, https or protocol-relative address)");
        Check(!Regex.IsMatch(page, @"<(script|link|img|iframe)[^>]*(src|href)=", RegexOptions.IgnoreCase), "the page loads no file of its own: scripts, styles and drawings are inline");
        var called = Regex.Matches(page, @"'(GET|POST) (/api/[^']+)'").Select(m => m.Groups[1].Value + " " + m.Groups[2].Value).ToList();
        Check(called.Count > 0 && called.ToHashSet().SetEquals(DashboardServer.Routes), $"the page calls exactly the routes the server answers: {string.Join(", ", called)}");
        Check(Regex.Matches(page, @"/api/").Count == called.Count, "the page names no route outside its list of calls");

        var (server, http) = UiServer(null);
        try
        {
            foreach (string route in DashboardServer.Routes)
            {
                string method = route.Split(' ')[0], path = route.Split(' ')[1].Replace("{id}", "no-such-run", StringComparison.Ordinal);
                if (path.EndsWith("/events", StringComparison.Ordinal))
                {
                    using var events = http.GetAsync(path, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
                    string text = events.Content.ReadAsStringAsync().GetAwaiter().GetResult();
                    Check(events.StatusCode == HttpStatusCode.NotFound && text.Contains("No run no-such-run", StringComparison.Ordinal), $"{route} answers (no such run): {text}");
                    continue;
                }

                var (status, body) = UiCall(http, method, path, method == "POST" ? new JsonObject() : null);
                Check(status != HttpStatusCode.NotFound || ((string?)body?["error"])?.Contains("no-such-run", StringComparison.Ordinal) == true,
                    $"{route} is answered by the server ({(int)status} {body?.ToJsonString()})");
            }

            string served = http.GetStringAsync("/").GetAwaiter().GetResult();
            Check(served == page && served.Contains("<html lang=\"en\" dir=\"ltr\">", StringComparison.Ordinal), "GET / serves the page, left to right in English");
            Check(DashboardPage.For("ar").Contains("<html lang=\"ar\" dir=\"rtl\">", StringComparison.Ordinal), "in Arabic the page reads right to left");
        }
        finally
        {
            http.Dispose();
            server.Dispose();
        }
    }

    private static void ClusterUiFollow(Device device)
    {
        if (!FirstRun(nameof(ClusterUiFollow)))
        {
            return;
        }

        string folder = TempFolder("cluster-ui-follow");
        var (server, http) = UiServer(null);
        try
        {
            int port;
            using (var probe = new TcpListener(IPAddress.Loopback, 0))
            {
                probe.Start();
                port = ((IPEndPoint)probe.LocalEndpoint).Port;
            }

            var shape = ClusterTopology.Create([new ClusterNodeSpec("box", string.Create(CultureInfo.InvariantCulture, $"127.0.0.1:{port}"),
                [new ClusterDeviceSpec("cpu", null), new ClusterDeviceSpec("cpu", null)])], null, null);
            string topology = Path.Combine(folder, "box.json"), run = Path.Combine(folder, "run");
            File.WriteAllText(topology, shape.ToJson());
            using (var output = new StringWriter(CultureInfo.InvariantCulture))
            using (var error = new StringWriter(CultureInfo.InvariantCulture))
            {
                int code = CommandLine.Run(["cluster", "start", "--topology", topology, "--node", "box", "--", "idrak", .. UiPretrain(run, 2)], output, error);
                Check(code == ExitCodes.Ok, $"cluster start ran pretrain on the node's two ranks: {output}{error}");
            }

            string ranks = Path.Combine(run, DashboardTail.RanksFile);
            Check(File.Exists(ranks) && File.Exists(Path.Combine(run, "log.jsonl")), "cluster start wrote its ranks' lines beside rank 0's log");
            var notRun = UiCall(http, "POST", "/api/runs", new JsonObject { ["follow"] = folder });
            Check(notRun.Status == HttpStatusCode.BadRequest && ((string?)notRun.Body?["error"])?.Contains("log.jsonl", StringComparison.Ordinal) == true, $"a folder that is not a run is refused naming why: {notRun.Body?.ToJsonString()}");
            string id = (string)UiOk(http, "POST", "/api/runs", new JsonObject { ["follow"] = run })["id"]!;
            using (var page = new UiListener(server.Url!, id))
            {
                Check(page.WaitFor(r => Is(r, "batch") && (long)r["step"]! == 2, 60) && page.WaitFor(r => Is(r, "cluster_end"), 60)
                      && page.Records.Count(r => Is(r, "rank_end") && (long?)r["exit_code"] == 0) == 2 && page.Records.Any(r => Is(r, "rank_line")) && page.Records.Any(r => Is(r, "cluster_start")),
                    $"rank 0's log and the ranks' lines stream: {string.Join(", ", page.Records.Select(r => (string?)r["event"]).Distinct())}");
                Check(page.Records.Any(r => Is(r, "collective") && (string?)r["operation"] == "step-summary" && (int)r["rank"]! == 1), "every rank's step summary from rank 0's log");
            }

            foreach (string action in new[] { "pause", "resume", "stop" })
            {
                var (status, body) = UiCall(http, "POST", $"/api/runs/{id}/{action}");
                Check(status == HttpStatusCode.Conflict && ((string?)body?["error"])?.Contains("read-only", StringComparison.Ordinal) == true
                      && ((string?)body?["error"])?.Contains("on the machines that run it", StringComparison.Ordinal) == true, $"{action} is refused naming why: {body?.ToJsonString()}");
            }

            var listed = UiOk(http, "GET", "/api/runs")["runs"]!.AsArray().First(r => (string?)r!["id"] == id)!;
            Check((string?)listed["kind"] == "followed" && ((string?)listed["controls"]!["pause"])?.Contains("read-only", StringComparison.Ordinal) == true,
                $"the runs list gives the page the reason its controls are off: {listed.ToJsonString()}");
        }
        finally
        {
            http.Dispose();
            server.Dispose();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterUiShardedStop(Device device)
    {
        string folder = TempFolder("cluster-ui-sharded");
        try
        {
            string corpus = PretrainData("tokens.bin");
            var options = DpOptions() with { Collective = "ring", Shard = ShardMode.Parameters };
            PretrainingOptions Options(int steps, ICollectives cluster) =>
                DpPretraining(corpus, steps, cluster, options) with { Optimizer = "adamw", BatchTokens = 32, GradientAccumulation = 2 };
            string stopped = Path.Combine(folder, "stopped"), straight = Path.Combine(folder, "straight");
            var control = new ClusterRunControl();
            control.Pause();                                       // before step 0
            var running = Task.Run(() => InProcessCluster.Run(DpTopology(device, 2), options,
                rank => DpPretrain(rank.Device, Options(40, rank.Collectives), rank.Rank == 0 ? stopped : Path.Combine(folder, "stopped-1")), control));
            Check(UiWait(() => control.State == ClusterRunState.Paused, 120) && control.LastStep == -1, "both ranks wait before step 0");
            control.Resume();
            Check(UiWait(() => control.LastStep >= 1, 120), "two steps begun");
            long at = control.Pause();
            Check(UiWait(() => control.State == ClusterRunState.Paused, 120) && control.LastStep == at - 1, $"both ranks wait before step {at}");
            control.Resume();
            Check(UiWait(() => control.LastStep >= at + 1, 120), "the run goes on");
            long last = control.Stop();
            Check(running.Wait(TimeSpan.FromMinutes(3)), "the stopped run ends");
            var result = running.Result;
            Check(result.Errors.All(e => e is ClusterRunStoppedException s && s.Step == last + 1),
                $"every rank stopped as it began step {last + 1}: {string.Join("; ", result.Errors.Select(e => e?.Message ?? "finished"))}");
            int steps = (int)last + 1;
            string checkpoint = Path.Combine(stopped, $"checkpoint-{steps}");
            Check(Directory.Exists(checkpoint) && !Directory.Exists(Path.Combine(folder, "stopped-1", $"checkpoint-{steps}")), $"rank 0 alone wrote the stop's checkpoint ({checkpoint})");

            DpRun(DpTopology(device, 2), options, rank => DpPretrain(rank.Device, Options(steps, rank.Collectives), rank.Rank == 0 ? straight : Path.Combine(folder, "straight-1")));
            string whole = Path.Combine(straight, $"checkpoint-{steps}");
            foreach (string file in new[] { "model.safetensors", "optimizer.safetensors" })
            {
                Check(File.ReadAllBytes(Path.Combine(checkpoint, file)).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(whole, file))),
                    $"the stop's {file} is the uninterrupted run's after {steps} steps, byte for byte (every rank's share joined)");
            }

            Check(Pretrainer.ReadCheckpoint(checkpoint).Losses.SequenceEqual(Pretrainer.ReadCheckpoint(whole).Losses), "and its losses");
            Console.WriteLine($"    sharded (parameters): paused at step {at}, stopped after {steps} steps; the checkpoint as the uninterrupted run's");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
