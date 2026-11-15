// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Reflection;

namespace Idrak.Cli.Commands.Developer.Stacks;

/// <summary>
/// Runs a command in a child process with the .NET runtime's own tracing on (DOTNET_EnableEventPipe and its settings, read
/// at the runtime's start: no diagnostics client), so the runtime writes the .nettrace file itself: the sample profiler's
/// stacks of every thread (about each millisecond, as the system's timer allows), the GC's suspensions and allocation
/// reports, and the methods compiled (with the end-of-run listing of every method's code).
/// </summary>
internal static class StackRecorder
{
    /// <summary>
    /// The providers asked: the sample profiler; the runtime's GC (0x1: collections, suspensions, allocation reports) and JIT
    /// (0x10: each method compiled, with its code's address) events at the verbose level.
    /// </summary>
    public const string Providers = "Microsoft-DotNETCore-SampleProfiler:0:5,Microsoft-Windows-DotNETRuntime:0x11:5";

    /// <summary>How the child is started for a command's arguments: this program again (the tests start their own runner's CLI mode).</summary>
    internal static Func<IReadOnlyList<string>, ProcessStartInfo> Launcher { get; set; } = ThisProgram;

    /// <summary>
    /// Runs <paramref name="info"/> with tracing into <paramref name="path"/>; its output is the console's where both writers
    /// are the console's, else copied into <paramref name="output"/> and <paramref name="error"/> as it comes. Returns its exit code.
    /// </summary>
    public static int Record(ProcessStartInfo info, string path, TextWriter output, TextWriter error)
    {
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path)!);
        File.Delete(path);
        info.UseShellExecute = false;
        info.Environment["DOTNET_EnableEventPipe"] = "1";
        info.Environment["DOTNET_EventPipeOutputPath"] = path;
        info.Environment["DOTNET_EventPipeConfig"] = Providers;
        info.Environment["DOTNET_EventPipeOutputStreaming"] = "1";              // written as it comes: a run stopped early keeps what it traced
        bool console = ReferenceEquals(output, Console.Out) && ReferenceEquals(error, Console.Error);
        if (!console)
        {
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;
            info.StandardOutputEncoding = Console.OutputEncoding;               // the child writes in the console's encoding
            info.StandardErrorEncoding = Console.OutputEncoding;
        }

        using var process = Process.Start(info) ?? throw new InvalidOperationException($"{info.FileName} did not start.");
        Task[] pumps = console ? [] : [Pump(process.StandardOutput, output), Pump(process.StandardError, error)];
        process.WaitForExit();
        Task.WaitAll(pumps);
        if (!File.Exists(path))
        {
            throw new InvalidOperationException($"The run wrote no trace at {path}: the .NET runtime did not start its tracing (DOTNET_EnableEventPipe).");
        }

        return process.ExitCode;
    }

    // This program with the command's arguments (under the dotnet host, the CLI's assembly first).
    internal static ProcessStartInfo ThisProgram(IReadOnlyList<string> args)
    {
        string program = Environment.ProcessPath ?? throw new InvalidOperationException("This process has no program path to start again.");
        var info = new ProcessStartInfo(program);
        if (System.IO.Path.GetFileNameWithoutExtension(program).Equals("dotnet", StringComparison.OrdinalIgnoreCase) && Assembly.GetEntryAssembly()?.Location is { Length: > 0 } entry)
        {
            info.ArgumentList.Add(entry);
        }

        foreach (string a in args)
        {
            info.ArgumentList.Add(a);
        }

        return info;
    }

    private static Task Pump(StreamReader from, TextWriter to) => Task.Run(() =>
    {
        var buffer = new char[4096];
        int count;
        while ((count = from.Read(buffer, 0, buffer.Length)) > 0)
        {
            lock (to)
            {
                to.Write(buffer, 0, count);
                to.Flush();
            }
        }
    });
}
