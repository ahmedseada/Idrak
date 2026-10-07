// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Testing.Devices;

/// <summary>A storage argument of a recorded call: the index of its values in <see cref="RecordedCall.Before"/>.</summary>
internal sealed record Slot(int Index);

/// <summary>
/// One operation call as the reference device ran it: the operation, its arguments (each storage as a <see cref="Slot"/>),
/// the values of every storage it was given before and after, and what it returned.
/// </summary>
internal sealed class RecordedCall(string @case, Operation operation, object?[] arguments, float[][] before)
{
    public string Case { get; } = @case;

    public Operation Operation { get; } = operation;

    // The arguments, storages replaced by slots.
    public object?[] Arguments { get; } = arguments;

    // The values of each storage before the call, by slot.
    public float[][] Before { get; } = before;

    // The values of each storage after the call on the reference device, by slot.
    public float[][] After { get; set; } = [];

    // What the call returned (true or false for the operations that may have no kernel), or null.
    public object? Result { get; set; }

    // The call as made: its storages read now, replaced by slots.
    public static RecordedCall Of(string @case, Operation operation, object?[] arguments)
    {
        var storages = StoragesOf(arguments);
        var slots = Map(arguments, s => new Slot(Array.FindIndex(storages, f => ReferenceEquals(f, s))));
        return new RecordedCall(@case, operation, slots, Array.ConvertAll(storages, Read));
    }

    // Every storage among the arguments, each once (a storage passed twice is one buffer).
    public static Storage[] StoragesOf(object?[] arguments)
    {
        var found = new List<Storage>();
        void Add(object? value)
        {
            switch (value)
            {
                case Storage s when !found.Exists(f => ReferenceEquals(f, s)):
                    found.Add(s);
                    break;
                case object?[][] tuples:
                    foreach (var tuple in tuples)
                    {
                        Array.ForEach(tuple, Add);
                    }

                    break;
            }
        }

        Array.ForEach(arguments, Add);
        return [.. found];
    }

    // The arguments with each storage replaced through `map`.
    public static object?[] Map(object?[] arguments, Func<Storage, object> map) =>
        Array.ConvertAll(arguments, value => value switch
        {
            Storage s => map(s),
            object?[][] tuples => Array.ConvertAll(tuples, t => Map(t, map)),
            _ => value,
        });

    // The arguments with each slot replaced by a storage.
    public static object?[] Bind(object?[] arguments, Storage[] storages) =>
        Array.ConvertAll(arguments, value => value switch
        {
            Slot s => storages[s.Index],
            object?[][] tuples => Array.ConvertAll(tuples, t => Bind(t, storages)),
            _ => value,
        });

    // The values of a storage, read with its device's copy.
    public static float[] Read(Storage storage)
    {
        var values = new float[storage.Length];
        storage.Backend.Download(storage, values);
        return values;
    }
}

/// <summary>
/// Records the operation calls one thread makes on the reference device while it runs a case (<see cref="OperationCalls.Intercept"/>
/// reports every call here). Calls an operation makes inside another (a composed kernel) are not recorded: replaying
/// the outer call runs them on the device under test.
/// </summary>
internal sealed class CallRecorder
{
    [ThreadStatic]
    private static CallRecorder? _current;

    [ThreadStatic]
    private static int _depth;

    private readonly List<RecordedCall> _calls = [];

    // The storages of each call of the current case (to read them again in Rewrite), parallel to the case's calls.
    private readonly List<Storage[]> _storages = [];
    private int _caseStart;
    private string _case = "";

    public IReadOnlyList<RecordedCall> Calls => _calls;

    // Records the calls this thread makes until the scope is disposed, as made by `caseName`.
    public IDisposable Start(string caseName)
    {
        _case = caseName;
        _caseStart = _calls.Count;
        _storages.Clear();
        _current = this;
        _depth = 0;
        return new Scope();
    }

    // Called before an operation runs: a new call (with its storages read) when this thread records and the call is not
    // inside another; else null.
    public Pending? Enter(Backend backend, Operation operation, object?[] arguments)
    {
        _ = backend;
        if (!ReferenceEquals(_current, this) || _depth++ > 0)
        {
            return null;
        }

        return new Pending(RecordedCall.Of(_case, operation, arguments), RecordedCall.StoragesOf(arguments));
    }

    // Called with what the operation returned.
    public static void Returned(Pending? call, object? result)
    {
        if (call is not null)
        {
            call.Call.Result = result;
        }
    }

    // Called after the operation, whether it returned or threw: keeps the call with its storages read again.
    public void Exit(Pending? call)
    {
        if (!ReferenceEquals(_current, this))
        {
            return;
        }

        _depth--;
        if (call is not null)
        {
            call.Call.After = Array.ConvertAll(call.Storages, RecordedCall.Read);
            _calls.Add(call.Call);
            _storages.Add(call.Storages);
        }
    }

    // Drops the calls recorded from `count` on (those of a case that failed on the reference device).
    public void Truncate(int count) => _calls.RemoveRange(count, _calls.Count - count);

    // The calls recorded so far.
    public int Count => _calls.Count;

    // Call `index` of the current case (an operation the CPU had no kernel for, which returned false) as if it had run:
    // its storages read now, after a composition of other operations wrote what it must write.
    public void Rewrite(int index, Operation operation)
    {
        if (index < _caseStart || index >= _calls.Count || _calls[index].Operation != operation)
        {
            throw new InvalidOperationException($"No call of {operation} was recorded to compose.");
        }

        _calls[index].After = Array.ConvertAll(_storages[index - _caseStart], RecordedCall.Read);
        _calls[index].Result = true;
    }

    // A call being recorded, with the storages it was given.
    public sealed record Pending(RecordedCall Call, Storage[] Storages);

    private sealed class Scope : IDisposable
    {
        public void Dispose()
        {
            _current = null;
            _depth = 0;
        }
    }
}
