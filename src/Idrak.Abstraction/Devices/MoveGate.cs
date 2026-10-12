// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

// Moves of live storages between memories (offloading, lookup tables placed in system memory) against the operations that
// read their addresses (Backend.GuardMoves). An operation reads a storage's address and queues work that uses it; a move
// copies the storage and points it at the new block, and the old block goes back to a pool. A move between another
// thread's read and its queued work would leave that work reading (or writing) a block that is someone else's by then.
//
// - Operations: each thread counts once while inside any operation of the device (nested operations, and an operation
//   inside a move by the same thread, do not count again or wait). A thread starting an operation while a move runs or
//   waits steps back until the move ends.
// - Moves: one thread at a time; a move waits until no other thread is inside an operation. A caller holding a lock the
//   operations it waits for may need says so (wait: false): its move runs only when no other thread is inside an
//   operation, else it does not run (the caller does without). Moves by the moving thread nest.
// - From inside an operation: a move of any storage never waits (the operation may hold addresses of what it moves). A
//   placement of lookup tables (an allocation that found the device full; lookupTablesOnly) waits like any move: no
//   kernel reads a lookup table's address before an allocation of its own, so while another thread places lookup tables
//   this one steps aside (no longer counted, and counted again before the placement ends, so no other move runs in
//   between) and then tries its allocation again; and a general move still waiting for this thread gives way to it (that
//   move does not run; its caller does without, as when the device is busy).
//
// Lock order: the gate before any lock of the device (its stream, its pool), as operations take it before the device's
// own locks; a move that waits holds none of them.
internal sealed class MoveGate
{
    private static int s_gates;

    // Each thread's operation depth per gate, by the gate's number (a device has one gate, so the array stays small).
    [ThreadStatic]
    private static int[]? t_depths;

    private readonly int _id = Interlocked.Increment(ref s_gates) - 1;
    private readonly object _sync = new();
    private int _inside;                                                  // threads inside an operation
    private int _mover;                                                   // the moving thread's managed id; 0 when none
    private int _moves;                                                   // the moving thread's nested moves
    private bool _tables;                                                 // the move places lookup tables only
    private bool _yield;                                                  // a placement inside an operation waits for the general move
    private int _parked;                                                  // threads inside operations stepped aside for a placement
    private long _ended;                                                  // moves ended so far (under the lock)

    private ref int Depth()
    {
        var depths = t_depths;
        if (depths is null || depths.Length <= _id)
        {
            Array.Resize(ref t_depths, Math.Max(_id + 1, 4));
            depths = t_depths!;
        }

        return ref depths[_id];
    }

    /// <summary>Whether this thread is inside an operation or a move of the gate's device (the addresses it reads stay valid).</summary>
    public bool Holds => Depth() > 0 || Volatile.Read(ref _mover) == Environment.CurrentManagedThreadId;

    public void Enter()
    {
        ref int depth = ref Depth();
        if (depth++ > 0)
        {
            return;
        }

        int self = Environment.CurrentManagedThreadId;
        while (true)
        {
            Interlocked.Increment(ref _inside);                           // a full fence: the mover sees it, or this sees the mover
            int mover = Volatile.Read(ref _mover);
            if (mover == 0 || mover == self)
            {
                return;
            }

            Interlocked.Decrement(ref _inside);                           // a move runs or waits: step back until it ends
            lock (_sync)
            {
                Monitor.PulseAll(_sync);
                while (_mover != 0 && _mover != self)
                {
                    Monitor.Wait(_sync);
                }
            }
        }
    }

    public void Exit()
    {
        ref int depth = ref Depth();
        if (--depth > 0)
        {
            return;
        }

        Interlocked.Decrement(ref _inside);
        if (Volatile.Read(ref _mover) != 0)
        {
            lock (_sync)
            {
                Monitor.PulseAll(_sync);                                  // a waiting move counts again
            }
        }
    }

    public bool BeginMove(bool wait, bool lookupTablesOnly)
    {
        int self = Environment.CurrentManagedThreadId;
        if (Volatile.Read(ref _mover) == self)
        {
            _moves++;
            return true;
        }

        int mine = Depth() > 0 ? 1 : 0;
        wait &= mine == 0 || lookupTablesOnly;                            // inside an operation only a placement waits
        lock (_sync)
        {
            while (_mover != 0)
            {
                if (!wait)
                {
                    return false;
                }

                if (mine > 0)
                {
                    if (_tables)
                    {
                        // Another thread places lookup tables: step aside until it ends (counted again by then); the
                        // caller tries again, with the tables gone.
                        long ended = _ended;
                        Interlocked.Decrement(ref _inside);
                        _parked++;
                        Monitor.PulseAll(_sync);
                        while (_ended == ended)
                        {
                            Monitor.Wait(_sync);
                        }

                        return false;
                    }

                    _yield = true;                                        // the general move waiting for this thread gives way
                    Monitor.PulseAll(_sync);
                }

                Monitor.Wait(_sync);
            }

            Interlocked.Exchange(ref _mover, self);                       // a full fence before counting the operations
            (_tables, _yield) = (lookupTablesOnly, false);
            while (Volatile.Read(ref _inside) > mine)
            {
                if (!wait || (_yield && !lookupTablesOnly))
                {
                    EndLocked();                                          // not now (or giving way to a placement)
                    return false;
                }

                Monitor.Wait(_sync);
            }

            _moves = 1;
            return true;
        }
    }

    public void EndMove()
    {
        if (--_moves > 0)
        {
            return;
        }

        lock (_sync)
        {
            EndLocked();
        }
    }

    // Under the lock: the threads that stepped aside count again before any other move can start, then the move ends.
    private void EndLocked()
    {
        Interlocked.Add(ref _inside, _parked);
        (_parked, _tables, _yield) = (0, false, false);
        _ended++;
        Interlocked.Exchange(ref _mover, 0);
        Monitor.PulseAll(_sync);
    }
}
