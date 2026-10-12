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
// - Moves: one thread at a time; a move waits until no other thread is inside an operation. A thread that is inside an
//   operation itself, or holds a lock the operations it waits for may need (the caller says so with wait: false), never
//   waits: its move runs only when no other thread is inside an operation, else it does not run (the caller does without).
//   Moves by the moving thread nest.
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

    public bool BeginMove(bool wait)
    {
        int self = Environment.CurrentManagedThreadId;
        if (Volatile.Read(ref _mover) == self)
        {
            _moves++;
            return true;
        }

        int mine = Depth() > 0 ? 1 : 0;
        wait &= mine == 0;                                                // inside an operation: never wait for the others
        lock (_sync)
        {
            while (_mover != 0)
            {
                if (!wait)
                {
                    return false;
                }

                Monitor.Wait(_sync);
            }

            Interlocked.Exchange(ref _mover, self);                       // a full fence before counting the operations
            while (Volatile.Read(ref _inside) > mine)
            {
                if (!wait)
                {
                    Interlocked.Exchange(ref _mover, 0);
                    Monitor.PulseAll(_sync);
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
            Interlocked.Exchange(ref _mover, 0);
            Monitor.PulseAll(_sync);
        }
    }
}
