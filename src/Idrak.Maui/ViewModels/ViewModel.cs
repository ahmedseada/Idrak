// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Idrak.Maui.ViewModels;

/// <summary>A view model's change notifications.</summary>
public abstract class ViewModel : INotifyPropertyChanged
{
    /// <inheritdoc />
    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>Sets <paramref name="field"/> and notifies when the value changes; true when it did.</summary>
    protected bool SetProperty<T>(ref T field, T value, [CallerMemberName] string name = "")
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        OnPropertyChanged(name);
        return true;
    }

    /// <summary>Notifies that <paramref name="name"/> changed.</summary>
    protected void OnPropertyChanged([CallerMemberName] string name = "") => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

/// <summary>One line of a card: a name and its value.</summary>
/// <param name="Name">What it is ("memory").</param>
/// <param name="Value">Its value.</param>
public sealed record InfoRow(string Name, string Value);

/// <summary>A card of facts on a page.</summary>
/// <param name="Title">Its heading ("vulkan:0").</param>
/// <param name="Subtitle">A line under it (the device's name), or "".</param>
/// <param name="Rows">Its lines.</param>
/// <param name="Problem">What went wrong (a device that could not start), or "".</param>
public sealed record InfoCard(string Title, string Subtitle, IReadOnlyList<InfoRow> Rows, string Problem = "")
{
    /// <summary>Whether there is a problem to show.</summary>
    public bool HasProblem => Problem.Length > 0;
}
