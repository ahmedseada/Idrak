// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Theme;

/// <summary>Idrak's sizes, spacing, radii and styles, merged by the app (see <see cref="IdrakThemes"/>).</summary>
public partial class IdrakStyles : ResourceDictionary
{
    /// <summary>
    /// The styles a desktop's pointer hovers and the visual states each takes (IdrakHover* in IdrakStyles.xaml): the buttons
    /// and chips, the lab cards, the outlined and round buttons, the rail's tabs. The implicit Button style is named by its type.
    /// </summary>
    public static IReadOnlyList<(string Style, string States)> Hovered { get; } =
    [
        (typeof(Button).FullName!, "IdrakHoverAccentButton"), ("IdrakGradientButton", "IdrakHoverAccentButton"), ("IdrakWideGradientButton", "IdrakHoverAccentButton"),
        ("IdrakChip", "IdrakHoverButton"), ("IdrakSecondaryButton", "IdrakHoverButton"), ("IdrakGhostButton", "IdrakHoverButton"), ("IdrakDangerButton", "IdrakHoverButton"),
        ("IdrakLabCard", "IdrakHoverCard"),
        ("IdrakChipBorder", "IdrakHoverOutline"), ("IdrakSuggestion", "IdrakHoverOutline"), ("IdrakConversation", "IdrakHoverOutline"),
        ("IdrakIconButton", "IdrakHoverOutline"), ("IdrakSideButton", "IdrakHoverOutline"),
        ("IdrakRoundButton", "IdrakHoverRound"), ("IdrakRoundButtonPlain", "IdrakHoverRound"), ("IdrakGradientIconButton", "IdrakHoverRound"),
        ("IdrakRailPill", "IdrakHoverRail"),
    ];

    /// <summary>The styles, with the pointer's hover states on a desktop (<see cref="IdrakPlatform.IsDesktop"/>).</summary>
    public IdrakStyles()
        : this(IdrakPlatform.IsDesktop)
    {
    }

    /// <summary>The styles; <paramref name="pointerHover"/> gives the tapped ones their hover states (<see cref="Hovered"/>), once, here.</summary>
    public IdrakStyles(bool pointerHover)
    {
        InitializeComponent();
        PointerHover = pointerHover;
        if (pointerHover)
        {
            foreach (var (style, states) in Hovered)
            {
                // A style's visual states are copied for each view it styles (MAUI's Setter clones the list).
                ((Style)this[style]).Setters.Add(new Setter { Property = VisualStateManager.VisualStateGroupsProperty, Value = this[states] });
            }
        }
    }

    /// <summary>Whether the tapped styles have the pointer's hover states.</summary>
    public bool PointerHover { get; }
}
