// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.ComponentModel;
using Idrak.Maui.Controls;

namespace Idrak.Maui.Markdown;

/// <summary>
/// A <see cref="MarkdownDocument"/> drawn: a label a paragraph (its runs as spans: bold, italic, code), a row a list item
/// with its marker, a code block with its language and Copy. A block changed in place redraws itself alone; the colours
/// and fonts are the theme's keys.
/// </summary>
public sealed class MarkdownView : ContentView
{
    /// <summary>The document shown.</summary>
    public static readonly BindableProperty DocumentProperty =
        BindableProperty.Create(nameof(Document), typeof(MarkdownDocument), typeof(MarkdownView), propertyChanged: (b, _, n) => ((MarkdownView)b).Show(n as MarkdownDocument));

    private readonly VerticalStackLayout _blocks = new() { Spacing = 8 };

    /// <summary>An empty view.</summary>
    public MarkdownView()
    {
        BindableLayout.SetItemTemplateSelector(_blocks, new BlockTemplates());
        Content = _blocks;
    }

    /// <summary>The document shown.</summary>
    public MarkdownDocument? Document
    {
        get => (MarkdownDocument?)GetValue(DocumentProperty);
        set => SetValue(DocumentProperty, value);
    }

    private void Show(MarkdownDocument? document) => BindableLayout.SetItemsSource(_blocks, document?.Blocks);

    /// <summary>The spans of <paramref name="runs"/>, set in the theme's fonts and colours.</summary>
    public static FormattedString Spans(IReadOnlyList<MarkdownRun> runs)
    {
        ArgumentNullException.ThrowIfNull(runs);
        var text = new FormattedString();
        foreach (var run in runs)
        {
            var span = new Span { Text = run.Text };
            if ((run.Style & MarkdownRunStyle.Code) != 0)
            {
                span.SetDynamicResource(Span.FontFamilyProperty, "IdrakFontMono");
                span.SetDynamicResource(Span.BackgroundColorProperty, "IdrakSurface3");
                span.SetDynamicResource(Span.TextColorProperty, "IdrakText");
            }
            else if ((run.Style & MarkdownRunStyle.Bold) != 0)
            {
                span.SetDynamicResource(Span.FontFamilyProperty, "IdrakFontBold");
                span.SetDynamicResource(Span.TextColorProperty, "IdrakText");
            }

            if ((run.Style & MarkdownRunStyle.Italic) != 0)
            {
                span.FontAttributes = FontAttributes.Italic;
            }

            text.Spans.Add(span);
        }

        return text;
    }

    private sealed class BlockTemplates : DataTemplateSelector
    {
        private readonly DataTemplate _paragraph = new(() => new ParagraphView());
        private readonly DataTemplate _list = new(() => new ListView());
        private readonly DataTemplate _code = new(() => new CodeView());

        protected override DataTemplate OnSelectTemplate(object item, BindableObject container) => item switch
        {
            MarkdownList => _list,
            MarkdownCode => _code,
            _ => _paragraph,
        };
    }

    // A paragraph or a heading: its spans rebuilt when its runs change.
    private sealed class ParagraphView : Label
    {
        private MarkdownParagraph? _paragraph;

        public ParagraphView() => SetDynamicResource(StyleProperty, "IdrakBody");

        protected override void OnBindingContextChanged()
        {
            base.OnBindingContextChanged();
            if (_paragraph is not null)
            {
                _paragraph.PropertyChanged -= OnChanged;
            }

            _paragraph = BindingContext as MarkdownParagraph;
            if (_paragraph is not null)
            {
                _paragraph.PropertyChanged += OnChanged;
                if (_paragraph.IsHeading)
                {
                    SetDynamicResource(StyleProperty, "IdrakHeading");
                }

                Draw();
            }
        }

        private void OnChanged(object? sender, PropertyChangedEventArgs e) => Draw();

        private void Draw()
        {
            if (_paragraph is { } p)
            {
                FormattedText = Spans(p.Runs);
                FlowDirection = p.RightToLeft ? FlowDirection.RightToLeft : FlowDirection.LeftToRight;
                HorizontalTextAlignment = TextAlignment.Start;
            }
        }
    }

    // A list: a row an item, its marker then its spans.
    private sealed class ListView : VerticalStackLayout
    {
        private MarkdownList? _list;

        public ListView() => Spacing = 3;

        protected override void OnBindingContextChanged()
        {
            base.OnBindingContextChanged();
            if (_list is not null)
            {
                _list.PropertyChanged -= OnChanged;
            }

            _list = BindingContext as MarkdownList;
            if (_list is not null)
            {
                _list.PropertyChanged += OnChanged;
                Draw();
            }
        }

        private void OnChanged(object? sender, PropertyChangedEventArgs e) => Draw();

        private void Draw()
        {
            var items = _list?.Items ?? [];
            while (Children.Count > items.Count)
            {
                Children.RemoveAt(Children.Count - 1);
            }

            for (int i = 0; i < items.Count; i++)
            {
                if (i >= Children.Count)
                {
                    Children.Add(Row());
                }

                var row = (Grid)Children[i];
                var item = items[i];
                ((Label)row.Children[0]).Text = item.Marker;
                var text = (Label)row.Children[1];
                text.FormattedText = Spans(item.Runs);
                row.FlowDirection = item.RightToLeft ? FlowDirection.RightToLeft : FlowDirection.LeftToRight;
            }
        }

        private static Grid Row()
        {
            var row = new Grid { ColumnDefinitions = [new ColumnDefinition(GridLength.Auto), new ColumnDefinition(GridLength.Star)], ColumnSpacing = 6 };
            var marker = new Label { MinimumWidthRequest = 16 };
            marker.SetDynamicResource(StyleProperty, "IdrakBody");
            var text = new Label();
            text.SetDynamicResource(StyleProperty, "IdrakBody");
            row.Add(marker, 0);
            row.Add(text, 1);
            return row;
        }
    }

    // A code block: its language and Copy above the code, in the mono font, wrapped.
    private sealed class CodeView : Border
    {
        private readonly Label _language = new();
        private readonly Label _code = new();
        private readonly Grid _copy = new() { ColumnDefinitions = [new ColumnDefinition(GridLength.Auto), new ColumnDefinition(GridLength.Auto)], ColumnSpacing = 4 };
        private readonly TapGestureRecognizer _tap = new();
        private MarkdownCode? _block;

        public CodeView()
        {
            SetDynamicResource(StyleProperty, "IdrakCodeBlock");
            _language.SetDynamicResource(StyleProperty, "IdrakMeta");
            _code.SetDynamicResource(StyleProperty, "IdrakMono");
            var copyLabel = new Label { Text = "Copy" };
            copyLabel.SetDynamicResource(StyleProperty, "IdrakMeta");
            _copy.Add(new IconView { Glyph = "copy", Size = 14 }, 0);
            _copy.Add(copyLabel, 1);
            ((IconView)_copy.Children[0]).SetDynamicResource(IconView.StrokeProperty, "IdrakTextFaint");
            _copy.GestureRecognizers.Add(_tap);
            SemanticProperties.SetDescription(_copy, "Copy the code");
            var header = new Grid { ColumnDefinitions = [new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Auto)], Padding = new Thickness(12, 7) };
            header.Add(_language, 0);
            header.Add(_copy, 1);
            var line = new BoxView { HeightRequest = 1 };
            line.SetDynamicResource(BoxView.ColorProperty, "IdrakLine");
            var body = new VerticalStackLayout { header, line, new ContentView { Padding = new Thickness(12, 9), Content = _code } };
            Content = body;
        }

        protected override void OnBindingContextChanged()
        {
            base.OnBindingContextChanged();
            if (_block is not null)
            {
                _block.PropertyChanged -= OnChanged;
            }

            _block = BindingContext as MarkdownCode;
            if (_block is not null)
            {
                _block.PropertyChanged += OnChanged;
                Draw();
            }
        }

        private void OnChanged(object? sender, PropertyChangedEventArgs e) => Draw();

        private void Draw()
        {
            if (_block is { } b)
            {
                _language.Text = b.LanguageTitle;
                _code.Text = b.Code;
                _tap.Command = b.CopyCommand;
                _tap.CommandParameter = b.Code;
                _copy.IsVisible = b.CopyCommand is not null;
            }
        }
    }
}
