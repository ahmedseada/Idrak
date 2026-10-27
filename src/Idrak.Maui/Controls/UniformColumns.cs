// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Microsoft.Maui.Layouts;

namespace Idrak.Maui.Controls;

/// <summary>
/// Children in rows of <see cref="Columns"/> equal columns (All labs' cards two a row on a phone, its groups a column each on
/// a tablet), each row as tall as its tallest child measured at the column's width. It measures the height its rows take,
/// whatever height it is offered (a scroll view offers any), so the scroll view around it knows how far to scroll.
/// </summary>
public sealed class UniformColumns : Layout
{
    /// <summary>The columns of a row.</summary>
    public static readonly BindableProperty ColumnsProperty =
        BindableProperty.Create(nameof(Columns), typeof(int), typeof(UniformColumns), 1, propertyChanged: (b, _, _) => ((UniformColumns)b).InvalidateMeasure(),
            validateValue: (_, v) => (int)v >= 1);

    /// <summary>The space between two columns.</summary>
    public static readonly BindableProperty ColumnSpacingProperty =
        BindableProperty.Create(nameof(ColumnSpacing), typeof(double), typeof(UniformColumns), 0d, propertyChanged: (b, _, _) => ((UniformColumns)b).InvalidateMeasure());

    /// <summary>The space between two rows.</summary>
    public static readonly BindableProperty RowSpacingProperty =
        BindableProperty.Create(nameof(RowSpacing), typeof(double), typeof(UniformColumns), 0d, propertyChanged: (b, _, _) => ((UniformColumns)b).InvalidateMeasure());

    /// <summary>The columns of a row.</summary>
    public int Columns
    {
        get => (int)GetValue(ColumnsProperty);
        set => SetValue(ColumnsProperty, value);
    }

    /// <summary>The space between two columns.</summary>
    public double ColumnSpacing
    {
        get => (double)GetValue(ColumnSpacingProperty);
        set => SetValue(ColumnSpacingProperty, value);
    }

    /// <summary>The space between two rows.</summary>
    public double RowSpacing
    {
        get => (double)GetValue(RowSpacingProperty);
        set => SetValue(RowSpacingProperty, value);
    }

    /// <inheritdoc />
    protected override ILayoutManager CreateLayoutManager() => new Manager(this);

    private sealed class Manager(UniformColumns layout) : LayoutManager(layout)
    {
        public override Size Measure(double widthConstraint, double heightConstraint)
        {
            var padding = layout.Padding;
            int columns = Math.Max(1, layout.Columns);
            double inner = widthConstraint - padding.HorizontalThickness;
            double column = double.IsFinite(inner) ? Math.Max(0, (inner - (columns - 1) * layout.ColumnSpacing) / columns) : double.PositiveInfinity;
            double height = 0, widest = 0, row = 0;
            int index = 0, rows = 0;
            foreach (var child in Shown())
            {
                var size = child.Measure(column, double.PositiveInfinity);
                widest = Math.Max(widest, size.Width);
                row = Math.Max(row, size.Height);
                if (++index % columns == 0)
                {
                    height += row;
                    rows++;
                    row = 0;
                }
            }

            if (index % columns != 0)
            {
                height += row;
                rows++;
            }

            height += Math.Max(0, rows - 1) * layout.RowSpacing + padding.VerticalThickness;
            double width = double.IsFinite(widthConstraint) ? widthConstraint
                : Math.Min(index, columns) * widest + Math.Max(0, Math.Min(index, columns) - 1) * layout.ColumnSpacing + padding.HorizontalThickness;
            return new Size(width, height);
        }

        public override Size ArrangeChildren(Rect bounds)
        {
            var padding = layout.Padding;
            int columns = Math.Max(1, layout.Columns);
            double column = Math.Max(0, (bounds.Width - padding.HorizontalThickness - (columns - 1) * layout.ColumnSpacing) / columns);
            var shown = Shown().ToList();
            double y = bounds.Top + padding.Top;
            for (int start = 0; start < shown.Count; start += columns)
            {
                double row = 0;
                for (int i = start; i < Math.Min(start + columns, shown.Count); i++)
                {
                    row = Math.Max(row, shown[i].DesiredSize.Height);
                }

                for (int i = start; i < Math.Min(start + columns, shown.Count); i++)
                {
                    double x = bounds.Left + padding.Left + (i - start) * (column + layout.ColumnSpacing);
                    shown[i].Arrange(new Rect(x, y, column, row));
                }

                y += row + layout.RowSpacing;
            }

            return bounds.Size;
        }

        private IEnumerable<IView> Shown()
        {
            foreach (var child in layout)
            {
                if (child.Visibility != Visibility.Collapsed)
                {
                    yield return child;
                }
            }
        }
    }
}
