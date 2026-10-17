// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// The unconstrained rational-quadratic spline of neural spline flows (Durkan et al., 2019; transformers'
/// <c>modeling_vits</c>): the identity outside ±tail, inside it a monotone piecewise rational-quadratic map over bins
/// whose widths, heights and inner derivatives a network gives. <see cref="Forward"/> runs it from tensor operations
/// (training); <see cref="Inverse"/> on the host in double (inference), so the inverse matches the float reference
/// within its tolerance.
/// </summary>
public static class RationalQuadraticSpline
{
    private const double MinBinWidth = 1e-3, MinBinHeight = 1e-3, MinDerivative = 1e-3;

    /// <summary>
    /// The spline forward (reverse false): the inputs <paramref name="x"/> [N, 1] through the splines of the unnormalized
    /// widths and heights [N, bins] and inner derivatives [N, bins − 1], and the log of each one's derivative, from tensor
    /// operations so their gradients reach every input: the widths' softmax cumulated by a product with a triangle of
    /// ones, each input's bin found on the host (from the cumulative widths) and its values picked by one-hot rows. Inputs
    /// outside ±<paramref name="tail"/> pass unchanged (their log-derivative 0), and the spline is evaluated at 0 for them so
    /// nothing there is not finite.
    /// </summary>
    public static (Tensor Outputs, Tensor LogAbsDet) Forward(Tensor x, Tensor widths, Tensor heights, Tensor derivatives, float tail)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(widths);
        int n = widths.Shape[0], bins = widths.Shape[1];
        var device = x.Device;
        var inputs = x.ToArray();
        var inside = new float[n];
        var outside = new float[n];
        for (int i = 0; i < n; i++)
        {
            bool within = inputs[i] >= -tail && inputs[i] <= tail;
            (inside[i], outside[i]) = within ? (1f, 0f) : (0f, 1f);
        }

        var keep = Tensor.From(inside, [n, 1], device);
        var pass = Tensor.From(outside, [n, 1], device);
        var at = x * keep;                                                        // outside inputs read at 0, a point inside
        var (cumWidths, binWidths) = Cumulative(widths, (float)MinBinWidth, tail);
        var (cumHeights, binHeights) = Cumulative(heights, (float)MinBinHeight, tail);
        var ones = Tensor.Ones([n, 1], device);
        var inner = derivatives.Relu() + ((derivatives.Abs() * -1f).Exp() + 1f).Log() + (float)MinDerivative;   // min + softplus
        var slopes = Tensor.Concat([ones, inner, ones], 1);                       // [N, bins + 1]; the edge derivatives are 1 (softplus of the constant + min)

        // Each input's bin: the cumulative widths it reaches (the top one nudged up), counted on the host.
        var cum = cumWidths.ToArray();
        var pickBin = new float[n * bins];
        var pickEdge = new float[n * (bins + 1)];
        var pickNext = new float[n * (bins + 1)];
        for (int i = 0; i < n; i++)
        {
            float v = inside[i] == 1f ? inputs[i] : 0f;
            int bin = -1;
            for (int k = 0; k <= bins; k++)
            {
                float edge = cum[i * (bins + 1) + k] + (k == bins ? 1e-6f : 0f);
                if (v >= edge)
                {
                    bin++;
                }
            }

            bin = Math.Clamp(bin, 0, bins - 1);
            pickBin[i * bins + bin] = 1f;
            pickEdge[i * (bins + 1) + bin] = 1f;
            pickNext[i * (bins + 1) + bin + 1] = 1f;
        }

        var inBin = Tensor.From(pickBin, [n, bins], device);
        var atEdge = Tensor.From(pickEdge, [n, bins + 1], device);
        var atNext = Tensor.From(pickNext, [n, bins + 1], device);
        Tensor Pick(Tensor values, Tensor rows) => (values * rows).Sum(1, keepDim: true);   // [N, 1]

        var inputCumWidth = Pick(cumWidths, atEdge);
        var inputBinWidth = Pick(binWidths, inBin);
        var inputCumHeight = Pick(cumHeights, atEdge);
        var inputHeight = Pick(binHeights, inBin);
        var inputDelta = Pick(binHeights * binWidths.Pow(-1f), inBin);
        var d0 = Pick(slopes, atEdge);
        var d1 = Pick(slopes, atNext);

        var theta = (at - inputCumWidth) * inputBinWidth.Pow(-1f);
        var rest = theta * -1f + 1f;
        var thetaRest = theta * rest;
        var numerator = inputHeight * (inputDelta * theta.Square() + d0 * thetaRest);
        var denominator = inputDelta + (d0 + d1 - inputDelta * 2f) * thetaRest;
        var outputs = inputCumHeight + numerator * denominator.Pow(-1f);
        var derivativeNumerator = inputDelta.Square() * (d1 * theta.Square() + inputDelta * 2f * thetaRest + d0 * rest.Square());
        var logAbsDet = derivativeNumerator.Log() - denominator.Log() * 2f;
        return (outputs * keep + x * pass, logAbsDet * keep);
    }

    // softmax, the minimum added, cumulated (a product with the upper triangle of ones), stretched to [-tail, tail] with
    // exact ends: the cumulative edges [N, bins + 1] and the bins' sizes [N, bins].
    private static (Tensor Cumulative, Tensor Sizes) Cumulative(Tensor unnormalized, float minimum, float tail)
    {
        int n = unnormalized.Shape[0], bins = unnormalized.Shape[1];
        var device = unnormalized.Device;
        var sizes = unnormalized.Softmax() * (1f - minimum * bins) + minimum;
        var triangle = new float[bins * (bins - 1)];
        for (int i = 0; i < bins; i++)
        {
            for (int j = i; j < bins - 1; j++)
            {
                triangle[i * (bins - 1) + j] = 1f;
            }
        }

        var running = sizes.MatMul(Tensor.From(triangle, [bins, bins - 1], device));   // [N, bins - 1]: the sums up to each inner edge
        var lower = Tensor.From(Enumerable.Repeat(-tail, n).ToArray(), [n, 1], device);
        var upper = Tensor.From(Enumerable.Repeat(tail, n).ToArray(), [n, 1], device);
        var cumulative = Tensor.Concat([lower, running * (2f * tail) - tail, upper], 1);
        return (cumulative, cumulative.Narrow(1, 1, bins) - cumulative.Narrow(1, 0, bins));
    }

    /// <summary>
    /// The input whose spline value is <paramref name="y"/>. <paramref name="derivatives"/> holds bins + 1 values whose
    /// first and last are overwritten with the constant that makes the edge derivatives 1.
    /// </summary>
    /// <exception cref="InvalidDataException">The quadratic has no real root.</exception>
    public static double Inverse(double y, ReadOnlySpan<double> unnormalizedWidths, ReadOnlySpan<double> unnormalizedHeights, Span<double> derivatives, double tail)
    {
        if (y < -tail || y > tail)
        {
            return y;
        }

        int bins = unnormalizedWidths.Length;
        double constant = Math.Log(Math.Exp(1 - MinDerivative) - 1);
        derivatives[0] = constant;
        derivatives[bins] = constant;

        Span<double> cumWidths = stackalloc double[bins + 1];
        Span<double> cumHeights = stackalloc double[bins + 1];
        Cumulative(unnormalizedWidths, MinBinWidth, tail, cumWidths);
        Cumulative(unnormalizedHeights, MinBinHeight, tail, cumHeights);

        // The bin: the last cumulative height the input reaches (the top edge nudged so the bound falls inside the last bin).
        cumHeights[bins] += 1e-6;
        int bin = -1;
        for (int k = 0; k <= bins; k++)
        {
            if (y >= cumHeights[k])
            {
                bin++;
            }
        }

        bin = Math.Clamp(bin, 0, bins - 1);
        cumHeights[bins] -= 1e-6;
        double width = cumWidths[bin + 1] - cumWidths[bin], height = cumHeights[bin + 1] - cumHeights[bin];
        double delta = height / width;
        double d0 = MinDerivative + Softplus(derivatives[bin]), d1 = MinDerivative + Softplus(derivatives[bin + 1]);
        double intermediate1 = d0 + d1 - 2 * delta;
        double intermediate2 = y - cumHeights[bin];
        double intermediate3 = intermediate2 * intermediate1;
        double a = height * (delta - d0) + intermediate3;
        double b = height * d0 - intermediate3;
        double c = -delta * intermediate2;
        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0)
        {
            throw new InvalidDataException($"The spline has no inverse at {y} (discriminant {discriminant}).");
        }

        double root = 2 * c / (-b - Math.Sqrt(discriminant));
        return root * width + cumWidths[bin];
    }

    // softmax, the minimum width added, cumulated from 0, stretched to [-tail, tail] with exact ends.
    private static void Cumulative(ReadOnlySpan<double> unnormalized, double minimum, double tail, Span<double> cumulative)
    {
        int bins = unnormalized.Length;
        double max = double.NegativeInfinity;
        foreach (double v in unnormalized)
        {
            max = Math.Max(max, v);
        }

        double sum = 0;
        for (int k = 0; k < bins; k++)
        {
            sum += Math.Exp(unnormalized[k] - max);
        }

        cumulative[0] = -tail;
        double running = 0;
        for (int k = 0; k < bins; k++)
        {
            running += minimum + (1 - minimum * bins) * (Math.Exp(unnormalized[k] - max) / sum);
            cumulative[k + 1] = 2 * tail * running - tail;
        }

        cumulative[bins] = tail;
    }

    private static double Softplus(double x) => x > 20 ? x : Math.Log(1 + Math.Exp(x));
}
