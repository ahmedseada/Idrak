namespace Idrak.Optimizers;

/// <summary>
/// AdamW with 8-bit moments (Dettmers et al., 2022; bitsandbytes' AdamW8bit): each moment is stored as one byte per
/// parameter, a code into a dynamic (logarithmically spaced) table scaled per block of <see cref="BlockSize"/> elements
/// by the block's largest magnitude. The optimizer state takes 2 bytes per parameter instead of 8, which is what lets a
/// billion-parameter model train on a 16 GB GPU; training curves match 32-bit AdamW closely. Parameters with fewer than
/// <see cref="MinimumSize"/> elements (norms, biases) keep float32 moments, as bitsandbytes does.
/// </summary>
public sealed class AdamW8Bit : Optimizer
{
    /// <summary>Elements per quantization block (one scale per moment per block).</summary>
    public const int BlockSize = 256;

    private readonly Tensor?[] _m, _v, _scales;
    private Tensor? _map;
    private int _step;

    /// <summary>Creates the optimizer.</summary>
    public AdamW8Bit(IEnumerable<Tensor> parameters, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f,
        float weightDecay = 0.01f, int minimumSize = 4096)
        : base(parameters, learningRate)
    {
        (Beta1, Beta2, Epsilon, WeightDecay, MinimumSize) = (beta1, beta2, epsilon, weightDecay, minimumSize);
        _m = new Tensor?[Parameters.Count];
        _v = new Tensor?[Parameters.Count];
        _scales = new Tensor?[Parameters.Count];
    }

    /// <summary>Decay rate of the first-moment estimate.</summary>
    public float Beta1 { get; }

    /// <summary>Decay rate of the second-moment estimate.</summary>
    public float Beta2 { get; }

    /// <summary>Term added to the denominator.</summary>
    public float Epsilon { get; }

    /// <summary>Decoupled weight decay: p -= lr·λ·p each step.</summary>
    public float WeightDecay { get; }

    /// <summary>Parameters smaller than this keep float32 moments.</summary>
    public int MinimumSize { get; }

    /// <summary>Steps taken (for the bias corrections).</summary>
    public int StepCount => _step;

    /// <summary>Bytes the optimizer state occupies (moments and scales).</summary>
    public long StateBytes =>
        _m.Zip(_v).Sum(p => (long)((p.First?.Size ?? 0) + (p.Second?.Size ?? 0)) * 4) + _scales.Sum(s => (long)(s?.Size ?? 0) * 4);

    /// <inheritdoc />
    protected override bool ScalesGradientsInStep => true;

    /// <inheritdoc />
    public override void Step()
    {
        float scale = GradientScale;
        GradientScale = 1f;
        _step++;
        float correctedLr = (float)(LearningRate * Math.Sqrt(1 - Math.Pow(Beta2, _step)) / (1 - Math.Pow(Beta1, _step)));
        for (int i = 0; i < Parameters.Count; i++)
        {
            var p = Parameters[i];
            if (p.Grad is null)
            {
                continue;
            }

            float decay = 1f - LearningRate * WeightDecay;
            if (p.Size < MinimumSize)
            {
                if (decay != 1f)
                {
                    p.Backend.Affine(p.Storage, p.Storage, p.Size, decay, 0f);
                }

                if (scale != 1f)
                {
                    p.Backend.Affine(p.Grad.Storage, p.Grad.Storage, p.Size, scale, 0f);
                }

                var m = _m[i] ??= CreateState(p);
                var v = _v[i] ??= CreateState(p);
                p.Backend.AdamStep(p.Storage, p.Grad.Storage, m.Storage, v.Storage, p.Size, correctedLr, Beta1, Beta2, Epsilon);
                continue;
            }

            int blocks = (p.Size + BlockSize - 1) / BlockSize;
            var m8 = _m[i] ??= Tensor.PersistentZeros([(p.Size + 3) / 4], p.Device);
            var v8 = _v[i] ??= Tensor.PersistentZeros([(p.Size + 3) / 4], p.Device);
            var scales = _scales[i] ??= Tensor.PersistentZeros([2 * blocks], p.Device);
            _map ??= Tensor.Persistent([.. DynamicMap(signed: true), .. DynamicMap(signed: false)], [512], p.Device, requiresGrad: false);
            p.Backend.AdamStep8Bit(p.Storage, p.Grad.Storage, m8.Storage, v8.Storage, scales.Storage, _map.Storage, p.Size,
                correctedLr, Beta1, Beta2, Epsilon, scale, decay);
        }
    }

    /// <summary>
    /// The 256 codes of bitsandbytes' dynamic quantization map (sorted, in [-1, 1] when signed, [0, 1] when not):
    /// for each power of ten from 1e-6 to 1, evenly spaced fractions between 0.1 and 1 (more for larger powers), plus 0 and 1.
    /// </summary>
    public static float[] DynamicMap(bool signed)
    {
        const int ExponentBits = 7, NonSignBits = 7;
        var data = new List<float>();
        for (int i = 0; i < ExponentBits; i++)
        {
            int items = signed ? (1 << (i + NonSignBits - ExponentBits)) + 1 : (1 << (i + NonSignBits - ExponentBits + 1)) + 1;
            float power = MathF.Pow(10f, -(ExponentBits - 1) + i);
            for (int j = 0; j + 1 < items; j++)
            {
                // Midpoints of linspace(0.1, 1, items), in float32 as torch computes them.
                float lo = 0.1f + (0.9f * j / (items - 1)), hi = 0.1f + (0.9f * (j + 1) / (items - 1));
                float mean = (lo + hi) / 2f;
                data.Add(power * mean);
                if (signed)
                {
                    data.Add(-power * mean);
                }
            }
        }

        data.Add(0f);
        data.Add(1f);
        while (data.Count < 256)
        {
            data.Add(0f);
        }

        data.Sort();
        return [.. data];
    }

    /// <summary>The index of the code in the sorted <paramref name="map"/> nearest to <paramref name="x"/> (as the GPU kernel finds it).</summary>
    public static byte Nearest(ReadOnlySpan<float> map, float x)
    {
        int lo = 0;
        for (int step = 128; step > 0; step >>= 1)
        {
            if (map[lo + step] <= x)
            {
                lo += step;
            }
        }

        if (lo < 255 && map[lo + 1] - x < x - map[lo])
        {
            lo++;
        }

        return (byte)lo;
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var t in _m.Concat(_v).Concat(_scales))
        {
            t?.Dispose();
        }

        _map?.Dispose();
        base.Dispose();
    }
}
