using System.Text;

namespace Idrak.Samples.Ocr;

/// <summary>
/// Reads and writes PGM (portable graymap) images, a trivial format every image editor can export
/// (GIMP, Photoshop, IrfanView, ImageMagick: <c>magick input.png output.pgm</c>).
/// </summary>
internal static class Pgm
{
    /// <summary>Loads P2 (text) or P5 (binary) PGM as ink intensities in [0, 1]; light backgrounds are inverted.</summary>
    public static (float[] Pixels, int Width, int Height) Read(string path)
    {
        var bytes = File.ReadAllBytes(path);
        int position = 0;
        string Token()
        {
            while (position < bytes.Length && (char.IsWhiteSpace((char)bytes[position]) || bytes[position] == '#'))
            {
                if (bytes[position] == '#')
                {
                    while (position < bytes.Length && bytes[position] != '\n')
                    {
                        position++;
                    }
                }

                position++;
            }

            int start = position;
            while (position < bytes.Length && !char.IsWhiteSpace((char)bytes[position]))
            {
                position++;
            }

            return Encoding.ASCII.GetString(bytes, start, position - start);
        }

        string magic = Token();
        if (magic is not ("P2" or "P5"))
        {
            throw new InvalidDataException($"{path} is not a PGM image (expected P2 or P5, found '{magic}').");
        }

        int width = int.Parse(Token()), height = int.Parse(Token()), max = int.Parse(Token());
        var pixels = new float[width * height];
        if (magic == "P5")
        {
            position++; // the single whitespace after maxval
            int bytesPerSample = max > 255 ? 2 : 1;
            for (int i = 0; i < pixels.Length; i++)
            {
                int value = bytesPerSample == 1 ? bytes[position + i] : (bytes[position + 2 * i] << 8) | bytes[position + 2 * i + 1];
                pixels[i] = value / (float)max;
            }
        }
        else
        {
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = int.Parse(Token()) / (float)max;
            }
        }

        // Ink should be bright: invert dark-on-light pages.
        if (pixels.Average() > 0.5f)
        {
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = 1f - pixels[i];
            }
        }

        return (pixels, width, height);
    }

    /// <summary>Writes the image as a binary PGM, dark ink on a white page.</summary>
    public static void Write(string path, float[] pixels, int width, int height)
    {
        using var stream = File.Create(path);
        stream.Write(Encoding.ASCII.GetBytes($"P5\n{width} {height}\n255\n"));
        foreach (float v in pixels)
        {
            stream.WriteByte((byte)(255 - Math.Clamp((int)(v * 255), 0, 255)));
        }
    }
}
