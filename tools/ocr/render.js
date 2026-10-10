// Lays out text as document pages on a canvas (the browser shapes the script: joining, ligatures, right to left) and
// prints every page and every line as base64, for render-text.ps1 to save. Input: window.ocrBatch =
// { docs: [{ paras: [...], cont: bool }], seed, fonts, degrade, lines }.
// Output lines in <pre id="out">: "P\t<page>\t<text b64>\t<jpeg data URL>" and "L\t<page>\t<line>\t<text b64>\t<png data URL>".
(function () {
    const job = window.ocrBatch;
    let state = job.seed >>> 0;
    const random = () => {
        state = (state + 0x6D2B79F5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const between = (a, b) => a + (b - a) * random();
    const pick = list => list[Math.floor(random() * list.length)];
    const utf8 = s => {
        let bytes = "";
        for (const b of new TextEncoder().encode(s)) bytes += String.fromCharCode(b);
        return btoa(bytes);
    };

    // A font is installed when text drawn with it measures differently from both generic fallbacks.
    const probe = document.createElement("canvas").getContext("2d");
    const sample = "بسم الله الرحمن الرحيم 0123 المادة";
    const width = font => { probe.font = font; return probe.measureText(sample).width; };
    const fonts = job.fonts.filter(f =>
        width(`40px "${f}", monospace`) !== width("40px monospace") || width(`40px "${f}", serif`) !== width("40px serif"));
    const out = [];
    if (fonts.length === 0) {
        out.push("E\tnone of these fonts is installed: " + job.fonts.join(", "));
        document.getElementById("out").textContent = out.join("\n");
        return;
    }

    let doc = 0, para = 0, word = 0, page = 0, docPage = 0;
    const words = () => job.docs[doc].paras[para].split(" ");
    const done = () => doc >= job.docs.length;
    const advance = () => {
        para++; word = 0;
        if (para >= job.docs[doc].paras.length) { doc++; para = 0; return true; }
        return false;
    };

    while (!done()) {
        docPage = para === 0 && word === 0 && !job.docs[doc].cont ? 1 : docPage + 1;
        const W = Math.round(between(1100, 1400)), H = Math.round(W * 1.414);
        const margin = Math.round(W * between(0.07, 0.11));
        const size = Math.round(W * between(0.016, 0.025));
        const spacing = between(1.45, 2.0);
        const family = `"${pick(fonts)}"`;
        const ink = Math.round(between(0, 45)), paper = Math.round(between(238, 255));
        const canvas = document.createElement("canvas");
        canvas.width = W; canvas.height = H;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = `rgb(${paper},${paper},${paper})`;
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = `rgb(${ink},${ink},${ink})`;
        ctx.direction = "rtl";
        ctx.textBaseline = "alphabetic";

        const lines = [];
        const footer = random() < 0.5 ? size * spacing : 0;
        let y = margin;
        const startDoc = doc;
        while (!done()) {
            if (doc !== startDoc && lines.length > 0) break; // every document starts on a new page
            const heading = para === 0 && word === 0 && !job.docs[doc].cont;
            const fontSize = heading ? Math.round(size * 1.25) : size;
            ctx.font = `${heading ? "bold " : ""}${fontSize}px ${family}`;
            const metrics = ctx.measureText("المادة");
            const ascent = metrics.fontBoundingBoxAscent, descent = metrics.fontBoundingBoxDescent;
            const height = fontSize * spacing;
            if (y + height > H - margin - footer) break;

            // As many words as fit the width (a word wider than the page on its own is cut at a character).
            const all = words(), room = W - 2 * margin;
            let text = "", next = word;
            while (next < all.length) {
                const tried = text ? text + " " + all[next] : all[next];
                if (ctx.measureText(tried).width > room) {
                    if (!text) {
                        let cut = all[next].length;
                        while (cut > 1 && ctx.measureText(all[next].slice(0, cut)).width > room) cut--;
                        text = all[next].slice(0, cut);
                        all[next] = all[next].slice(cut);
                        job.docs[doc].paras[para] = all.join(" ");
                    }
                    break;
                }
                text = tried; next++;
            }
            const baseline = y + (height - (ascent + descent)) / 2 + ascent;
            const textWidth = ctx.measureText(text).width;
            const right = heading ? (W + textWidth) / 2 : W - margin;
            ctx.textAlign = "right";
            ctx.fillText(text, right, baseline);
            lines.push({ text, left: right - textWidth, right, top: baseline - ascent, bottom: baseline + descent, size: fontSize });
            y += height;
            word = next;
            if (word >= all.length) {
                const newDoc = advance();
                y += newDoc ? 0 : height * 0.35;
            }
        }
        if (lines.length === 0) break; // nothing fits a page this size: give up on the batch rather than loop

        if (footer) {
            ctx.font = `${Math.round(size * 0.85)}px ${family}`;
            const text = String(docPage);
            const m = ctx.measureText(text);
            const baseline = H - margin;
            ctx.textAlign = "center";
            ctx.fillText(text, W / 2, baseline);
            lines.push({ text, left: W / 2 - m.width / 2, right: W / 2 + m.width / 2,
                top: baseline - m.fontBoundingBoxAscent, bottom: baseline + m.fontBoundingBoxDescent, size: Math.round(size * 0.85) });
        }

        // Lines are cut from the clean page; scan-like damage is the trainer's augmentation for them.
        if (job.lines) {
            lines.forEach((line, i) => {
                const pad = Math.round(line.size * 0.3);
                const x0 = Math.max(0, Math.floor(line.left) - pad), x1 = Math.min(W, Math.ceil(line.right) + pad);
                const y0 = Math.max(0, Math.floor(line.top) - Math.round(pad / 2)), y1 = Math.min(H, Math.ceil(line.bottom) + Math.round(pad / 2));
                const crop = document.createElement("canvas");
                crop.width = x1 - x0; crop.height = y1 - y0;
                crop.getContext("2d").drawImage(canvas, x0, y0, crop.width, crop.height, 0, 0, crop.width, crop.height);
                out.push(["L", page, i, utf8(line.text), crop.toDataURL("image/png").split(",")[1]].join("\t"));
            });
        }

        // The page image: optionally a little rotated, blurred and noisy, saved as JPEG like most scans.
        let image = canvas;
        if (job.degrade) {
            image = document.createElement("canvas");
            image.width = W; image.height = H;
            const g = image.getContext("2d");
            g.fillStyle = `rgb(${paper},${paper},${paper})`;
            g.fillRect(0, 0, W, H);
            g.filter = `blur(${between(0, 0.7).toFixed(2)}px)`;
            g.translate(W / 2, H / 2);
            g.rotate(between(-0.8, 0.8) * Math.PI / 180);
            g.drawImage(canvas, -W / 2, -H / 2);
            g.setTransform(1, 0, 0, 1, 0, 0);
            g.filter = "none";
            const pixels = g.getImageData(0, 0, W, H), d = pixels.data, amount = between(0, 14);
            for (let p = 0; p < d.length; p += 4) {
                const v = Math.max(0, Math.min(255, d[p] + (random() + random() + random() - 1.5) * amount));
                d[p] = d[p + 1] = d[p + 2] = v;
            }
            g.putImageData(pixels, 0, 0);
        }
        const quality = job.degrade ? between(0.7, 0.92) : 0.92;
        out.push(["P", page, utf8(lines.map(l => l.text).join("\n")), image.toDataURL("image/jpeg", quality).split(",")[1]].join("\t"));
        page++;
    }
    document.getElementById("out").textContent = out.join("\n");
})();
