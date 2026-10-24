/* The C harness behind tools/audio/make_opus_reference.py (plan 32, phase O0): libopus' own encoder and decoder around
   an Ogg page writer and reader written here (RFC 3533; Ogg Opus as RFC 7845 maps it), so the fixtures and their
   decoded samples come from xiph's code alone. Built by the script against a libopus checkout and its static library;
   not part of Idrak.

   Commands (every result on stdout as one JSON object; a failure's cause too, with exit code 2):
     encode CONFIG IN.f32 OUT.opus
         Encodes interleaved float32 samples at the CONFIG's rate (see read_config for its keys) into an Ogg Opus file:
         OpusHead (pre-skip from OPUS_GET_LOOKAHEAD at 48 kHz, the input rate, the output gain, mapping family 0, or 1
         for a surround multistream encoder), OpusTags, then the packets; each page's granule is the 48 kHz sample count
         of the packets completed on it plus nothing else (pre-skip included, as RFC 7845 counts it); the last page's
         granule is pre-skip + the input's length, so a decoder trims the padding the encoder needed. Prints each
         packet's encoder final range (OPUS_GET_FINAL_RANGE).
     decode IN.opus RATE LINK GAIN OUT.f32 OUT.i16 [FRAMES.f32]
         Reads the file's pages (CRC-32 checked) and the LINK-th logical stream's packets (0: the first; streams are
         counted in the order their first page appears), decodes them with opus_decode_float and, on a second decoder,
         opus_decode (int16), at RATE (48000, 24000, 16000, 12000 or 8000), with OpusHead's output gain set by
         OPUS_SET_GAIN when GAIN is 1, drops pre-skip x RATE / 48000 samples from the start and keeps
         (last granule - pre-skip) x RATE / 48000 samples, writing them interleaved. Prints the header, the tags, the
         pages and each packet's TOC fields, frame sizes, padding and final range (per stream for a multistream); with
         FRAMES.f32 also decodes every frame alone (a code-0 packet of the frame's TOC and bytes) on a third decoder,
         printing each frame's final range and writing its samples (which must equal the packets').
     repack IN.opus OUT.opus GROUP PAD SERIAL
         Joins every GROUP consecutive packets with libopus' repacketizer (opus_repacketizer_cat / _out) and, when PAD is
         over 0, pads each to its length + PAD with opus_packet_pad; same header, granules and end. GROUP -1 puts every
         packet's frames in twice (a packet of one frame becomes a code-1 packet of two equal frames, a new stream twice
         as long, its end trimmed by as much as the source's).
     symbols IN.opus PACKETS
         For the first PACKETS packets of the first stream: the first symbols of their first frame decoded with libopus'
         range decoder (celt/entdec.c) as the frame's decoder reads them (SILK and hybrid: the VAD and LBRR flags;
         CELT: silence, post-filter, transient, intra), with the decoder's rng, val, tell and tell_frac after each.
     rangecoder
         Seeded symbol sequences (bit_logp, icdf, uint, bits, laplace) encoded with libopus' range encoder, then
         decoded, with each symbol's value and the decoder's state after it.
*/
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "opus.h"
#include "opus_multistream.h"
#include "opus_private.h"
#include "entenc.h"
#include "entdec.h"
#include "laplace.h"

#ifdef OPUS_REFERENCE_TRACE
/* Built with OPUS_REFERENCE_TRACE against a copy of libopus' src/opus_decoder.c that calls these (two lines added by
   make_opus_reference.py, nothing else changed): for each frame the first decoder decodes, the mode, the previous
   mode, where the SILK part ended (ec_tell, bits; -1 without data), whether a redundant CELT frame follows (RFC 6716
   4.5.1), its direction (1: CELT to SILK) and its bytes. */
int opus_reference_silk_tell = -1;
static int trace_on = 0, trace_n = 0, trace[48][6];
void opus_reference_trace(int mode, int prev_mode, int redundancy, int celt_to_silk, int redundancy_bytes, int len);
void opus_reference_trace(int mode, int prev_mode, int redundancy, int celt_to_silk, int redundancy_bytes, int len) {
  (void)len;
  if (!trace_on || trace_n >= 48) return;
  int *t = trace[trace_n++];
  t[0] = mode; t[1] = prev_mode; t[2] = opus_reference_silk_tell; t[3] = redundancy; t[4] = celt_to_silk; t[5] = redundancy_bytes;
}
#endif

#define MAX_FRAME_48K 5760          /* 120 ms at 48 kHz */
#define MAX_PACKET 1500 * 6         /* room for a 120 ms packet of a 6-channel stream */

/* A failure: its cause as the last line of stdout ({"error": ...}, after whatever was printed before), exit code 2. */
static void fail(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  printf("\n{\"error\": \"");
  vprintf(fmt, ap);
  printf("\"}\n");
  va_end(ap);
  exit(2);
}

static void *grow(void *p, size_t n) {
  void *q = realloc(p, n ? n : 1);
  if (!q) { fprintf(stderr, "out of memory\n"); exit(1); }
  return q;
}

static unsigned char *read_file(const char *path, long *size) {
  FILE *f = fopen(path, "rb");
  if (!f) fail("cannot open %s", path);
  fseek(f, 0, SEEK_END);
  *size = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *data = grow(NULL, (size_t)*size);
  if (*size && fread(data, 1, (size_t)*size, f) != (size_t)*size) fail("cannot read %s", path);
  fclose(f);
  return data;
}

static void json_string(const unsigned char *s, size_t n) {
  putchar('"');
  for (size_t i = 0; i < n; i++) {
    unsigned char c = s[i];
    if (c == '"' || c == '\\') printf("\\%c", c);
    else if (c < 0x20) printf("\\u%04x", c);
    else putchar(c);
  }
  putchar('"');
}

/* ---- Ogg (RFC 3533): CRC-32 with polynomial 0x04C11DB7, no reflection, initial value 0, no final xor ---- */

static uint32_t crc_table[256];

static void crc_init(void) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t r = i << 24;
    for (int k = 0; k < 8; k++) r = (r & 0x80000000u) ? (r << 1) ^ 0x04C11DB7u : r << 1;
    crc_table[i] = r;
  }
}

static uint32_t crc_update(uint32_t crc, const unsigned char *p, size_t n) {
  for (size_t i = 0; i < n; i++) crc = (crc << 8) ^ crc_table[((crc >> 24) ^ p[i]) & 0xFF];
  return crc;
}

static void put16(unsigned char *p, uint32_t v) { p[0] = v; p[1] = v >> 8; }
static void put32(unsigned char *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }
static void put64(unsigned char *p, int64_t v) { put32(p, (uint32_t)v); put32(p + 4, (uint32_t)((uint64_t)v >> 32)); }
static uint32_t get16(const unsigned char *p) { return p[0] | p[1] << 8; }
static uint32_t get32(const unsigned char *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static int64_t get64(const unsigned char *p) { return (int64_t)((uint64_t)get32(p) | (uint64_t)get32(p + 4) << 32); }

/* The writer queues lacing values; a page is cut when it reaches 255 of them, when its body reaches max_page bytes, or
   at a packet's end once the page holds page_samples (48 kHz) or more since the last page's granule. */
typedef struct {
  FILE *out;
  uint32_t serial, sequence;
  int max_page, page_samples, pages;
  unsigned char *data; size_t data_len, data_head;          /* the queued packets' bytes */
  int *lace; int *ends; int64_t *granule; int nseg, head;    /* queued segments: length, packet end, granule there */
  int continued;                                             /* the next page starts inside a packet */
  int64_t last_granule;
  int first;                                                 /* the next page is the stream's first (BOS) */
} OggWriter;

static void ogg_page(OggWriter *w, int count, int eos, int64_t override_granule) {
  unsigned char header[27 + 255];
  int body = 0;
  int64_t granule = -1;
  for (int i = 0; i < count; i++) {
    header[27 + i] = (unsigned char)w->lace[w->head + i];
    body += w->lace[w->head + i];
    if (w->ends[w->head + i]) granule = w->granule[w->head + i];
  }
  if (eos && override_granule >= 0) granule = override_granule;
  memcpy(header, "OggS", 4);
  header[4] = 0;
  header[5] = (unsigned char)((w->continued ? 1 : 0) | (w->first ? 2 : 0) | (eos ? 4 : 0));
  put64(header + 6, granule);
  put32(header + 14, w->serial);
  put32(header + 18, w->sequence++);
  put32(header + 22, 0);
  header[26] = (unsigned char)count;
  uint32_t crc = crc_update(0, header, 27 + count);
  crc = crc_update(crc, w->data + w->data_head, body);
  put32(header + 22, crc);
  fwrite(header, 1, 27 + count, w->out);
  fwrite(w->data + w->data_head, 1, body, w->out);
  w->continued = !w->ends[w->head + count - 1];
  w->head += count;
  w->data_head += body;
  w->first = 0;
  w->pages++;
  if (granule >= 0) w->last_granule = granule;
}

static void ogg_packet(OggWriter *w, const unsigned char *p, int n, int64_t granule) {
  w->data = grow(w->data, w->data_len + n);
  memcpy(w->data + w->data_len, p, n);
  w->data_len += n;
  int segs = n / 255 + 1;
  w->lace = grow(w->lace, sizeof(int) * (w->nseg + segs));
  w->ends = grow(w->ends, sizeof(int) * (w->nseg + segs));
  w->granule = grow(w->granule, sizeof(int64_t) * (w->nseg + segs));
  for (int i = 0; i < segs; i++) {
    w->lace[w->nseg] = i < segs - 1 ? 255 : n % 255;
    w->ends[w->nseg] = i == segs - 1;
    w->granule[w->nseg] = granule;
    w->nseg++;
  }
}

/* Writes the pages that are complete; with flush, everything queued (the last page EOS when eos). */
static void ogg_pageout(OggWriter *w, int flush, int eos, int64_t final_granule) {
  for (;;) {
    int avail = w->nseg - w->head;
    if (avail == 0) return;
    int bytes = 0, cut = -1;
    for (int i = 0; i < avail && i < 255; i++) {
      bytes += w->lace[w->head + i];
      if (i == 254) { cut = 255; break; }
      if (bytes >= w->max_page) { cut = i + 1; break; }
      if (w->ends[w->head + i] && w->page_samples > 0 && w->granule[w->head + i] - w->last_granule >= w->page_samples) { cut = i + 1; break; }
    }
    if (cut < 0) {
      if (!flush) return;
      cut = avail;
    }
    ogg_page(w, cut, eos && cut == avail, final_granule);
  }
}

typedef struct { int offset, size, flags, segments, sequence, packets_ended; int64_t granule; uint32_t serial; } Page;
typedef struct { unsigned char *data; int len, page; int64_t granule; } Packet;
typedef struct { uint32_t serial; Packet *packets; int npackets; int64_t last_granule; int eos; int pages; } Stream;
typedef struct { Page *pages; int npages; Stream *streams; int nstreams; } Ogg;

/* Reads every page; refuses (naming the page) a bad capture pattern, version, CRC, a page cut short, a sequence gap,
   a continuation the stream does not have, and a packet not finished when its stream ends. */
static Ogg ogg_read(const unsigned char *file, long size) {
  Ogg ogg = {0};
  int *pending_len = NULL;
  unsigned char **pending = NULL;
  long off = 0;
  while (off < size) {
    int index = ogg.npages;
    if (size - off < 27) fail("page %d at byte %ld: truncated header", index, off);
    if (memcmp(file + off, "OggS", 4) != 0) fail("page %d at byte %ld: no OggS capture pattern", index, off);
    if (file[off + 4] != 0) fail("page %d at byte %ld: stream structure version not 0", index, off);
    int nseg = file[off + 26];
    if (size - off < 27 + nseg) fail("page %d at byte %ld: truncated segment table", index, off);
    int body = 0;
    for (int i = 0; i < nseg; i++) body += file[off + 27 + i];
    if (size - off < 27 + nseg + body) fail("page %d at byte %ld: truncated body", index, off);
    unsigned char header[27 + 255];
    memcpy(header, file + off, 27 + nseg);
    put32(header + 22, 0);
    uint32_t crc = crc_update(crc_update(0, header, 27 + nseg), file + off + 27 + nseg, body);
    if (crc != get32(file + off + 22)) fail("page %d at byte %ld: CRC mismatch", index, off);
    Page pg = { (int)off, 27 + nseg + body, file[off + 5], nseg, (int)get32(file + off + 18), 0, get64(file + off + 6), get32(file + off + 14) };
    int s;
    for (s = 0; s < ogg.nstreams && ogg.streams[s].serial != pg.serial; s++) {}
    if (s == ogg.nstreams) {
      if (!(pg.flags & 2)) fail("page %d at byte %ld: a new stream without BOS", index, off);
      ogg.streams = grow(ogg.streams, sizeof(Stream) * (ogg.nstreams + 1));
      memset(&ogg.streams[s], 0, sizeof(Stream));
      ogg.streams[s].serial = pg.serial;
      ogg.streams[s].last_granule = -1;
      pending = grow(pending, sizeof(*pending) * (ogg.nstreams + 1));
      pending_len = grow(pending_len, sizeof(int) * (ogg.nstreams + 1));
      pending[s] = NULL; pending_len[s] = -1;
      ogg.nstreams++;
      if (pg.sequence != 0) fail("page %d at byte %ld: first page's sequence number not 0", index, off);
    } else {
      Stream *st = &ogg.streams[s];
      if (st->eos) fail("page %d at byte %ld: a page after its stream's EOS", index, off);
      if (pg.flags & 2) fail("page %d at byte %ld: BOS on a stream already begun", index, off);
      if (pg.sequence != st->pages) fail("page %d at byte %ld: sequence gap (a page lost)", index, off);
    }
    Stream *st = &ogg.streams[s];
    int continued = pg.flags & 1;
    if (continued && pending_len[s] < 0) fail("page %d at byte %ld: continues a packet its stream does not have", index, off);
    if (!continued && pending_len[s] >= 0) fail("page %d at byte %ld: a packet left unfinished by the page before", index, off);
    const unsigned char *p = file + off + 27 + nseg;
    for (int i = 0; i < nseg; i++) {
      int l = file[off + 27 + i];
      if (pending_len[s] < 0) pending_len[s] = 0;
      pending[s] = grow(pending[s], pending_len[s] + l);
      memcpy(pending[s] + pending_len[s], p, l);
      pending_len[s] += l;
      p += l;
      if (l < 255) {
        st->packets = grow(st->packets, sizeof(Packet) * (st->npackets + 1));
        Packet pk = { pending[s], pending_len[s], index, -1 };
        st->packets[st->npackets++] = pk;
        pending[s] = NULL; pending_len[s] = -1;
        pg.packets_ended++;
      }
    }
    if (pg.packets_ended > 0) {
      st->packets[st->npackets - 1].granule = pg.granule;
      if (pg.granule < 0) fail("page %d at byte %ld: packets end on it but its granule is -1", index, off);
    } else if (pg.granule != -1) fail("page %d at byte %ld: no packet ends on it but its granule is not -1", index, off);
    if (pg.granule >= 0) {
      if (st->last_granule > pg.granule) fail("page %d at byte %ld: granule position goes back", index, off);
      st->last_granule = pg.granule;
    }
    if (pg.flags & 4) st->eos = 1;
    st->pages++;
    ogg.pages = grow(ogg.pages, sizeof(Page) * (ogg.npages + 1));
    ogg.pages[ogg.npages++] = pg;
    off += pg.size;
  }
  for (int s = 0; s < ogg.nstreams; s++)
    if (pending_len[s] >= 0) fail("stream %d: the file ends inside a packet (%d bytes of it)", s, pending_len[s]);
  return ogg;
}

/* ---- OpusHead and OpusTags (RFC 7845 5.1, 5.2) ---- */

typedef struct { int version, channels, pre_skip, gain, family, streams, coupled; uint32_t input_rate; unsigned char mapping[255]; } OpusHead;

static OpusHead parse_head(const Packet *p) {
  OpusHead h = {0};
  if (p->len < 19 || memcmp(p->data, "OpusHead", 8) != 0) fail("packet 0 (page %d): not an OpusHead", p->page);
  h.version = p->data[8];
  if (h.version >> 4 != 0) fail("OpusHead version %d: major version not 0", h.version);
  h.channels = p->data[9];
  h.pre_skip = (int)get16(p->data + 10);
  h.input_rate = get32(p->data + 12);
  h.gain = (int16_t)get16(p->data + 16);
  h.family = p->data[18];
  if (h.channels == 0) fail("OpusHead: 0 channels");
  if (h.family == 0) {
    if (h.channels > 2) fail("OpusHead: mapping family 0 with %d channels", h.channels);
    h.streams = 1; h.coupled = h.channels - 1; h.mapping[0] = 0; h.mapping[1] = 1;
  } else {
    if (p->len < 21 + h.channels) fail("OpusHead: the channel mapping table is cut short (%d bytes)", p->len);
    h.streams = p->data[19];
    h.coupled = p->data[20];
    memcpy(h.mapping, p->data + 21, h.channels);
  }
  return h;
}

static void print_tags(const Packet *p) {
  if (p->len < 16 || memcmp(p->data, "OpusTags", 8) != 0) fail("packet 1 (page %d): not an OpusTags", p->page);
  uint32_t vlen = get32(p->data + 8);
  printf("\"vendor\": ");
  json_string(p->data + 12, vlen);
  uint32_t at = 12 + vlen, count = get32(p->data + at);
  at += 4;
  printf(", \"comments\": [");
  for (uint32_t i = 0; i < count; i++) {
    uint32_t n = get32(p->data + at);
    printf(i ? ", " : "");
    json_string(p->data + at + 4, n);
    at += 4 + n;
  }
  printf("]");
}

static void write_head(OggWriter *w, int channels, int pre_skip, uint32_t rate, int gain, int family, int streams, int coupled, const unsigned char *mapping) {
  unsigned char h[21 + 255];
  int n = 19;
  memcpy(h, "OpusHead", 8);
  h[8] = 1;
  h[9] = (unsigned char)channels;
  put16(h + 10, (uint32_t)pre_skip);
  put32(h + 12, rate);
  put16(h + 16, (uint32_t)(uint16_t)(int16_t)gain);
  h[18] = (unsigned char)family;
  if (family != 0) {
    h[19] = (unsigned char)streams;
    h[20] = (unsigned char)coupled;
    memcpy(h + 21, mapping, channels);
    n = 21 + channels;
  }
  ogg_packet(w, h, n, 0);
  ogg_pageout(w, 1, 0, -1);
}

static void write_tags(OggWriter *w, const char *vendor, int ncomments, char comments[][512]) {
  unsigned char t[4096];
  int n = 0;
  memcpy(t, "OpusTags", 8); n = 8;
  put32(t + n, (uint32_t)strlen(vendor)); n += 4;
  memcpy(t + n, vendor, strlen(vendor)); n += (int)strlen(vendor);
  put32(t + n, (uint32_t)ncomments); n += 4;
  for (int i = 0; i < ncomments; i++) {
    int l = (int)strlen(comments[i]);
    put32(t + n, (uint32_t)l); n += 4;
    memcpy(t + n, comments[i], l); n += l;
  }
  ogg_packet(w, t, n, 0);
  ogg_pageout(w, 1, 0, -1);
}

/* ---- the TOC (RFC 6716 3.1, 3.2) ---- */

static const char *mode_name(int config) { return config < 12 ? "silk" : config < 16 ? "hybrid" : "celt"; }
static const char *bandwidth_name(int config) {
  static const char *silk[] = { "nb", "mb", "wb" }, *celt[] = { "nb", "wb", "swb", "fb" };
  if (config < 12) return silk[config / 4];
  if (config < 16) return config < 14 ? "swb" : "fb";
  return celt[(config - 16) / 4];
}
/* The frame's duration in 1/10 ms. */
static int frame_tenths(int config) {
  static const int silk[] = { 100, 200, 400, 600 }, hybrid[] = { 100, 200 }, celt[] = { 25, 50, 100, 200 };
  if (config < 12) return silk[config % 4];
  if (config < 16) return hybrid[config % 2];
  return celt[config % 4];
}

/* Prints the packet's TOC fields, frame sizes and padding as libopus parses them (opus_packet_parse), with the code-3
   header read here for the padding and VBR flag. */
static int print_toc(const unsigned char *d, int len, const unsigned char **frames, opus_int16 *sizes) {
  unsigned char toc;
  int offset = 0;
  int count = opus_packet_parse(d, len, &toc, frames, sizes, &offset);
  if (count < 0) return count;
  int config = d[0] >> 3, code = d[0] & 3, padding = 0, vbr = code == 2;
  if (code == 3) {
    vbr = (d[1] >> 7) & 1;
    if (d[1] & 0x40) {
      int at = 2;
      for (;;) { int b = d[at++]; padding += b == 255 ? 254 : b; if (b != 255) break; }
    }
  }
  printf("\"bytes\": %d, \"config\": %d, \"mode\": \"%s\", \"bandwidth\": \"%s\", \"frame_ms\": %g, \"stereo\": %d, \"code\": %d, \"vbr\": %d, \"padding\": %d, \"frames\": [",
         len, config, mode_name(config), bandwidth_name(config), frame_tenths(config) / 10.0, (d[0] >> 2) & 1, code, vbr, padding);
  for (int i = 0; i < count; i++) printf(i ? ", %d" : "%d", sizes[i]);
  printf("]");
  return count;
}

/* A multistream packet (RFC 7845 5.1.1): every stream's packet but the last self-delimited (RFC 6716 appendix B). */
static int print_ms_toc(const unsigned char *d, int len, int streams) {
  printf("\"bytes\": %d, \"streams\": [", len);
  int at = 0;
  for (int s = 0; s < streams; s++) {
    unsigned char toc;
    const unsigned char *frames[48], *padding;
    opus_int16 sizes[48];
    opus_int32 packet_offset = 0, padding_len = 0;
    int payload = 0;
    int count = opus_packet_parse_impl(d + at, len - at, s < streams - 1, &toc, frames, sizes, &payload, &packet_offset, &padding, &padding_len);
    if (count < 0) return count;
    int config = toc >> 3, code = toc & 3;
    printf("%s{\"config\": %d, \"mode\": \"%s\", \"bandwidth\": \"%s\", \"frame_ms\": %g, \"stereo\": %d, \"code\": %d, \"bytes\": %d, \"padding\": %d, \"frames\": [",
           s ? ", " : "", config, mode_name(config), bandwidth_name(config), frame_tenths(config) / 10.0, (toc >> 2) & 1, code, (int)packet_offset, (int)padding_len);
    for (int i = 0; i < count; i++) printf(i ? ", %d" : "%d", sizes[i]);
    printf("]}");
    at += packet_offset;
  }
  printf("]");
  return 1;
}

/* ---- encode ---- */

typedef struct { int start; char key[32]; char value[64]; } Change;

typedef struct {
  int rate, channels, family, application, frame_tenths, bitrate, vbr, cvbr, complexity, dtx, fec, loss, gain;
  int bandwidth, mode, signal, max_page, page_ms, length, lsb, force_channels;
  uint32_t serial;
  int ncomments; char comments[8][512];
  int nchanges; Change changes[64];
} Config;

static int parse_named(const char *key, const char *v) {
  if (!strcmp(key, "bandwidth")) {
    if (!strcmp(v, "nb")) return OPUS_BANDWIDTH_NARROWBAND;
    if (!strcmp(v, "mb")) return OPUS_BANDWIDTH_MEDIUMBAND;
    if (!strcmp(v, "wb")) return OPUS_BANDWIDTH_WIDEBAND;
    if (!strcmp(v, "swb")) return OPUS_BANDWIDTH_SUPERWIDEBAND;
    if (!strcmp(v, "fb")) return OPUS_BANDWIDTH_FULLBAND;
    return OPUS_AUTO;
  }
  if (!strcmp(key, "mode")) {
    if (!strcmp(v, "silk")) return MODE_SILK_ONLY;
    if (!strcmp(v, "hybrid")) return MODE_HYBRID;
    if (!strcmp(v, "celt")) return MODE_CELT_ONLY;
    return OPUS_AUTO;
  }
  if (!strcmp(key, "signal")) {
    if (!strcmp(v, "voice")) return OPUS_SIGNAL_VOICE;
    if (!strcmp(v, "music")) return OPUS_SIGNAL_MUSIC;
    return OPUS_AUTO;
  }
  if (!strcmp(key, "application")) {
    if (!strcmp(v, "voip")) return OPUS_APPLICATION_VOIP;
    if (!strcmp(v, "lowdelay")) return OPUS_APPLICATION_RESTRICTED_LOWDELAY;
    return OPUS_APPLICATION_AUDIO;
  }
  if (!strcmp(key, "frame")) return (int)lround(atof(v) * 10);
  return atoi(v);
}

static void set_key(Config *c, const char *key, const char *v) {
  int x = parse_named(key, v);
  if (!strcmp(key, "rate")) c->rate = x;
  else if (!strcmp(key, "channels")) c->channels = x;
  else if (!strcmp(key, "family")) c->family = x;
  else if (!strcmp(key, "application")) c->application = x;
  else if (!strcmp(key, "frame")) c->frame_tenths = x;
  else if (!strcmp(key, "bitrate")) c->bitrate = x;
  else if (!strcmp(key, "vbr")) c->vbr = x;
  else if (!strcmp(key, "cvbr")) c->cvbr = x;
  else if (!strcmp(key, "complexity")) c->complexity = x;
  else if (!strcmp(key, "dtx")) c->dtx = x;
  else if (!strcmp(key, "fec")) c->fec = x;
  else if (!strcmp(key, "loss")) c->loss = x;
  else if (!strcmp(key, "gain")) c->gain = x;
  else if (!strcmp(key, "bandwidth")) c->bandwidth = x;
  else if (!strcmp(key, "mode")) c->mode = x;
  else if (!strcmp(key, "signal")) c->signal = x;
  else if (!strcmp(key, "max_page")) c->max_page = x;
  else if (!strcmp(key, "page_ms")) c->page_ms = x;
  else if (!strcmp(key, "length")) c->length = x;
  else if (!strcmp(key, "lsb")) c->lsb = x;
  else if (!strcmp(key, "force_channels")) c->force_channels = x;
  else if (!strcmp(key, "serial")) c->serial = (uint32_t)strtoul(v, NULL, 10);
  else fail("config: unknown key %s", key);
}

/* The config: lines "key value" (rate, channels, family, application voip|audio|lowdelay, frame ms, bitrate, vbr, cvbr,
   complexity, dtx, fec, loss, gain (OpusHead's, Q7.8 dB), bandwidth nb|mb|wb|swb|fb|auto, mode silk|hybrid|celt|auto,
   signal voice|music|auto, max_page, page_ms, length (samples per channel), lsb, serial, force_channels 1|2|-1000 for
   auto), "comment TEXT" for OpusTags,
   and "at SAMPLE key value" for a change from the frame that starts at or after that input sample. */
static Config read_config(const char *path) {
  Config c = { .rate = 48000, .channels = 1, .application = OPUS_APPLICATION_AUDIO, .frame_tenths = 200, .bitrate = 32000, .vbr = 1,
                .complexity = 10, .bandwidth = OPUS_AUTO, .mode = OPUS_AUTO, .signal = OPUS_AUTO, .max_page = 4096, .page_ms = 200,
                .lsb = 24, .serial = 1, .force_channels = OPUS_AUTO };
  FILE *f = fopen(path, "r");
  if (!f) fail("cannot open %s", path);
  char line[512];
  while (fgets(line, sizeof line, f)) {
    line[strcspn(line, "\r\n")] = 0;
    if (!line[0] || line[0] == '#') continue;
    char key[64], value[256];
    if (!strncmp(line, "comment ", 8)) { snprintf(c.comments[c.ncomments++], 512, "%s", line + 8); continue; }
    if (!strncmp(line, "at ", 3)) {
      Change *ch = &c.changes[c.nchanges++];
      if (sscanf(line + 3, "%d %31s %63s", &ch->start, ch->key, ch->value) != 3) fail("config: bad line %s", line);
      continue;
    }
    if (sscanf(line, "%63s %255s", key, value) != 2) fail("config: bad line %s", line);
    set_key(&c, key, value);
  }
  fclose(f);
  return c;
}

static int ctl(OpusEncoder *e, OpusMSEncoder *ms, int request, int value) {
  return e ? opus_encoder_ctl(e, request, value) : opus_multistream_encoder_ctl(ms, request, value);
}

static void apply(OpusEncoder *e, OpusMSEncoder *ms, const Config *c) {
  int r = 0;
  r |= ctl(e, ms, OPUS_SET_BITRATE_REQUEST, c->bitrate);
  r |= ctl(e, ms, OPUS_SET_VBR_REQUEST, c->vbr);
  r |= ctl(e, ms, OPUS_SET_VBR_CONSTRAINT_REQUEST, c->cvbr);
  r |= ctl(e, ms, OPUS_SET_COMPLEXITY_REQUEST, c->complexity);
  r |= ctl(e, ms, OPUS_SET_DTX_REQUEST, c->dtx);
  r |= ctl(e, ms, OPUS_SET_INBAND_FEC_REQUEST, c->fec);
  r |= ctl(e, ms, OPUS_SET_PACKET_LOSS_PERC_REQUEST, c->loss);
  r |= ctl(e, ms, OPUS_SET_BANDWIDTH_REQUEST, c->bandwidth);
  r |= ctl(e, ms, OPUS_SET_SIGNAL_REQUEST, c->signal);
  r |= ctl(e, ms, OPUS_SET_LSB_DEPTH_REQUEST, c->lsb);
  r |= ctl(e, ms, OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, OPUS_FRAMESIZE_ARG);
  r |= ctl(e, ms, OPUS_SET_FORCE_CHANNELS_REQUEST, c->force_channels);
  if (e) r |= opus_encoder_ctl(e, OPUS_SET_FORCE_MODE(c->mode));
  else if (c->mode != OPUS_AUTO) fail("mode: not forced on a multistream encoder");
  if (r != OPUS_OK) fail("an encoder ctl refused a setting");
}

static int cmd_encode(const char *config_path, const char *in_path, const char *out_path) {
  Config c = read_config(config_path);
  long insize;
  float *input = (float *)read_file(in_path, &insize);
  if (insize != (long)sizeof(float) * c.length * c.channels) fail("input: %ld bytes, the config's length needs %ld", insize, (long)sizeof(float) * c.length * c.channels);
  int err = 0, streams = 1, coupled = c.channels - 1;
  unsigned char mapping[255] = { 0, 1 };
  OpusEncoder *e = NULL;
  OpusMSEncoder *ms = NULL;
  if (c.family == 0) {
    e = opus_encoder_create(c.rate, c.channels, c.application, &err);
  } else {
    ms = opus_multistream_surround_encoder_create(c.rate, c.channels, c.family, &streams, &coupled, mapping, c.application, &err);
  }
  if (err != OPUS_OK) fail("encoder create: %s", opus_strerror(err));
  apply(e, ms, &c);
  int lookahead = 0;
  if (e) opus_encoder_ctl(e, OPUS_GET_LOOKAHEAD(&lookahead));
  else opus_multistream_encoder_ctl(ms, OPUS_GET_LOOKAHEAD(&lookahead));
  int scale = 48000 / c.rate;
  int pre_skip = lookahead * scale;

  OggWriter w = {0};
  w.out = fopen(out_path, "wb");
  if (!w.out) fail("cannot write %s", out_path);
  w.serial = c.serial; w.max_page = c.max_page; w.page_samples = c.page_ms * 48; w.first = 1;
  write_head(&w, c.channels, pre_skip, (uint32_t)c.rate, c.gain, c.family, streams, coupled, mapping);
  char vendor[128];
  snprintf(vendor, sizeof vendor, "%s", opus_get_version_string());
  write_tags(&w, vendor, c.ncomments, c.comments);
  w.last_granule = 0;

  printf("{\"lookahead\": %d, \"pre_skip\": %d, \"streams\": %d, \"coupled\": %d, \"mapping\": [", lookahead, pre_skip, streams, coupled);
  for (int i = 0; i < c.channels; i++) printf(i ? ", %d" : "%d", mapping[i]);
  printf("], \"version\": \"%s\", \"packets\": [", vendor);
  int64_t granule = 0, final_granule = (int64_t)pre_skip + (int64_t)c.length * scale;
  int pos = 0, n = 0, next_change = 0;
  float *frame = grow(NULL, sizeof(float) * MAX_FRAME_48K * c.channels);
  unsigned char packet[MAX_PACKET];
  /* Frames until the decoded samples cover the pre-skip and the whole input. */
  while ((int64_t)pos < (int64_t)c.length + lookahead) {
    int changed = 0;
    while (next_change < c.nchanges && c.changes[next_change].start <= pos) {
      set_key(&c, c.changes[next_change].key, c.changes[next_change].value);
      next_change++; changed = 1;
    }
    if (changed) apply(e, ms, &c);
    int fs = c.rate * c.frame_tenths / 10000;
    for (int i = 0; i < fs * c.channels; i++) {
      int at = pos * c.channels + i;
      frame[i] = at < c.length * c.channels ? input[at] : 0.0f;
    }
    int len = e ? opus_encode_float(e, frame, fs, packet, sizeof packet) : opus_multistream_encode_float(ms, frame, fs, packet, sizeof packet);
    if (len < 0) fail("opus_encode_float: %s (frame at sample %d)", opus_strerror(len), pos);
    uint32_t range = 0;
    if (e) opus_encoder_ctl(e, OPUS_GET_FINAL_RANGE(&range));
    else opus_multistream_encoder_ctl(ms, OPUS_GET_FINAL_RANGE(&range));
    granule += (int64_t)fs * scale;
    ogg_packet(&w, packet, len, granule);
    ogg_pageout(&w, 0, 0, -1);
    printf("%s[%d, %u]", n ? ", " : "", len, range);
    pos += fs; n++;
  }
  ogg_pageout(&w, 1, 1, final_granule);
  fclose(w.out);
  printf("], \"final_granule\": %lld, \"pages\": %d}\n", (long long)final_granule, w.pages);
  if (e) opus_encoder_destroy(e); else opus_multistream_encoder_destroy(ms);
  return 0;
}

/* ---- decode ---- */

typedef struct { OpusDecoder *d; OpusMSDecoder *ms; } Dec;

static Dec dec_create(const OpusHead *h, int rate, int gain) {
  Dec x = {0};
  int err;
  if (h->family == 0) x.d = opus_decoder_create(rate, h->channels, &err);
  else x.ms = opus_multistream_decoder_create(rate, h->channels, h->streams, h->coupled, h->mapping, &err);
  if (err != OPUS_OK) fail("decoder create: %s", opus_strerror(err));
  if (gain) {
    int r = x.d ? opus_decoder_ctl(x.d, OPUS_SET_GAIN(h->gain)) : opus_multistream_decoder_ctl(x.ms, OPUS_SET_GAIN(h->gain));
    if (r != OPUS_OK) fail("OPUS_SET_GAIN %d: %s", h->gain, opus_strerror(r));
  }
  return x;
}

static void print_ranges(Dec *x, int streams) {
  uint32_t r = 0;
  if (x->d) { opus_decoder_ctl(x->d, OPUS_GET_FINAL_RANGE(&r)); printf("%u", r); return; }
  printf("[");
  for (int s = 0; s < streams; s++) {
    OpusDecoder *one;
    opus_multistream_decoder_ctl(x->ms, OPUS_MULTISTREAM_GET_DECODER_STATE(s, &one));
    opus_decoder_ctl(one, OPUS_GET_FINAL_RANGE(&r));
    printf(s ? ", %u" : "%u", r);
  }
  printf("]");
}

static int cmd_decode(int argc, char **argv) {
  const char *in_path = argv[2];
  int rate = atoi(argv[3]), link = atoi(argv[4]), gain = atoi(argv[5]);
  const char *out_f = argv[6], *out_i = argv[7], *out_frames = argc > 8 ? argv[8] : NULL;
  long size;
  unsigned char *file = read_file(in_path, &size);
  Ogg ogg = ogg_read(file, size);
  printf("{\"pages\": [");
  for (int i = 0; i < ogg.npages; i++) {
    Page *p = &ogg.pages[i];
    printf("%s[%d, %d, %d, %lld, %u, %d, %d, %d]", i ? ", " : "", p->offset, p->size, p->flags, (long long)p->granule, p->serial, p->sequence, p->segments, p->packets_ended);
  }
  printf("], \"streams\": %d", ogg.nstreams);
  if (link >= ogg.nstreams) fail("stream %d: the file has %d", link, ogg.nstreams);
  Stream *st = &ogg.streams[link];
  if (st->npackets < 1) fail("stream %d: no packets", link);
  OpusHead h = parse_head(&st->packets[0]);
  if (st->npackets < 2) fail("stream %d: %d packets, OpusHead and OpusTags need 2", link, st->npackets);
  printf(", \"serial\": %u, \"head\": {\"version\": %d, \"channels\": %d, \"pre_skip\": %d, \"input_rate\": %u, \"gain\": %d, \"family\": %d, \"streams\": %d, \"coupled\": %d, \"mapping\": [",
         st->serial, h.version, h.channels, h.pre_skip, h.input_rate, h.gain, h.family, h.streams, h.coupled);
  for (int i = 0; i < h.channels; i++) printf(i ? ", %d" : "%d", h.mapping[i]);
  printf("]}, ");
  print_tags(&st->packets[1]);
  if (st->packets[0].granule != 0 || ogg.pages[st->packets[0].page].packets_ended != 1) fail("OpusHead: not alone on its page with granule 0");
  if (h.family > 1) fail("OpusHead: mapping family %d not read", h.family);
  if (rate != 48000 && rate != 24000 && rate != 16000 && rate != 12000 && rate != 8000) fail("rate %d: not an Opus decoding rate", rate);
  fflush(stdout);

  Dec df = dec_create(&h, rate, gain), di = dec_create(&h, rate, gain), dfr = {0};
  if (out_frames) dfr = dec_create(&h, rate, gain);
  int ratio = 48000 / rate, maxf = MAX_FRAME_48K / ratio;
  size_t total = 0, cap = 0, ftotal = 0, fcap = 0;
  float *pcm = NULL, *fpcm = NULL, *buf = grow(NULL, sizeof(float) * maxf * h.channels);
  opus_int16 *ipcm = NULL, *ibuf = grow(NULL, sizeof(opus_int16) * maxf * h.channels);
  int64_t samples48 = 0;
  printf(", \"packets\": [");
  for (int k = 2; k < st->npackets; k++) {
    Packet *pk = &st->packets[k];
    const unsigned char *frames[48];
    opus_int16 sizes[48];
    printf("%s{\"page\": %d, ", k > 2 ? ",\n " : "\n ", pk->page);
    int count = h.family == 0 ? print_toc(pk->data, pk->len, frames, sizes) : print_ms_toc(pk->data, pk->len, h.streams);
    if (count < 0) {
      fail("packet %d (page %d): %s", k - 2, pk->page, opus_strerror(count));
    }
#ifdef OPUS_REFERENCE_TRACE
    trace_n = 0; trace_on = 1;
#endif
    int n = df.d ? opus_decode_float(df.d, pk->data, pk->len, buf, maxf, 0) : opus_multistream_decode_float(df.ms, pk->data, pk->len, buf, maxf, 0);
#ifdef OPUS_REFERENCE_TRACE
    trace_on = 0;
#endif
    int ni = di.d ? opus_decode(di.d, pk->data, pk->len, ibuf, maxf, 0) : opus_multistream_decode(di.ms, pk->data, pk->len, ibuf, maxf, 0);
    if (n < 0 || ni != n) {
      fail("packet %d (page %d): %s", k - 2, pk->page, opus_strerror(n < 0 ? n : ni));
    }
    printf(", \"samples\": %d, \"final_range\": ", n);
    print_ranges(&df, h.streams);
#ifdef OPUS_REFERENCE_TRACE
    printf(", \"trace\": [");
    for (int t = 0; t < trace_n; t++)
      printf("%s[%d, %d, %d, %d, %d, %d]", t ? ", " : "", trace[t][0], trace[t][1], trace[t][2], trace[t][3], trace[t][4], trace[t][5]);
    printf("]");
#endif
    if (cap < total + (size_t)n * h.channels) { cap = (total + (size_t)n * h.channels) * 2; pcm = grow(pcm, cap * sizeof(float)); ipcm = grow(ipcm, cap * sizeof(opus_int16)); }
    memcpy(pcm + total, buf, sizeof(float) * n * h.channels);
    memcpy(ipcm + total, ibuf, sizeof(opus_int16) * n * h.channels);
    total += (size_t)n * h.channels;
    samples48 += (int64_t)n * ratio;
    if (out_frames) {
      /* Every frame alone: a code-0 packet of this packet's TOC and the frame's bytes (with a multistream, the
         self-delimited framing makes this meaningless, so it is skipped there). */
      printf(", \"frame_ranges\": [");
      for (int f = 0; f < count && dfr.d; f++) {
        unsigned char one[1 + 1275];
        one[0] = (unsigned char)(pk->data[0] & 0xFC);
        memcpy(one + 1, frames[f], sizes[f]);
        int m = opus_decode_float(dfr.d, one, 1 + sizes[f], buf, maxf, 0);
        if (m < 0) fail("frame %d of packet %d alone: %s", f, k - 2, opus_strerror(m));
        uint32_t r;
        opus_decoder_ctl(dfr.d, OPUS_GET_FINAL_RANGE(&r));
        printf(f ? ", %u" : "%u", r);
        if (fcap < ftotal + (size_t)m * h.channels) { fcap = (ftotal + (size_t)m * h.channels) * 2; fpcm = grow(fpcm, fcap * sizeof(float)); }
        memcpy(fpcm + ftotal, buf, sizeof(float) * m * h.channels);
        ftotal += (size_t)m * h.channels;
      }
      printf("]");
    }
    printf("}");
  }
  printf("]");
  int64_t end = st->last_granule;
  if (end < h.pre_skip) fail("the last granule %ld is before the pre-skip %d", (long)end, h.pre_skip);
  if (end > samples48) fail("the last granule %ld is past the %ld samples decoded", (long)end, (long)samples48);
  int64_t skip = (int64_t)h.pre_skip / ratio, length = (end - h.pre_skip) / ratio;
  int exact = h.pre_skip % ratio == 0 && (end - h.pre_skip) % ratio == 0;
  FILE *ff = fopen(out_f, "wb"), *fi = fopen(out_i, "wb");
  fwrite(pcm + skip * h.channels, sizeof(float), (size_t)length * h.channels, ff);
  fwrite(ipcm + skip * h.channels, sizeof(opus_int16), (size_t)length * h.channels, fi);
  fclose(ff); fclose(fi);
  if (out_frames) {
    FILE *fr = fopen(out_frames, "wb");
    if (ftotal >= (size_t)(skip + length) * h.channels) fwrite(fpcm + skip * h.channels, sizeof(float), (size_t)length * h.channels, fr);
    fclose(fr);
  }
  printf(", \"decoded_48k\": %lld, \"last_granule\": %lld, \"skip\": %lld, \"length\": %lld, \"exact_scaling\": %d}\n",
         (long long)samples48, (long long)end, (long long)skip, (long long)length, exact);
  return 0;
}

/* ---- repack ---- */

static int cmd_repack(const char *in_path, const char *out_path, int group, int pad, uint32_t serial) {
  long size;
  unsigned char *file = read_file(in_path, &size);
  Ogg ogg = ogg_read(file, size);
  Stream *st = &ogg.streams[0];
  OpusHead h = parse_head(&st->packets[0]);
  OggWriter w = {0};
  w.out = fopen(out_path, "wb");
  w.serial = serial; w.max_page = 4096; w.page_samples = 200 * 48; w.first = 1;
  ogg_packet(&w, st->packets[0].data, st->packets[0].len, 0);
  ogg_pageout(&w, 1, 0, -1);
  ogg_packet(&w, st->packets[1].data, st->packets[1].len, 0);
  ogg_pageout(&w, 1, 0, -1);
  w.last_granule = 0;
  OpusRepacketizer *rp = opus_repacketizer_create();
  int64_t granule = 0;
  printf("{\"packets\": [");
  int out = 0;
  int64_t source_samples = 0, total_samples = 0;
  for (int k = 2; k < st->npackets; k++) source_samples += opus_packet_get_nb_samples(st->packets[k].data, st->packets[k].len, 48000);
  for (int k = 2; k < st->npackets; k += group < 0 ? 1 : group) {
    opus_repacketizer_init(rp);
    int samples = 0;
    int copies = group < 0 ? 2 : 1, n = group < 0 ? 1 : group;
    for (int j = k; j < k + n && j < st->npackets; j++) {
      for (int c = 0; c < copies; c++) {
        int r = opus_repacketizer_cat(rp, st->packets[j].data, st->packets[j].len);
        if (r != OPUS_OK) fail("repacketizer_cat packet %d: %s", j - 2, opus_strerror(r));
        samples += opus_packet_get_nb_samples(st->packets[j].data, st->packets[j].len, 48000);
      }
    }
    unsigned char packet[1275 * 48 + 4096];
    int len = opus_repacketizer_out(rp, packet, sizeof packet);
    if (len < 0) fail("repacketizer_out: %s", opus_strerror(len));
    if (pad > 0) {
      int r = opus_packet_pad(packet, len, len + pad);
      if (r != OPUS_OK) fail("opus_packet_pad: %s", opus_strerror(r));
      len += pad;
    }
    granule += samples;
    total_samples += samples;
    ogg_packet(&w, packet, len, granule);
    ogg_pageout(&w, 0, 0, -1);
    printf(out++ ? ", %d" : "%d", len);
  }
  /* The end trim stays what it was: the padding the source decodes past its last granule, after the new total. */
  int64_t final_granule = st->last_granule + (total_samples - source_samples);
  ogg_pageout(&w, 1, 1, final_granule);
  fclose(w.out);
  opus_repacketizer_destroy(rp);
  printf("], \"pre_skip\": %d, \"final_granule\": %lld}\n", h.pre_skip, (long long)final_granule);
  return 0;
}

/* ---- range decoder traces ---- */

static void state(ec_dec *d) {
  printf(", \"rng\": %u, \"val\": %u, \"tell\": %d, \"tell_frac\": %u}", d->rng, d->val, ec_tell(d), ec_tell_frac(d));
}

static int cmd_symbols(const char *in_path, int npackets) {
  long size;
  unsigned char *file = read_file(in_path, &size);
  Ogg ogg = ogg_read(file, size);
  Stream *st = &ogg.streams[0];
  printf("{\"packets\": [");
  for (int k = 2; k < st->npackets && k < 2 + npackets; k++) {
    Packet *pk = &st->packets[k];
    const unsigned char *frames[48];
    opus_int16 sizes[48];
    unsigned char toc;
    int payload = 0;
    int count = opus_packet_parse(pk->data, pk->len, &toc, frames, sizes, &payload);
    int config = toc >> 3, stereo = (toc >> 2) & 1;
    printf("%s\n {\"packet\": %d, \"config\": %d, \"mode\": \"%s\", \"frame_bytes\": %d, \"frame_hex\": \"", k > 2 ? "," : "", k - 2, config, mode_name(config), count > 0 ? sizes[0] : 0);
    for (int i = 0; count > 0 && i < sizes[0]; i++) printf("%02x", frames[0][i]);
    printf("\", \"symbols\": [");
    if (count <= 0 || sizes[0] <= 1) { printf("]}"); continue; }
    ec_dec d;
    ec_dec_init(&d, (unsigned char *)frames[0], sizes[0]);
    printf("\n  {\"op\": \"init\", \"value\": 0");
    state(&d);
    if (config < 16) {
      /* silk_Decode on a packet's first frame: for each channel, a VAD flag per SILK frame then the LBRR flag. */
      int tenths = frame_tenths(config), nframes = tenths <= 200 ? 1 : tenths / 200;
      for (int c = 0; c <= stereo; c++) {
        for (int i = 0; i < nframes; i++) {
          int v = ec_dec_bit_logp(&d, 1);
          printf(",\n  {\"op\": \"bit_logp\", \"logp\": 1, \"what\": \"vad ch%d frame%d\", \"value\": %d", c, i, v);
          state(&d);
        }
        int v = ec_dec_bit_logp(&d, 1);
        printf(",\n  {\"op\": \"bit_logp\", \"logp\": 1, \"what\": \"lbrr ch%d\", \"value\": %d", c, v);
        state(&d);
      }
    } else {
      /* celt_decode_with_ec on a CELT-only frame (start band 0). */
      int lm = config % 4, total = sizes[0] * 8, tell = ec_tell(&d), silence = 0;
      if (tell == 1) {
        silence = ec_dec_bit_logp(&d, 15);
        printf(",\n  {\"op\": \"bit_logp\", \"logp\": 15, \"what\": \"silence\", \"value\": %d", silence);
        state(&d);
      }
      if (!silence) {
        if (tell + 16 <= total) {
          int pf = ec_dec_bit_logp(&d, 1);
          printf(",\n  {\"op\": \"bit_logp\", \"logp\": 1, \"what\": \"postfilter\", \"value\": %d", pf);
          state(&d);
          if (pf) {
            int octave = (int)ec_dec_uint(&d, 6);
            printf(",\n  {\"op\": \"uint\", \"ft\": 6, \"what\": \"postfilter octave\", \"value\": %d", octave);
            state(&d);
            int bits = (int)ec_dec_bits(&d, 4 + octave);
            printf(",\n  {\"op\": \"bits\", \"bits\": %d, \"what\": \"postfilter period bits\", \"value\": %d", 4 + octave, bits);
            state(&d);
            int qg = (int)ec_dec_bits(&d, 3);
            printf(",\n  {\"op\": \"bits\", \"bits\": 3, \"what\": \"postfilter gain\", \"value\": %d", qg);
            state(&d);
            if (ec_tell(&d) + 2 <= total) {
              int t = ec_dec_icdf(&d, tapset_icdf, 2);
              printf(",\n  {\"op\": \"icdf\", \"icdf\": [2, 1, 0], \"ftb\": 2, \"what\": \"postfilter tapset\", \"value\": %d", t);
              state(&d);
            }
          }
          tell = ec_tell(&d);
        }
        if (lm > 0 && tell + 3 <= total) {
          int t = ec_dec_bit_logp(&d, 3);
          printf(",\n  {\"op\": \"bit_logp\", \"logp\": 3, \"what\": \"transient\", \"value\": %d", t);
          state(&d);
          tell = ec_tell(&d);
        }
        if (tell + 3 <= total) {
          int t = ec_dec_bit_logp(&d, 3);
          printf(",\n  {\"op\": \"bit_logp\", \"logp\": 3, \"what\": \"intra\", \"value\": %d", t);
          state(&d);
        }
      }
    }
    printf("]}");
  }
  printf("]}\n");
  return 0;
}

/* A fixed LCG (Numerical Recipes' constants), so the sequences are the same everywhere. */
static uint32_t seed;
static uint32_t next(void) { seed = seed * 1664525u + 1013904223u; return seed; }

typedef struct { int op; uint32_t a, b; uint32_t value; } Sym;
enum { OP_BIT_LOGP, OP_ICDF, OP_UINT, OP_BITS, OP_LAPLACE };

static const unsigned char icdf_tables[4][12] = {
  { 2, 1, 0 },                                         /* celt tapset, ftb 2 */
  { 25, 23, 2, 0 },                                    /* celt spread, ftb 5 */
  { 126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0 },      /* celt trim, ftb 7 */
  { 232, 158, 10, 0 },                                 /* silk type/offset with VAD, ftb 8 */
};
static const int icdf_ftb[4] = { 2, 5, 7, 8 }, icdf_len[4] = { 3, 4, 11, 4 };

static void sequence(const char *name, uint32_t s, int nsyms, int storage, int raw) {
  seed = s;
  Sym *syms = grow(NULL, sizeof(Sym) * nsyms);
  unsigned char *buf = calloc(storage, 1);
  ec_enc e;
  ec_enc_init(&e, buf, storage);
  int n = 0;
  for (; n < nsyms; n++) {
    Sym *y = &syms[n];
    /* range-only: bit_logp, icdf, laplace; mixed: those and uint, bits (raw bits from the end). */
    static const int range_ops[3] = { OP_BIT_LOGP, OP_ICDF, OP_LAPLACE };
    y->op = raw ? (int)(next() % 5) : range_ops[next() % 3];
    switch (y->op) {
      case OP_BIT_LOGP: y->a = 1 + next() % 15; y->value = (next() % (1u << y->a)) == 0; ec_enc_bit_logp(&e, (int)y->value, y->a); break;
      case OP_ICDF: { int t = (int)(next() % 4); y->a = (uint32_t)t; y->value = next() % icdf_len[t];
        while (y->value > 0 && icdf_tables[t][y->value - 1] == icdf_tables[t][y->value]) y->value--;
        ec_enc_icdf(&e, (int)y->value, icdf_tables[t], icdf_ftb[t]); break; }
      case OP_UINT: { uint32_t ft = (next() & 1) ? 2 + next() % 300 : 2 + next() % 0xFFFFFFF0u; y->a = ft; y->value = next() % ft; ec_enc_uint(&e, y->value, ft); break; }
      case OP_BITS: y->a = 1 + next() % 16; y->value = next() & ((1u << y->a) - 1); ec_enc_bits(&e, y->value, y->a); break;
      case OP_LAPLACE: { static const int fs[3] = { 72, 127, 42 }, decay[3] = { 127, 112, 89 }; int t = (int)(next() % 3);
        y->a = (uint32_t)fs[t] << 7; y->b = (uint32_t)decay[t] << 6; int v = (int)(next() % 9) - 4; ec_laplace_encode(&e, &v, y->a, (int)y->b); y->value = (uint32_t)v; break; }
    }
    if (ec_tell(&e) + 48 > storage * 8) { n++; break; }
  }
  uint32_t enc_rng = e.rng;
  ec_enc_done(&e);
  if (e.error) fail("rangecoder %s: the encoder ran out of room", name);
  printf("%s\n {\"name\": \"%s\", \"seed\": %u, \"storage\": %d, \"encoder_final_range\": %u, \"hex\": \"", strcmp(name, "range-only") ? ",\n" : "", name, s, storage, enc_rng);
  for (int i = 0; i < storage; i++) printf("%02x", buf[i]);
  printf("\", \"symbols\": [");
  ec_dec d;
  ec_dec_init(&d, buf, storage);
  printf("\n  {\"op\": \"init\", \"value\": 0");
  state(&d);
  for (int i = 0; i < n; i++) {
    Sym *y = &syms[i];
    uint32_t v = 0;
    switch (y->op) {
      case OP_BIT_LOGP: v = (uint32_t)ec_dec_bit_logp(&d, y->a); printf(",\n  {\"op\": \"bit_logp\", \"logp\": %u", y->a); break;
      case OP_ICDF: v = (uint32_t)ec_dec_icdf(&d, icdf_tables[y->a], icdf_ftb[y->a]); printf(",\n  {\"op\": \"icdf\", \"icdf\": [");
        for (int j = 0; j < icdf_len[y->a]; j++) printf(j ? ", %d" : "%d", icdf_tables[y->a][j]);
        printf("], \"ftb\": %d", icdf_ftb[y->a]); break;
      case OP_UINT: v = ec_dec_uint(&d, y->a); printf(",\n  {\"op\": \"uint\", \"ft\": %u", y->a); break;
      case OP_BITS: v = ec_dec_bits(&d, y->a); printf(",\n  {\"op\": \"bits\", \"bits\": %u", y->a); break;
      case OP_LAPLACE: v = (uint32_t)ec_laplace_decode(&d, y->a, (int)y->b); printf(",\n  {\"op\": \"laplace\", \"fs\": %u, \"decay\": %u", y->a, y->b); break;
    }
    if (v != y->value)
      fail("rangecoder %s: symbol %d decoded %u, not what was encoded", name, i, v);
    if (y->op == OP_LAPLACE) printf(", \"value\": %d", (int)v);
    else printf(", \"value\": %u", v);
    state(&d);
  }
  if (d.rng != enc_rng) fail("rangecoder %s: the decoder's final range differs from the encoder's", name);
  printf("]}");
  free(syms); free(buf);
}

static int cmd_rangecoder(void) {
  printf("{\"sequences\": [");
  sequence("range-only", 20261024u, 160, 160, 0);
  sequence("mixed", 32u, 200, 400, 1);
  sequence("tight", 7u, 100000, 40, 1);
  printf("]}\n");
  return 0;
}

int main(int argc, char **argv) {
  crc_init();
  if (argc >= 5 && !strcmp(argv[1], "encode")) return cmd_encode(argv[2], argv[3], argv[4]);
  if (argc >= 8 && !strcmp(argv[1], "decode")) return cmd_decode(argc, argv);
  if (argc == 7 && !strcmp(argv[1], "repack")) return cmd_repack(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), (uint32_t)strtoul(argv[6], NULL, 10));
  if (argc == 4 && !strcmp(argv[1], "symbols")) return cmd_symbols(argv[2], atoi(argv[3]));
  if (argc == 2 && !strcmp(argv[1], "rangecoder")) return cmd_rangecoder();
  fprintf(stderr, "usage: %s encode CONFIG IN.f32 OUT.opus | decode IN.opus RATE LINK GAIN OUT.f32 OUT.i16 [FRAMES.f32] |\n"
                  "       repack IN.opus OUT.opus GROUP PAD SERIAL | symbols IN.opus PACKETS | rangecoder\n", argv[0]);
  return 1;
}
