/*
 * omen-enum — native (C) Ordered Markov ENumerator.
 *
 * Reads a model directory produced by `omen train` (manifest.bin + mmap'd
 * ip.dat / cp.dat / ep.dat) and streams candidate passwords to stdout in
 * exactly the same order as the pure-Python PyEnumerator — so the two are
 * byte-for-byte interchangeable, just ~10-100x faster.
 *
 * Algorithm (mirrors omen/enumerate.py):
 *   total level = IP(initial ctx) + sum CP(transition) + EP(final ctx) + LN(len)
 *   walk total = 0,1,2,...; for each length; emit every candidate whose total
 *   equals the current level before advancing -> globally non-decreasing order.
 *
 * Build:  make           (see native/Makefile)
 * Usage:  omen-enum <model_dir> [--max-guesses N] [--max-level L]
 *                               [--min-length L] [--max-length L]
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

/* Must match omen/model.py: MANIFEST_MAGIC, FORMAT_VERSION, MAX_TABLE_ENTRIES. */
#define MANIFEST_MAGIC "OMN1"
#define FORMAT_VERSION 1
#define MAX_TABLE_ENTRIES (1u << 28)
#define OUT_FLUSH_AT (1u << 20)
#define OUT_CAP (OUT_FLUSH_AT + 4096)

/* One continuation option: emit `code` after the current context, at `level`. */
typedef struct {
    u8 code;
    u8 level;
} CodeLvl;

typedef struct {
    /* shape */
    int ngram, ctx_len, A, levels, max_length, ep_enabled, max_level;
    u32 num_contexts; /* A^(ctx_len)   */
    u64 cp_stride;    /* == A          */
    u64 drop_mod;     /* A^(ctx_len-1) */

    /* mmap'd level tables (read-only) */
    const u8 *ip, *cp, *ep;
    size_t ip_sz, cp_sz, ep_sz;

    /* from manifest */
    u8 *ln;          /* length levels, max_length+1 entries */
    u8 *alpha_utf8;  /* concatenated UTF-8 of the alphabet  */
    u32 *code_off;   /* per code: offset into alpha_utf8    */
    u8 *code_len;    /* per code: UTF-8 byte length         */

    /* derived bounds for pruning */
    int min_cp, max_cp, min_ep, max_ep, ip_max;

    /* IP contexts grouped by initial level (ctx ascending within a level) */
    u32 **ip_bucket;
    u32 *ip_bucket_cnt;

    /* lazily-built per-context continuation order, sorted by (level, code) */
    CodeLvl **cp_cache;

    /* output + limits */
    u8 *out;
    size_t out_len;
    u64 emitted, limit; /* limit == 0 means unlimited */
    int stop;
} Model;

static void die(const char *msg) {
    fprintf(stderr, "omen-enum: %s\n", msg);
    exit(2);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "omen-enum: %s: %s\n", msg, strerror(errno));
    exit(2);
}

/* base^exp with the same overflow guard as the Python model loader. */
static u64 checked_pow(u64 base, int exp, const char *what) {
    u64 r = 1;
    for (int i = 0; i < exp; i++) {
        if (base != 0 && r > MAX_TABLE_ENTRIES / base) die(what);
        r *= base;
    }
    if (r > MAX_TABLE_ENTRIES) die(what);
    return r;
}

static const u8 *map_table(const char *dir, const char *name, size_t *out_sz) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) die_errno(path);
    struct stat st;
    if (fstat(fd, &st) != 0) die_errno(path);
    size_t sz = (size_t)st.st_size;
    const u8 *p = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) die_errno("mmap");
    *out_sz = sz;
    return p;
}

/* Read the whole (small) manifest into a malloc'd buffer. */
static u8 *read_file(const char *dir, const char *name, size_t *out_sz) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) die_errno(path);
    if (fseek(f, 0, SEEK_END) != 0) die_errno(path);
    long sz = ftell(f);
    if (sz < 0) die_errno(path);
    rewind(f);
    u8 *buf = malloc((size_t)sz);
    if (!buf) die("out of memory reading manifest");
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) die("short read on manifest");
    fclose(f);
    *out_sz = (size_t)sz;
    return buf;
}

static u32 rd_u32(const u8 *p) {
    u32 v;
    memcpy(&v, p, 4);
    return v; /* file is little-endian; assume LE host (x86-64 rig) */
}

static void load_manifest(Model *m, const char *dir) {
    size_t sz;
    u8 *buf = read_file(dir, "manifest.bin", &sz);
    if (sz < 44) die("manifest too small");
    if (memcmp(buf, MANIFEST_MAGIC, 4) != 0) die("bad manifest magic");
    if (rd_u32(buf + 4) != FORMAT_VERSION) die("unsupported manifest version");
    m->ngram = (int)rd_u32(buf + 8);
    m->levels = (int)rd_u32(buf + 12);
    m->max_length = (int)rd_u32(buf + 16);
    m->ep_enabled = (int)rd_u32(buf + 20);
    m->A = (int)rd_u32(buf + 24);
    u32 ln_len = rd_u32(buf + 28);
    u32 alpha_bytes = rd_u32(buf + 32);
    /* lam at offset 36 (f64) is not needed for enumeration; skip it. */

    if (m->ngram < 2 || m->ngram > 5) die("ngram out of range");
    if (m->levels < 2 || m->levels > 256) die("levels out of range");
    if (m->A < 1 || m->A > 256) die("alphabet size out of range");
    if ((int)ln_len != m->max_length + 1) die("ln length mismatch");

    size_t need = 44 + (size_t)m->A + alpha_bytes + ln_len;
    if (sz != need) die("manifest size mismatch");

    m->ctx_len = m->ngram - 1;
    m->max_level = m->levels - 1;
    m->num_contexts = (u32)checked_pow((u64)m->A, m->ctx_len, "ip/ep table too large");
    m->cp_stride = (u64)m->A;
    m->drop_mod = checked_pow((u64)m->A, m->ctx_len - 1, "context shift too large");

    const u8 *code_lens = buf + 44;
    const u8 *alpha = code_lens + m->A;
    const u8 *ln = alpha + alpha_bytes;

    m->alpha_utf8 = malloc(alpha_bytes ? alpha_bytes : 1);
    m->code_off = malloc(sizeof(u32) * (size_t)m->A);
    m->code_len = malloc((size_t)m->A);
    m->ln = malloc(ln_len);
    if (!m->alpha_utf8 || !m->code_off || !m->code_len || !m->ln)
        die("out of memory for manifest tables");
    memcpy(m->alpha_utf8, alpha, alpha_bytes);
    memcpy(m->ln, ln, ln_len);
    u32 off = 0;
    for (int c = 0; c < m->A; c++) {
        m->code_len[c] = code_lens[c];
        m->code_off[c] = off;
        off += code_lens[c];
    }
    if (off != alpha_bytes) die("alphabet byte length mismatch");
    for (u32 i = 0; i < ln_len; i++)
        if (m->ln[i] > m->max_level) die("ln level out of range");
    free(buf);
}

static void load_tables(Model *m, const char *dir) {
    m->ip = map_table(dir, "ip.dat", &m->ip_sz);
    m->cp = map_table(dir, "cp.dat", &m->cp_sz);
    m->ep = map_table(dir, "ep.dat", &m->ep_sz);
    if (m->ip_sz != m->num_contexts) die("ip.dat size mismatch");
    if (m->ep_sz != m->num_contexts) die("ep.dat size mismatch");
    if (m->cp_sz != (size_t)m->num_contexts * (size_t)m->A) die("cp.dat size mismatch");
}

static void compute_bounds(Model *m) {
    int lo = 255, hi = 0;
    for (size_t i = 0; i < m->cp_sz; i++) {
        int v = m->cp[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    m->min_cp = lo;
    m->max_cp = hi;
    if (m->ep_enabled) {
        lo = 255;
        hi = 0;
        for (u32 i = 0; i < m->num_contexts; i++) {
            int v = m->ep[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        m->min_ep = lo;
        m->max_ep = hi;
    } else {
        m->min_ep = 0;
        m->max_ep = 0;
    }
    hi = 0;
    for (u32 i = 0; i < m->num_contexts; i++)
        if (m->ip[i] > hi) hi = m->ip[i];
    m->ip_max = hi;
}

static void build_ip_buckets(Model *m) {
    m->ip_bucket = calloc((size_t)m->levels, sizeof(u32 *));
    m->ip_bucket_cnt = calloc((size_t)m->levels, sizeof(u32));
    if (!m->ip_bucket || !m->ip_bucket_cnt) die("out of memory for ip buckets");
    for (u32 ctx = 0; ctx < m->num_contexts; ctx++) m->ip_bucket_cnt[m->ip[ctx]]++;
    u32 *fill = calloc((size_t)m->levels, sizeof(u32));
    if (!fill) die("out of memory for ip buckets");
    for (int l = 0; l < m->levels; l++) {
        if (m->ip_bucket_cnt[l]) {
            m->ip_bucket[l] = malloc(sizeof(u32) * m->ip_bucket_cnt[l]);
            if (!m->ip_bucket[l]) die("out of memory for ip buckets");
        }
    }
    /* ctx ascending -> matches Python's enumerate(model.ip_levels) order */
    for (u32 ctx = 0; ctx < m->num_contexts; ctx++) {
        u8 l = m->ip[ctx];
        m->ip_bucket[l][fill[l]++] = ctx;
    }
    free(fill);
}

static int codelvl_cmp(const void *a, const void *b) {
    const CodeLvl *x = a, *y = b;
    if (x->level != y->level) return (int)x->level - (int)y->level;
    return (int)x->code - (int)y->code;
}

/* Per-context continuation order, sorted by (level, code); built once, cached. */
static CodeLvl *cp_get(Model *m, u32 ctx) {
    CodeLvl *c = m->cp_cache[ctx];
    if (c) return c;
    c = malloc(sizeof(CodeLvl) * (size_t)m->A);
    if (!c) die("out of memory for cp cache");
    const u8 *row = m->cp + (size_t)ctx * (size_t)m->A;
    for (int code = 0; code < m->A; code++) {
        c[code].code = (u8)code;
        c[code].level = row[code];
    }
    qsort(c, (size_t)m->A, sizeof(CodeLvl), codelvl_cmp);
    m->cp_cache[ctx] = c;
    return c;
}

static void flush_out(Model *m) {
    size_t done = 0;
    while (done < m->out_len) {
        ssize_t w = write(STDOUT_FILENO, m->out + done, m->out_len - done);
        if (w < 0) {
            if (errno == EINTR) continue;
            if (errno == EPIPE) {
                /* consumer (e.g. hashcat) closed the pipe: a normal end */
                m->out_len = 0;
                m->stop = 1;
                return;
            }
            die_errno("write");
        }
        done += (size_t)w;
    }
    m->out_len = 0;
}

static void emit(Model *m, const u8 *codes, int len) {
    u8 *o = m->out + m->out_len;
    for (int i = 0; i < len; i++) {
        u8 code = codes[i];
        u8 cl = m->code_len[code];
        memcpy(o, m->alpha_utf8 + m->code_off[code], cl);
        o += cl;
    }
    *o++ = '\n';
    m->out_len = (size_t)(o - m->out);
    m->emitted++;
    if (m->limit && m->emitted >= m->limit) m->stop = 1;
    if (m->out_len >= OUT_FLUSH_AT) flush_out(m);
}

static void recurse(Model *m, u32 ctx, u8 *codes, int pos, int tl, int budget) {
    if (tl == 0) {
        int ep = m->ep_enabled ? m->ep[ctx] : 0;
        if (ep == budget) emit(m, codes, pos);
        return;
    }
    int tt = tl - 1;
    int low = tt * m->min_cp + m->min_ep;
    int high = tt * m->max_cp + m->max_ep;
    CodeLvl *cc = cp_get(m, ctx);
    for (int i = 0; i < m->A; i++) {
        int level = cc[i].level;
        if (level > budget) break; /* sorted ascending: nothing further fits */
        int rem = budget - level;
        if (rem < low || rem > high) continue;
        u8 code = cc[i].code;
        u32 nctx = (u32)((ctx % m->drop_mod) * (u64)m->A + code);
        codes[pos] = code;
        recurse(m, nctx, codes, pos + 1, tt, rem);
        if (m->stop) return;
    }
}

static void unpack_ctx(const Model *m, u32 ctx, u8 *codes) {
    for (int i = m->ctx_len - 1; i >= 0; i--) {
        codes[i] = (u8)(ctx % (u32)m->A);
        ctx /= (u32)m->A;
    }
}

static void enumerate_length(Model *m, int length, int budget, u8 *codes) {
    int transitions = length - m->ctx_len;
    int tail_low = transitions * m->min_cp + m->min_ep;
    int tail_high = transitions * m->max_cp + m->max_ep;
    int a_lo = budget - tail_high;
    if (a_lo < 0) a_lo = 0;
    int a_hi = budget - tail_low;
    if (a_hi > m->max_level) a_hi = m->max_level;
    for (int a = a_lo; a <= a_hi; a++) {
        u32 cnt = m->ip_bucket_cnt[a];
        if (!cnt) continue;
        u32 *ctxs = m->ip_bucket[a];
        int remaining = budget - a;
        for (u32 j = 0; j < cnt; j++) {
            unpack_ctx(m, ctxs[j], codes);
            recurse(m, ctxs[j], codes, m->ctx_len, transitions, remaining);
            if (m->stop) return;
        }
    }
}

static int total_level_ceiling(const Model *m, int lo, int hi) {
    int ceiling = 0;
    for (int length = lo; length <= hi; length++) {
        int transitions = length - m->ctx_len;
        int per = m->ln[length] + m->ip_max + transitions * m->max_cp + m->max_ep;
        if (per > ceiling) ceiling = per;
    }
    return ceiling;
}

static void run(Model *m, int min_len, int max_len, int max_level) {
    int lo = (min_len < m->ctx_len) ? m->ctx_len : min_len;
    int hi = (max_len < 0 || max_len > m->max_length) ? m->max_length : max_len;
    if (lo > hi) return;

    int ceiling = total_level_ceiling(m, lo, hi);
    int t_cap = (max_level < 0) ? ceiling : (max_level < ceiling ? max_level : ceiling);

    u8 *codes = malloc((size_t)m->max_length + 1);
    if (!codes) die("out of memory for candidate buffer");
    for (int total = 0; total <= t_cap; total++) {
        for (int length = lo; length <= hi; length++) {
            int budget = total - m->ln[length];
            if (budget < 0) continue;
            enumerate_length(m, length, budget, codes);
            if (m->stop) goto done;
        }
    }
done:
    flush_out(m);
    free(codes);
}

static u64 parse_u64(const char *s, const char *flag) {
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        fprintf(stderr, "omen-enum: invalid value for %s: %s\n", flag, s);
        exit(2);
    }
    return (u64)v;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: omen-enum <model_dir> [--max-guesses N] [--max-level L]\n"
                "                             [--min-length L] [--max-length L]\n");
        return 2;
    }
    const char *dir = argv[1];
    u64 max_guesses = 0; /* 0 = unlimited */
    int max_level = -1, min_len = 0, max_len = -1;

    int i = 2;
    while (i < argc) {
        const char *opt = argv[i++];
        if (i >= argc) die("missing value for option");
        const char *v = argv[i++];
        if (strcmp(opt, "--max-guesses") == 0) max_guesses = parse_u64(v, "--max-guesses");
        else if (strcmp(opt, "--max-level") == 0) max_level = (int)parse_u64(v, "--max-level");
        else if (strcmp(opt, "--min-length") == 0) min_len = (int)parse_u64(v, "--min-length");
        else if (strcmp(opt, "--max-length") == 0) max_len = (int)parse_u64(v, "--max-length");
        else die("unknown option");
    }

    Model m;
    memset(&m, 0, sizeof(m));
    m.limit = max_guesses;
    load_manifest(&m, dir);
    load_tables(&m, dir);
    compute_bounds(&m);
    build_ip_buckets(&m);
    m.cp_cache = calloc((size_t)m.num_contexts, sizeof(CodeLvl *));
    if (!m.cp_cache) die("out of memory for cp cache index");
    m.out = malloc(OUT_CAP);
    if (!m.out) die("out of memory for output buffer");

    run(&m, min_len, max_len, max_level);
    return 0;
}
