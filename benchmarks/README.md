# Benchmarks

Reproducible comparison of guessing efficiency: **how many held-out passwords a
method cracks as a function of the number of guesses it makes.** This is the
standard offline-guessing metric used in the OMEN and PCFG papers.

## Methodology

- **Dataset:** `rockyou.txt` (14,344,391 lines).
- **Split:** each password is assigned to *train* or *test* by
  `md5(password) % 10` — bucket 0 is the held-out **test** set, buckets 1–9 are
  **train**. Because the bucket is a function of the password itself, the same
  password never lands in both sides: the test set is **disjoint** from training
  (no leakage). Result: ~12.89M train, ~1.46M unique test.
- **Same data for every method.** OMEN trains its model on the train split;
  PRINCE and the wordlist baseline use the same train split as their input
  wordlist. No method sees a test password during preparation.
- **Metric:** cumulative count of unique *test* passwords matched, sampled at
  log-spaced guess counts up to 20,000,000. Crack rate = matched / 1,456,297.
- **Common denominator.** The full test set is the denominator for every method
  (including passwords OMEN cannot represent, e.g. length > 20 or out-of-alphabet
  characters), so no method is given a handicap.

## Methods

| Method | Source of guesses |
|--------|-------------------|
| `OMEN n=3` | This tool, order-3 model, default settings. |
| `OMEN n=4` | This tool, order-4 model. |
| `PRINCE (train words)` | `pp64` over the deduplicated training words (same data OMEN trained on). |
| `PRINCE (top-100k)` | `pp64` over the 100k most-frequent training words (denser base; friendlier to PRINCE's real-world use). |
| `rockyou wordlist` | The training split replayed verbatim — a **control**. Since train/test are disjoint, this must score ~0%; any higher would indicate leakage. |

## Reproduce

```bash
# 0. deps (benchmark-only; the omen runtime is stdlib-only)
pip install -e ".[dev]" matplotlib

# 1. split rockyou into disjoint train/test
python benchmarks/make_split.py ~/wordlists/rockyou.txt /tmp/ry_train.txt /tmp/ry_test.txt

# 2. train OMEN models
omen train -i /tmp/ry_train.txt -m /tmp/ry_model  --ngram 3 --max-length 20
omen train -i /tmp/ry_train.txt -m /tmp/ry_model4 --ngram 4 --max-length 20

# 3. build PRINCE input wordlists
awk '!seen[$0]++' /tmp/ry_train.txt > /tmp/ry_words_unique.txt
head -100000 /tmp/ry_train.txt          > /tmp/ry_words_top100k.txt

# 4. run each method (results merge into benchmarks/results.json)
python benchmarks/run_benchmark.py omen     --model /tmp/ry_model  --label "OMEN n=3"
python benchmarks/run_benchmark.py omen     --model /tmp/ry_model4 --label "OMEN n=4"
python benchmarks/run_benchmark.py wordlist --wordlist /tmp/ry_train.txt --label "rockyou wordlist"
python benchmarks/run_benchmark.py prince   --prince-bin ./pp64.bin --wordlist /tmp/ry_words_top100k.txt --label "PRINCE (top-100k)"
python benchmarks/run_benchmark.py prince   --prince-bin ./pp64.bin --wordlist /tmp/ry_words_unique.txt --label "PRINCE (train words)"

# 5. render graphs into docs/img/
python benchmarks/plot_results.py
```

## Interpreting the result

OMEN generates novel character sequences from a Markov model, so it cracks
held-out passwords it has never seen. PRINCE and the raw wordlist only
recombine / replay existing words, so on a disjoint split their reach is
limited — which is exactly why OMEN wins on *ordered guessing efficiency*.
PRINCE's practical strengths (no training, direct hashcat pipelining, no
alphabet/length limits) are real but orthogonal to this metric.

Hardware note: OMEN figures use the pure-Python reference enumerator
(~10⁵ candidates/s). The ordering — not the raw speed — is what these graphs
measure; a native enumerator would produce identical curves faster.
