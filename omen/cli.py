"""Command-line interface.

A thin dispatch layer: each subcommand parses its arguments, constructs the
relevant domain objects (:class:`ModelTrainer`, :class:`PyEnumerator`,
:class:`PasswordScorer`, :class:`ModelInspector`), and delegates.  No business
logic lives here.

Subcommands::

    omen train    -i CORPUS -m MODEL [--ngram N] [--levels NL] ...
    omen generate -m MODEL [--max-guesses N] [--max-level L] ...   # streams to stdout
    omen eval     -m MODEL [-i FILE | PW ...] [--rank]
    omen alphabet -i CORPUS [--size K]
    omen inspect  -m MODEL
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from omen.alphabet import select_alphabet
from omen.enumerate import PyEnumerator
from omen.errors import OmenError
from omen.inspect import ModelInspector
from omen.io_utils import ByteSink, read_corpus
from omen.model import NgramModel
from omen.score import PasswordScorer
from omen.train import ModelTrainer, TrainingOptions

CorpusFactory = Callable[[], Iterable[str]]


def main(argv: list[str] | None = None) -> int:
    """Program entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except BrokenPipeError:
        # Downstream consumer (e.g. hashcat) closed the pipe: a normal end.
        _silence_broken_pipe()
        return 0
    except OmenError as exc:
        print(f"omen: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


# -- parser ----------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omen",
        description="OMEN — ordered Markov password candidate generator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a model from a password corpus")
    p_train.add_argument("-i", "--input", required=True, help="corpus path, or '-' for stdin")
    p_train.add_argument("-m", "--model", required=True, help="output model directory")
    p_train.add_argument("--ngram", type=int, default=3, help="n-gram order (default: 3)")
    p_train.add_argument("--levels", type=int, default=11, help="number of levels (default: 11)")
    p_train.add_argument(
        "--smoothing", type=float, default=0.01, help="add-δ smoothing (default: 0.01)"
    )
    p_train.add_argument(
        "--max-length", type=int, default=20, help="maximum password length (default: 20)"
    )
    alpha_group = p_train.add_mutually_exclusive_group()
    alpha_group.add_argument(
        "--alphabet", help="explicit alphabet string (overrides automatic selection)"
    )
    alpha_group.add_argument(
        "--alphabet-size",
        type=int,
        default=72,
        help="number of most-frequent chars to auto-select (default: 72)",
    )
    p_train.add_argument(
        "--no-ep", action="store_true", help="disable the word-ending (EP) component"
    )
    p_train.set_defaults(handler=_cmd_train)

    p_gen = sub.add_parser("generate", help="stream ranked candidates to stdout")
    p_gen.add_argument("-m", "--model", required=True, help="model directory")
    p_gen.add_argument("--max-guesses", type=int, default=None, help="stop after N candidates")
    p_gen.add_argument("--max-level", type=int, default=None, help="stop after total level L")
    p_gen.add_argument("--min-length", type=int, default=None, help="minimum candidate length")
    p_gen.add_argument("--max-length", type=int, default=None, help="maximum candidate length")
    p_gen.set_defaults(handler=_cmd_generate)

    p_eval = sub.add_parser("eval", help="score passwords against a model")
    p_eval.add_argument("-m", "--model", required=True, help="model directory")
    p_eval.add_argument("-i", "--input", help="file of passwords, or '-' for stdin")
    p_eval.add_argument("passwords", nargs="*", help="passwords to score")
    p_eval.add_argument("--rank", action="store_true", help="also estimate guess rank")
    p_eval.add_argument(
        "--rank-cap", type=int, default=10_000_000, help="cap for rank estimation work"
    )
    p_eval.set_defaults(handler=_cmd_eval)

    p_alpha = sub.add_parser("alphabet", help="select and report an alphabet from a corpus")
    p_alpha.add_argument("-i", "--input", required=True, help="corpus path, or '-' for stdin")
    p_alpha.add_argument("--size", type=int, default=72, help="alphabet size (default: 72)")
    p_alpha.set_defaults(handler=_cmd_alphabet)

    p_inspect = sub.add_parser("inspect", help="print model metadata and histograms")
    p_inspect.add_argument("-m", "--model", required=True, help="model directory")
    p_inspect.set_defaults(handler=_cmd_inspect)

    return parser


# -- command handlers ------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    options = TrainingOptions(
        ngram=args.ngram,
        levels=args.levels,
        smoothing=args.smoothing,
        max_length=args.max_length,
        alphabet=args.alphabet,
        alphabet_size=args.alphabet_size,
        ep_enabled=not args.no_ep,
    )
    factory = _corpus_factory(args.input)
    model = ModelTrainer(options).train(factory)
    model.save(args.model)
    print(
        f"omen: trained model saved to {args.model} "
        f"(alphabet={model.alphabet_size}, coverage={model.coverage:.1%}, "
        f"lam={model.scale.lam:.4g})",
        file=sys.stderr,
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    model = NgramModel.load(args.model)
    enumerator = PyEnumerator(model)
    with ByteSink(sys.stdout.buffer) as sink:
        enumerator.stream(
            sink,
            max_guesses=args.max_guesses,
            max_level=args.max_level,
            min_length=args.min_length,
            max_length=args.max_length,
        )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    model = NgramModel.load(args.model)
    scorer = PasswordScorer(model)
    passwords = _eval_inputs(args)
    for pw in passwords:
        result = scorer.score(pw)
        if not result.in_model:
            print(f"{pw}\tUNREACHABLE\t({result.reason})")
            continue
        line = (
            f"{pw}\tlevel={result.total_level}\tlen={result.length}\t"
            f"ip={result.ip_level} cp={result.cp_sum} ep={result.ep_level} ln={result.ln_level}"
        )
        if args.rank:
            rank = scorer.estimate_rank(pw, cap=args.rank_cap)
            line += f"\trank>={rank}"
        print(line)
    return 0


def _cmd_alphabet(args: argparse.Namespace) -> int:
    with read_corpus(args.input) as passwords:
        selection = select_alphabet(passwords, args.size)
    print(f"size      : {selection.alphabet.size}")
    print(f"coverage  : {selection.coverage:.4%}")
    print(f"distinct  : {selection.distinct_chars}")
    print(f"total     : {selection.total_chars}")
    print(f"alphabet  : {selection.alphabet.as_string()}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    model = NgramModel.load(args.model)
    print(ModelInspector(model).report())
    return 0


# -- helpers ---------------------------------------------------------------


def _corpus_factory(source: str) -> CorpusFactory:
    """Return a factory yielding a fresh password iterator on each call.

    Training needs up to two passes (alphabet selection, then counting).  For a
    file we re-open on each pass; for stdin (single-shot) we materialise once.
    """
    if source == "-":
        with read_corpus(source) as passwords:
            buffered = list(passwords)
        return lambda: iter(buffered)

    path = Path(source)

    def factory() -> Iterator[str]:
        with read_corpus(path) as passwords:
            yield from passwords

    return factory


def _eval_inputs(args: argparse.Namespace) -> list[str]:
    if args.input:
        with read_corpus(args.input) as passwords:
            return list(passwords)
    positional: list[str] = list(args.passwords)
    if positional:
        return positional
    raise OmenError("provide passwords as arguments or via -i/--input")


def _silence_broken_pipe() -> None:
    """Redirect stdout to /dev/null so interpreter shutdown won't re-raise.

    Without this, Python flushes stdout at exit, hits the broken pipe again, and
    prints a noisy 'Exception ignored' message.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except OSError:
        pass
