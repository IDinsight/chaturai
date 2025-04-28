"""This module contains general utilities.

NB: As a general rule of thumb, this module should not import utilities from other
utils modules (in order to avoid circular imports). If a utility function is needed in
multiple modules, then it is a general utility and should be defined in this module
instead.
"""

# Future Library
from __future__ import annotations

# Standard Library
import csv
import functools
import hashlib
import json
import re
import string
import sys
import time

from pathlib import Path
from typing import Any, Awaitable, Callable, NoReturn, ParamSpec, TypeVar

# Third Party Library
from loguru import logger

# Package Library
from naukriwaala.schemas import Valid

# Globals.
P = ParamSpec("P")
T = TypeVar("T")


def cleanup(  # type: ignore
    signum: int, frame  # pylint: disable=unused-argument
) -> NoReturn:
    """Call system exit for clean up.

    Parameters
    ----------
    signum
        Signal number.
    frame
        Current stack frame object.
    """

    print(f"Received signal {signum}. Exiting")
    sys.exit()


def combine_jsonl_files(
    *,
    data_dir: str | Path,
    encoding: str = "utf-8",
    overwrite: bool = False,
    output_fp: str | Path,
) -> None:
    """Combine all JSONL files in `data_dir` into a single JSONL file and save to
    `output_fp`.

    Parameters
    ----------
    data_dir
        Directory containing JSONL files to combine.
    encoding
        The encoding for opening and writing to files.
    overwrite
        Specifies whether to overwrite the existing combined file.
    output_fp
        Filepath for saving the combined JSONL file.
    """

    data_dir = Path(data_dir)
    output_fp = Path(output_fp)
    if Path.is_file(output_fp):
        if not overwrite:
            logger.info(
                f"Combined JSONL file already exists: {output_fp}\n"
                f"Set `overwrite` to `True` if you wish to generate a new combined "
                f"JSONL file."
            )
            return
        output_fp.unlink()
    all_jsonl = [x for fp in data_dir.glob("*.jsonl") for x in open_json_type(fp)]
    write_to_json(output_fp, all_jsonl, encoding=encoding)


def convert_to_list(x: Any) -> list[Any]:
    """Wrap `x` in a list.

    Parameters
    ----------
    x
        Any object to wrap in a list.

    Returns
    -------
    list[Any]
        The passed in `x` wrapped in a list if it's not a list.
    """

    return [x] if not isinstance(x, list) else x


def escape_angle_brackets(x: Any) -> str:
    """Escape angle brackets for colorized logging. If this is not done, then
    `loguru` will throw a `ValueError` when attempting to log objects with angle
    brackets. See: https://github.com/Delgan/loguru/issues/140 for more details.

    Parameters
    ----------
    x
        Any object.

    Returns
    -------
    str
        The string version of `x` with escaped angle brackets.
    """

    return recurse_replace(r"\>", ">", recurse_replace(r"\<", "<", str(x)))


def flatten_list(*, item: str | list | dict) -> list[str]:
    """Recursively flatten nested lists or dicts into a list of strings.

    Parameters
    ----------
    item
        The item to flatten. This can be a string, list, or dictionary.

    Returns
    -------
    list[str]
        A list of strings representing the flattened item.

    Raises
    ------
    TypeError
        If the item is not a string, list, or dictionary.
    """

    results = []
    if isinstance(item, str):
        results.append(item)
    elif isinstance(item, list):
        for sub in item:
            results.extend(flatten_list(item=sub))
    elif isinstance(item, dict):
        # When a dict represents a logic operator (e.g. AND, OR, IF), flatten its
        # values.
        for v in item.values():
            results.extend(flatten_list(item=v))
    else:
        raise TypeError(f"Unsupported type: {type(item)}. Expected str, list, or dict.")
    return results


def generate_unique_id(*, obj: Any) -> str:
    """Generate a unique SHA256-based identifier for any Python object.

    Parameters
    ----------
    obj
        The object to generate a unique identifier for.

    Returns
    -------
    str
        A unique SHA256-based identifier for the object.
    """

    if isinstance(obj, (list, tuple, dict, set)):
        obj_str = json.dumps(obj, sort_keys=True)
    elif isinstance(obj, str):
        obj_str = obj
    else:
        obj_str = json.dumps(obj, default=str)
    return hashlib.sha256(obj_str.encode()).hexdigest()


def is_uppercase_letter(*, s: str) -> bool:
    """Check if a string is a single uppercase letter.

    Parameters
    ----------
    s
        The string to check.

    Returns
    -------
    bool
        Specifies whether the string is a single uppercase letter.
    """

    return len(s) == 1 and s.isalpha() and s.isupper()


def make_dir(dir_: str | Path, verbose: bool = True) -> None:
    """Create a directory.

    Parameters
    ----------
    dir_
        Directory to create.
    verbose
        Specifies whether to log directory creation.
    """

    dir_ = Path(dir_)
    if not Path.is_dir(dir_):
        if verbose:
            logger.info(f"Creating directory: {dir_}")
        Path.mkdir(dir_, exist_ok=True, parents=True)
        if verbose:
            logger.success(f"Created directory: {dir_}")


def open_json_type(filepath: str | Path) -> Any:
    """Helper function to open JSON-type files. This includes JSON, JSONL, JSONNET, and
    YAML file types.

    Parameters
    ----------
    filepath
        Path to the file to be loaded.

    Returns
    -------
    Any
        Contains (key, value) pairs from the file specified by `filepath`. This can
        either be a dictionary or a list of dictionaries.

    Raises
    ------
    RuntimeError
        If an error occurs when loading a .jsonnet file.
    ValueError
        If an error occurs when loading a .json or YAML file.
    """

    filepath = Path(filepath)
    assert Path.is_file(filepath)
    file_ext = filepath.suffix
    assert Valid.is_valid_job_file_ext(file_ext=file_ext)
    if file_ext == ".json":
        with filepath.open("r", encoding="utf-8") as f:
            dict_ = json.load(f)
        return dict_
    with filepath.open("r", encoding="utf-8") as f:
        json_list = list(f)
    return [json.loads(json_str) for json_str in json_list]


def recurse_replace(new_str: str, orig_str: str, x: Any) -> Any:
    """Recursively replace all instances of `orig_str` in `x` with the value specified
    by `new_str`.

    Parameters
    ----------
    new_str
        The replacement string.
    orig_str
        The original string.
    x
        Either a string, list, or dictionary. This object will be recursively scanned
        in order to replace all instances of `orig_str` with `new_str`.

    Returns
    -------
    Any
        The final return of this function is the original passed in `x` with all
        instances of `orig_str` replaced with `new_str`.

    """

    if isinstance(x, str) and orig_str in x:
        return x.replace(orig_str, new_str)
    if isinstance(x, list):
        for i, item in enumerate(x):
            x[i] = recurse_replace(new_str, orig_str, x=item)
    elif isinstance(x, dict):
        for k, v in list(x.items()):
            k_ = recurse_replace(new_str, orig_str, x=k)
            x.pop(k)
            x[k_] = recurse_replace(new_str, orig_str, x=v)
    return x


def remove_json_markdown(*, text: str) -> str:
    """Remove JSON markdown from text.

    Parameters
    ----------
    text
        The text containing the JSON markdown.

    Returns
    -------
    str
        The text with the json markdown removed.
    """

    text = text.strip()
    text = re.sub(r"```(json)?\n", "", text).rstrip("```")
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    return text.strip()


def remove_punctuation(s: str, keep_punctuation: str | list[str] | None = None) -> str:
    """Remove all punctuation from a string.

    Parameters
    ----------
    s
        The string to remove punctuation from.
    keep_punctuation
        If not None, then a list of punctuation characters to keep.

    Returns
    -------
    str
        The string after removing all punctuation.
    """

    if keep_punctuation is not None and not isinstance(keep_punctuation, list):
        keep_punctuation = [keep_punctuation]
    keep_punctuation = keep_punctuation or []
    punc = "".join(x for x in string.punctuation if x not in keep_punctuation)
    return s.translate(s.maketrans("", "", punc))


def remove_whitespace(
    s: str, keep_newlines: bool = False, remove_all: bool = False
) -> str:
    """Remove all whitespace from beginning and end of string and extra whitespaces
    between words in the string.

    Parameters
    ----------
    s
        The string to remove whitespace from.
    keep_newlines
        Specifies whether to keep newlines in the returned string.
    remove_all
        Specifies whether to remove all space characters.

    Returns
    -------
    str
        The passed in string after cleaning for whitespace.
    """

    if remove_all:
        return "".join(s.replace(" ", "").strip().split())
    if not keep_newlines:
        return " ".join(s.strip().split())
    return re.sub(r"[^\S\n]+", "", s)


def telemetry_timer(
    *, metric_fn: Any, unit: str = "s"
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to measure the duration of an async function and record it with a
    telemetry metric.

    Parameters
    ----------
    metric_fn
        A metric instrument (e.g., Logfire/OpenTelemetry Histogram) with a
        `.record(value)` method.
    unit
        Time unit to record the duration in.

    Returns
    -------
    Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]
        A decorated coroutine function that records its execution time into the given
        metric.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        """Decorator function to wrap the async function.

        Parameters
        ----------
        func
            The async function to wrap.

        Returns
        -------
        Callable[P, Awaitable[T]]
            The wrapped function that records its execution time.
        """

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            """Wrapper function to measure execution time of the async function.

            Parameters
            ----------
            args
                Additional positional arguments to pass to the function.
            kwargs
                Additional keyword arguments to pass to the function.

            Returns
            -------
            T
                The result of the async function after measuring its execution time.
            """

            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if unit == "ms":
                elapsed *= 1000
            metric_fn.record(elapsed)
            return result

        return wrapper

    return decorator


def write_to_csv(
    *, fieldnames: list[str], fp: str | Path, rows: list[dict[str, Any]]
) -> None:
    """Write a list of dictionaries to a CSV file.

    Parameters
    ----------
    fieldnames
        The field names for the CSV file.
    fp
        Filepath to write the CSV file to.
    rows
        The rows of data to write to the CSV file. This should be a list of
        dictionaries where each dictionary represents a row.
    """

    fp = Path(fp)
    with fp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_to_json(
    fp: str | Path,
    json_info: dict[str, Any] | list[dict[str, Any]],
    encoding: str = "utf-8",
) -> None:
    """Write data either to .json or .jsonl file. The format is determined by the
    filepath extension.

    Parameters
    ----------
    fp
        Filepath to write the JSON file to.
    json_info
        JSON data to write out.
    encoding
        The encoding scheme for the JSON file.

    Raises
    ------
    ValueError
        If an incorrect suffix is specified for the filepath.
    """

    fp = Path(fp)
    suffix = fp.suffix
    if suffix == ".json":
        with fp.open("w", encoding=encoding) as f:
            json.dump(json_info, f)
    elif suffix == ".jsonl":
        with fp.open("w", encoding=encoding) as f:
            for dict_ in json_info:
                f.write(json.dumps(dict_) + "\n")
    else:
        raise ValueError(
            f"Invalid suffix for writing to JSON: {suffix}. "
            f"Valid suffixes are: '.json' and '.jsonl'"
        )
