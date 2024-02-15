import pytest

from predictables.util.enums._ProgrammingLanguage import (
    ProgrammingLanguage as Lang,
)


@pytest.fixture(
    params=[
        ("python", Lang.PYTHON),
        ("python3", Lang.PYTHON),
        ("py", Lang.PYTHON),
        ("py3", Lang.PYTHON),
        ("javascript", Lang.JAVASCRIPT),
        ("js", Lang.JAVASCRIPT),
        ("ecmascript", Lang.JAVASCRIPT),
        ("typescript", Lang.TYPESCRIPT),
        ("ts", Lang.TYPESCRIPT),
        ("c", Lang.C),
        ("cpp", Lang.CPP),
        ("c++", Lang.CPP),
        ("c plus plus", Lang.CPP),
        ("c plusplus", Lang.CPP),
        ("cplusplus", Lang.CPP),
        ("csharp", Lang.CSHARP),
        ("c#", Lang.CSHARP),
        ("c sharp", Lang.CSHARP),
        ("java", Lang.JAVA),
        ("go", Lang.GO),
        ("golang", Lang.GO),
        ("rust", Lang.RUST),
        ("rustlang", Lang.RUST),
        ("rust-lang", Lang.RUST),
        ("rs", Lang.RUST),
        ("ruby", Lang.RUBY),
        ("php", Lang.PHP),
        ("swift", Lang.SWIFT),
        ("kotlin", Lang.KOTLIN),
        ("objectivec", Lang.OBJECTIVEC),
        ("scala", Lang.SCALA),
        ("haskell", Lang.HASKELL),
        ("clojure", Lang.CLOJURE),
        ("erlang", Lang.ERLANG),
        ("elm", Lang.ELM),
        ("ocaml", Lang.OCAML),
        ("fsharp", Lang.FSHARP),
        ("f#", Lang.FSHARP),
        ("lisp", Lang.LISP),
        ("lua", Lang.LUA),
        ("pascal", Lang.PASCAL),
        ("perl", Lang.PERL),
        ("r", Lang.R),
        ("racket", Lang.RACKET),
        ("scheme", Lang.SCHEME),
        ("shell", Lang.SHELL),
        ("sql", Lang.SQL),
        ("visualbasic", Lang.VISUALBASIC),
        ("vb", Lang.VISUALBASIC),
        ("vb.net", Lang.VISUALBASIC),
        ("vbnet", Lang.VISUALBASIC),
        ("vb .net", Lang.VISUALBASIC),
        ("vba", Lang.VISUALBASIC),
        ("vbs", Lang.VISUALBASIC),
        ("vb6", Lang.VISUALBASIC),
        ("vb 6", Lang.VISUALBASIC),
        ("webassembly", Lang.WEBASSEMBLY),
        ("wasm", Lang.WEBASSEMBLY),
        ("xml", Lang.XML),
        ("yaml", Lang.YAML),
        ("js", Lang.JAVASCRIPT),
        ("ts", Lang.TYPESCRIPT),
        ("c++", Lang.CPP),
        ("c#", Lang.CSHARP),
        ("objective-c", Lang.OBJECTIVEC),
        ("visual basic", Lang.VISUALBASIC),
        ("vb", Lang.VISUALBASIC),
        ("vb.net", Lang.VISUALBASIC),
        ("vbnet", Lang.VISUALBASIC),
        ("vb .net", Lang.VISUALBASIC),
        ("vba", Lang.VISUALBASIC),
        ("vb6", Lang.VISUALBASIC),
        ("vb 6", Lang.VISUALBASIC),
        ("html", Lang.XML),
        ("xhtml", Lang.XML),
        ("yml", Lang.YAML),
        ("py", Lang.PYTHON),
        ("py3", Lang.PYTHON),
        ("python3", Lang.PYTHON),
    ]
)
def valid_input(request):
    return request.param


@pytest.fixture(
    params=[
        "not a language",
        "123",
        "",
        " ",
        None,
    ]
)
def invalid_input_returns_python(request):
    return request.param


def test_from_string_valid_input(valid_input):
    string, expected = valid_input
    assert (
        Lang.from_string(string) == expected
    ), f"Expected Lang.from_string('{string}') to be: {expected}, got {Lang.from_string(string)}."


def test_from_string_invalid_input(invalid_input_returns_python):
    assert (
        Lang.from_string(invalid_input_returns_python) == Lang.PYTHON
    ), f"Failed on {invalid_input_returns_python}. Expected Lang.from_string('{invalid_input_returns_python}') (or any other invalid string) to be python: {Lang.PYTHON}."
