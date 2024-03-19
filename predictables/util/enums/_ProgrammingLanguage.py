from __future__ import annotations

from enum import Enum


class ProgrammingLanguage(Enum):
    C = "c"
    CSHARP = "csharp"
    CPP = "cpp"
    CLOJURE = "clojure"
    ELM = "elm"
    ERLANG = "erlang"
    FSHARP = "fsharp"
    GO = "go"
    KOTLIN = "kotlin"
    HASKELL = "haskell"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    LISP = "lisp"
    LUA = "lua"
    OBJECTIVEC = "objectivec"
    OCAML = "ocaml"
    PASCAL = "pascal"
    PERL = "perl"
    PHP = "php"
    PYTHON = "python"
    R = "r"
    RACKET = "racket"
    SCHEME = "scheme"
    SHELL = "shell"
    SQL = "sql"
    RUBY = "ruby"
    RUST = "rust"
    SWIFT = "swift"
    SCALA = "scala"
    TOML = "toml"
    TYPESCRIPT = "typescript"
    VISUALBASIC = "visualbasic"
    WEBASSEMBLY = "webassembly"
    XML = "xml"
    YAML = "yaml"

    @staticmethod
    def from_string(string: str | None) -> "ProgrammingLanguage":
        if string is None:
            UserWarning("Programming language is None. Defaulting to Python.")
            return ProgrammingLanguage.PYTHON

        programming_language_map = {
            "c": ProgrammingLanguage.C,
            "c#": ProgrammingLanguage.CSHARP,
            "c sharp": ProgrammingLanguage.CSHARP,
            "csharp": ProgrammingLanguage.CSHARP,
            "c++": ProgrammingLanguage.CPP,
            "cpp": ProgrammingLanguage.CPP,
            "c plus plus": ProgrammingLanguage.CPP,
            "c plusplus": ProgrammingLanguage.CPP,
            "cplusplus": ProgrammingLanguage.CPP,
            "clojure": ProgrammingLanguage.CLOJURE,
            "elm": ProgrammingLanguage.ELM,
            "erlang": ProgrammingLanguage.ERLANG,
            "f#": ProgrammingLanguage.FSHARP,
            "fsharp": ProgrammingLanguage.FSHARP,
            "f sharp": ProgrammingLanguage.FSHARP,
            "go": ProgrammingLanguage.GO,
            "golang": ProgrammingLanguage.GO,
            "haskell": ProgrammingLanguage.HASKELL,
            "kotlin": ProgrammingLanguage.KOTLIN,
            "java": ProgrammingLanguage.JAVA,
            "js": ProgrammingLanguage.JAVASCRIPT,
            "javascript": ProgrammingLanguage.JAVASCRIPT,
            "ecmascript": ProgrammingLanguage.JAVASCRIPT,
            "lisp": ProgrammingLanguage.LISP,
            "lua": ProgrammingLanguage.LUA,
            "objective-c": ProgrammingLanguage.OBJECTIVEC,
            "objectivec": ProgrammingLanguage.OBJECTIVEC,
            "objective c": ProgrammingLanguage.OBJECTIVEC,
            "ocaml": ProgrammingLanguage.OCAML,
            "pascal": ProgrammingLanguage.PASCAL,
            "perl": ProgrammingLanguage.PERL,
            "php": ProgrammingLanguage.PHP,
            "py": ProgrammingLanguage.PYTHON,
            "python": ProgrammingLanguage.PYTHON,
            "py3": ProgrammingLanguage.PYTHON,
            "python3": ProgrammingLanguage.PYTHON,
            "r": ProgrammingLanguage.R,
            "racket": ProgrammingLanguage.RACKET,
            "ruby": ProgrammingLanguage.RUBY,
            "rust": ProgrammingLanguage.RUST,
            "rustlang": ProgrammingLanguage.RUST,
            "rust-lang": ProgrammingLanguage.RUST,
            "rs": ProgrammingLanguage.RUST,
            "scheme": ProgrammingLanguage.SCHEME,
            "shell": ProgrammingLanguage.SHELL,
            "sh": ProgrammingLanguage.SHELL,
            "sql": ProgrammingLanguage.SQL,
            "swift": ProgrammingLanguage.SWIFT,
            "scala": ProgrammingLanguage.SCALA,
            "toml": ProgrammingLanguage.TOML,
            "tml": ProgrammingLanguage.TOML,
            "ts": ProgrammingLanguage.TYPESCRIPT,
            "typescript": ProgrammingLanguage.TYPESCRIPT,
            "vb": ProgrammingLanguage.VISUALBASIC,
            "vb.net": ProgrammingLanguage.VISUALBASIC,
            "vbnet": ProgrammingLanguage.VISUALBASIC,
            "vb .net": ProgrammingLanguage.VISUALBASIC,
            "vba": ProgrammingLanguage.VISUALBASIC,
            "visualbasic": ProgrammingLanguage.VISUALBASIC,
            "visual basic": ProgrammingLanguage.VISUALBASIC,
            "vbs": ProgrammingLanguage.VISUALBASIC,
            "vb6": ProgrammingLanguage.VISUALBASIC,
            "vb 6": ProgrammingLanguage.VISUALBASIC,
            "wasm": ProgrammingLanguage.WEBASSEMBLY,
            "webassembly": ProgrammingLanguage.WEBASSEMBLY,
            "xml": ProgrammingLanguage.XML,
            "html": ProgrammingLanguage.XML,
            "xhtml": ProgrammingLanguage.XML,
            "yml": ProgrammingLanguage.YAML,
            "yaml": ProgrammingLanguage.YAML,
        }

        string = string.lower().strip()
        return programming_language_map.get(string, ProgrammingLanguage.PYTHON)
