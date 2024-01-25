from enum import Enum
from typing import Optional


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
    TYPESCRIPT = "typescript"
    VISUALBASIC = "visualbasic"
    WEBASSEMBLY = "webassembly"
    XML = "xml"
    YAML = "yaml"

    @staticmethod
    def from_string(string: Optional[str]) -> "ProgrammingLanguage":
        if string is None:
            UserWarning("Programming language is None. Defaulting to Python.")
            return ProgrammingLanguage.PYTHON

        string = string.lower().strip()

        if string in ["c"]:
            return ProgrammingLanguage.C
        elif string in ["c#", "c sharp", "csharp"]:
            return ProgrammingLanguage.CSHARP
        elif string in ["c++", "cpp", "c plus plus", "c plusplus", "cplusplus"]:
            return ProgrammingLanguage.CPP
        elif string in ["clojure"]:
            return ProgrammingLanguage.CLOJURE
        elif string in ["elm"]:
            return ProgrammingLanguage.ELM
        elif string in ["erlang"]:
            return ProgrammingLanguage.ERLANG
        elif string in ["f#", "fsharp", "f sharp"]:
            return ProgrammingLanguage.FSHARP
        elif string in ["go", "golang"]:
            return ProgrammingLanguage.GO
        elif string in ["haskell"]:
            return ProgrammingLanguage.HASKELL
        elif string in ["kotlin"]:
            return ProgrammingLanguage.KOTLIN
        elif string in ["java"]:
            return ProgrammingLanguage.JAVA
        elif string in ["js", "javascript", "ecmascript"]:
            return ProgrammingLanguage.JAVASCRIPT
        elif string in ["lisp"]:
            return ProgrammingLanguage.LISP
        elif string in ["lua"]:
            return ProgrammingLanguage.LUA
        elif string in ["objective-c", "objectivec", "objective c"]:
            return ProgrammingLanguage.OBJECTIVEC
        elif string in ["ocaml"]:
            return ProgrammingLanguage.OCAML
        elif string in ["pascal"]:
            return ProgrammingLanguage.PASCAL
        elif string in ["perl"]:
            return ProgrammingLanguage.PERL
        elif string in ["php"]:
            return ProgrammingLanguage.PHP
        elif string in ["py", "python", "py3", "python3"]:
            return ProgrammingLanguage.PYTHON
        elif string in ["r"]:
            return ProgrammingLanguage.R
        elif string in ["racket"]:
            return ProgrammingLanguage.RACKET

        elif string in ["ruby"]:
            return ProgrammingLanguage.RUBY
        elif string in ["rust", "rustlang", "rust-lang", "rs"]:
            return ProgrammingLanguage.RUST
        elif string in ["scheme"]:
            return ProgrammingLanguage.SCHEME
        elif string in ["shell"]:
            return ProgrammingLanguage.SHELL
        elif string in ["sql"]:
            return ProgrammingLanguage.SQL
        elif string in ["swift"]:
            return ProgrammingLanguage.SWIFT
        elif string in ["scala"]:
            return ProgrammingLanguage.SCALA
        elif string in ["ts", "typescript"]:
            return ProgrammingLanguage.TYPESCRIPT
        elif string in [
            "visual basic",
            "vb",
            "vb.net",
            "vbnet",
            "vb .net",
            "vba",
            "visualbasic",
            "vbs",
            "vb6",
            "vb 6",
        ]:
            return ProgrammingLanguage.VISUALBASIC
        elif string in ["wasm", "webassembly"]:
            return ProgrammingLanguage.WEBASSEMBLY
        elif string in ["xml", "html", "xhtml"]:
            return ProgrammingLanguage.XML
        elif string in ["yml", "yaml"]:
            return ProgrammingLanguage.YAML

        else:
            UserWarning(
                f"Programming language {string} not recognized. Defaulting to Python."
            )
            return ProgrammingLanguage.PYTHON
