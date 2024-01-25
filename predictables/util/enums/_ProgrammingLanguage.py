from enum import Enum


class ProgrammingLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    OBJECTIVEC = "objectivec"
    SCALA = "scala"
    HASKELL = "haskell"
    CLOJURE = "clojure"
    ERLANG = "erlang"
    ELM = "elm"
    OCAML = "ocaml"
    FSHARP = "fsharp"
    LISP = "lisp"
    LUA = "lua"
    PASCAL = "pascal"
    PERL = "perl"
    R = "r"
    RACKET = "racket"
    SCHEME = "scheme"
    SHELL = "shell"
    SQL = "sql"
    VISUALBASIC = "visualbasic"
    WEBASSEMBLY = "webassembly"
    XML = "xml"
    YAML = "yaml"

    @staticmethod
    def from_string(string: str) -> "ProgrammingLanguage":
        string = string.lower().strip()

        if string in {
            member.value for member in ProgrammingLanguage.__members__.values()
        }:
            return ProgrammingLanguage[string]
        elif string in ["js", "javascript"]:
            return ProgrammingLanguage.JAVASCRIPT
        elif string in ["ts", "typescript"]:
            return ProgrammingLanguage.TYPESCRIPT
        elif string in ["c++", "cpp"]:
            return ProgrammingLanguage.CPP
        elif string in ["c#"]:
            return ProgrammingLanguage.CSHARP
        elif string in ["objective-c"]:
            return ProgrammingLanguage.OBJECTIVEC
        elif string in [
            "visual basic",
            "vb",
            "vb.net",
            "vbnet",
            "vb .net",
            "vba",
            "vb6",
            "vb 6",
        ]:
            return ProgrammingLanguage.WEBASSEMBLY
        elif string in ["xml", "html", "xhtml"]:
            return ProgrammingLanguage.XML
        elif string in ["yml", "yaml"]:
            return ProgrammingLanguage.YAML
        elif string in ["py", "python", "py3", "python3"]:
            return ProgrammingLanguage.PYTHON
        elif string in ["c"]:
            return ProgrammingLanguage.C
        elif string in ["java"]:
            return ProgrammingLanguage.JAVA
        else:
            raise ValueError(f"Programming language '{string}' not recognized.")
