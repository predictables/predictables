from __future__ import annotations

from pathlib import Path
from transformers import BertTokenizer
from transformers import BertModel, BertTokenizerFast

default_folder = "/sas/data/project/EG/ActShared/SmallBusiness/aw/bert/bert-cased"


class Bert:
    def __init__(
        self,
        bert_folder: str = default_folder,
        bert_model_name: str = "pytorch_model.bin",
        null_symbol: str = "[MASK]",
        padding_symbol: str = "[PAD]",
        use_fast_tokenizer: bool = True,
        truncation: bool | str = True,
        padding: bool | str = True,
        text: str | list | None = None,
    ):
        """Initialize the BERT model and tokenizer for generating embeddings.

        Parameters
        ----------
        bert_folder : str
            The folder containing the BERT model and tokenizer files. The default
            is "/sas/data/project/EG/ActShared/SmallBusiness/aw/bert/bert-cased".
        bert_model_name : str
            The name of the BERT model file. The default is "pytorch_model.bin".
        null_symbol : str
            The null symbol to use for masking. The default is "[MASK]".
        padding_symbol : str
            The padding symbol to use for padding. The default is "[PAD]".
        use_fast_tokenizer : bool
            Whether to use the fast tokenizer or not. The default is True. Use the
            slow tokenizer if you need to tokenize text longer than 512 tokens. The
            fast tokenizer is limited to 512 tokens.
        truncation : bool | str
            Whether to truncate text to 512 tokens. The default is True.
        padding : bool | str
            Whether to pad text to 512 tokens. The default is True.
        text : Union[str, list]
            A string or list of strings to add to the BERT model. The default is None.

        Returns
        -------
        None

        Attributes
        ----------
        folder : str
            The folder containing the BERT model and tokenizer files.
        model_name : str
            The name of the BERT model file.
        null_symbol : str
            The null symbol to use for masking.
        padding_symbol : str
            The padding symbol to use for padding.
        fast : bool
            Whether to use the fast tokenizer or not.
        tokenizer : BertTokenizerFast or BertTokenizer
            The BERT tokenizer, either the fast tokenizer or the slow tokenizer.
        model_path : str
            The path to the BERT model specifically. This is a concatenation of the
            folder and model name.
        model : BertModel
            The BERT model object.
        text : list
            A list of text strings, each of which is individually tokenized and
            embedded.
        """
        self.folder = bert_folder
        self.model_name = bert_model_name
        self.null_symbol = null_symbol
        self.padding_symbol = padding_symbol
        self.fast = use_fast_tokenizer
        if self.fast:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.folder, truncation=truncation, padding=padding
            )
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                self.folder, truncation=truncation, padding=padding
            )

        self.model_path = Path(self.folder) / self.model_name
        self.model = BertModel.from_pretrained(
            self.model_path, config=self.tokenizer.config
        )

        self.text = text

    def _pad(self, string: str) -> str:
        """Pad a string to the maximum length of 512 tokens.

        Parameters
        ----------
        string : str
            The string to pad.

        Returns
        -------
        str
            The padded string.
        """
        # string gets reshaped to max length of 512 tokens
        if len(string) > 512:
            # truncate string if longer than 512 tokens
            string = string[:512]
        elif len(string) < 512:
            # pad string if shorter than 512 tokens
            string = f"{string} {self.padding_symbol * (512 - len(string) + 1)}"
        return string

    def add(self, text: list | str) -> None:
        """Add text to the BERT model.

        Parameters
        ----------
        text : Union[list, str]
            A string or list of strings to add to the BERT model.
        """
        if isinstance(text, list):
            self.text = text if self.text is None else self.text
        elif isinstance(text, str):
            self.text = [text] if self.text is None else [self.text, text]
        else:
            raise ValueError(
                "`text` must be a string or a list of strings. Please try again."
            )

    def tokens(self, text: str) -> dict:
        """
        Tokenizes a string using the BERT tokenizer.

        Parameters
        ----------
        text : str
            The string to tokenize.

        Returns
        -------
        dict
            The tokenized string, along with the token type ids and attention mask.
        """
        # tokenize
        return self.tokenizer(text, return_tensors="pt")

    def embeddings(self, masked_string: str, pad: bool = True) -> dict:
        """Generate embeddings for a string using the BERT model.

        Parameters
        ----------
        masked_string : str
            The string to generate embeddings for.
        pad : bool
            Whether to pad the string to the maximum length of 512 tokens. The
            default is True.

        Returns
        -------
        dict
            The embeddings for the string.
        """
        # pad string to max length of 512 tokens if needed
        if pad:
            masked_string = self._pad(masked_string)

        # generate embeddings
        return self.model(**self.tokens(masked_string))

    def get_embeddings(self, masked_string: str, pad: bool = True) -> list:
        """Generate embeddings for a string using the BERT model.

        Parameters
        ----------
        masked_string : str
            The string to generate embeddings for.
        pad : bool
            Whether to pad the string to the maximum length of 512 tokens. The
            default is True.

        Returns
        -------
        list
            The embeddings for the string.
        """
        # pad string to max length of 512 tokens if needed
        if pad:
            masked_string = self._pad(masked_string)

        # generate embeddings
        outputs = self.embeddings(masked_string)

        # get embeddings
        return outputs["last_hidden_state"]
