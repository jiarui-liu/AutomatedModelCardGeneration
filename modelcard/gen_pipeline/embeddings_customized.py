from typing import Any, Dict, List, Optional
from langchain.pydantic_v1 import BaseModel, Extra, Field
from langchain.schema.embeddings import Embeddings
from transformers import AutoModel

class TransformerEmbeddings(BaseModel, Embeddings):
    """Transformer embedding models.
    """

    client: Any  #: :meta private:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.client = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_folder,
            trust_remote_code=True,
            **self.model_kwargs
        )
        self.client = self.client.to("cuda")

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
