import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlitedict import SqliteDict

from pylate import evaluation, indexes, models, retrieve

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class IndexType(Enum):
    """Supported index types."""

    PREBUILT = "prebuilt"
    LOCAL = "local"


@dataclass
class IndexConfig:
    """Configuration for a search index."""

    name: str
    type: IndexType
    path: str
    description: Optional[str] = None


class MCPyLate:
    """Main server class that manages Pyserini indexes and search operations."""

    def __init__(self, override: bool = False):
        self.logger = logging.getLogger(__name__)
        dataset_name = "nfcorpus"

        model_name = "lightonai/GTE-ModernColBERT-v1"

        override = override or not os.path.exists(
            f"indexes/{dataset_name}_{model_name.split('/')[-1]}"
        )
        self.model = models.ColBERT(
            model_name_or_path=model_name,
        )
        self.index = indexes.PLAID(
            override=override,
            index_name=f"{dataset_name}_{model_name.split('/')[-1]}",
        )

        self.retriever = retrieve.ColBERT(index=self.index)
        self.id_to_doc = SqliteDict(
            f"./indexes/{dataset_name}_{model_name.split('/')[-1]}/id_to_doc.sqlite",
            outer_stack=False,
        )
        if override:
            documents, _, _ = evaluation.load_beir(
                dataset_name=dataset_name,
                split="dev" if "msmarco" in dataset_name else "test",
            )

            for doc in documents:
                self.id_to_doc[doc["id"]] = doc["text"]
            self.id_to_doc.commit()  # Don't forget to commit to save changes!
            documents_embeddings = self.model.encode(
                sentences=[document["text"] for document in documents],
                batch_size=20,
                pool_factor=2,
                is_query=False,
                show_progress_bar=True,
            )
            self.index.add_documents(
                documents_ids=[document["id"] for document in documents],
                documents_embeddings=documents_embeddings,
            )

        self.logger.info("Created PyLate MCP Server")

    def get_document(
        self,
        docid: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve full document by document ID."""

        return {"docid": docid, "text": self.id_to_doc[docid]}

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform multi-vector search on specified index."""
        try:
            query_embeddings = self.model.encode(
                sentences=[query],
                is_query=True,
                show_progress_bar=True,
                batch_size=32,
            )
            scores = self.retriever.retrieve(queries_embeddings=query_embeddings, k=20)
            results = []
            for score in scores[0]:
                results.append(
                    {
                        "docid": score["id"],
                        "score": round(score["score"], 5),
                        "text": self.id_to_doc[score["id"]],
                    }
                )
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}")
