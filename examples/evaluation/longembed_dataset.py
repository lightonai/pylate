"""Evaluation script for the LongEmbed task using the MTEB library."""

from __future__ import annotations

import mteb

from pylate import evaluation, indexes, models, retrieve

if __name__ == "__main__":
    tasks = mteb.get_tasks(
        tasks=[
            "LEMBNarrativeQARetrieval",
            "LEMBNeedleRetrieval",
            "LEMBPasskeyRetrieval",
            "LEMBQMSumRetrieval",
            "LEMBSummScreenFDRetrieval",
            "LEMBWikimQARetrieval",
        ]
    )
    for task in tasks:
        task.load_data()
        model_name = "lightonai/GTE-ModernColBERT-v1"
        model = models.ColBERT(
            model_name_or_path=model_name,
            document_length=16384,
            trust_remote_code=True,
        )
        for eval_set in task.queries.keys():
            index = indexes.PLAID(
                override=True,
                nbits=4,
                index_name=f"{task.metadata.name}_{eval_set}_{model_name.split('/')[-1]}_4bits_ir",
            )

            retriever = retrieve.ColBERT(index=index)

            documents_embeddings = model.encode(
                sentences=list(task.corpus[eval_set].values()),
                batch_size=10,
                is_query=False,
                show_progress_bar=True,
            )

            index.add_documents(
                documents_ids=list(task.corpus[eval_set].keys()),
                documents_embeddings=documents_embeddings,
            )
            queries_embeddings = model.encode(
                sentences=list(task.queries[eval_set].values()),
                is_query=True,
                show_progress_bar=True,
                batch_size=32,
            )

            scores = retriever.retrieve(queries_embeddings=queries_embeddings)

            evaluation_scores = evaluation.evaluate(
                scores=scores,
                qrels=task.relevant_docs[eval_set],
                queries=list(task.queries[eval_set].keys()),
                metrics=[
                    "map",
                    "ndcg@1",
                    "ndcg@10",
                    "ndcg@100",
                    "recall@10",
                    "recall@100",
                ],
            )

            print(evaluation_scores)
