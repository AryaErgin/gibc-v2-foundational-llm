from gibc_llm.data import benchmark_sources_for_index


def test_contamination_provenance_locks_all_public_multiple_choice_splits() -> None:
    """Breaks if contamination generation silently returns to evaluation-only benchmark splits."""
    sources = {source["task"]: source for source in benchmark_sources_for_index()}

    assert sources["hellaswag"]["splits"] == ["train", "validation", "test"]
    assert sources["arc_easy"]["splits"] == ["train", "validation", "test"]
    assert sources["piqa"]["splits"] == ["train", "dev", "test"]
    assert sources["winogrande"]["splits"] == ["train", "validation", "test"]
    assert sources["wikitext103"]["splits"] == ["validation", "test"]
    assert all(len(source["revision"]) == 40 for source in sources.values())
