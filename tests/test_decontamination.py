from gibc_llm.data import NgramContaminationFilter, normalize_for_ngrams


def test_canonical_normalization_makes_case_unicode_and_whitespace_deterministic() -> None:
    """Breaks if equivalent source text produces different contamination n-gram inputs."""
    first = normalize_for_ngrams("CAFÉ\t\n—  TWO")
    second = normalize_for_ngrams("café — two")

    assert first == second


def test_synthetic_clean_document_survives_thirteen_gram_screen() -> None:
    """Breaks if contamination filtering rejects unrelated documents indiscriminately."""
    filter_ = NgramContaminationFilter.from_texts(
        ["amber birch cedar drift ember fable granite harbor indigo juniper kettle lantern moon"], ngram_size=13
    )

    assert not filter_.screen("violet willow xylophone yellow zephyr atlas bridge cloud delta ember forest garden").rejected


def test_sqlite_contamination_index_matches_in_memory_screening(tmp_path) -> None:
    """Breaks if scalable hash-only indexing changes a contamination decision."""
    benchmark = "one two three four five six seven eight nine ten eleven twelve thirteen"
    memory = NgramContaminationFilter.from_texts([benchmark], ngram_size=13)
    sqlite = NgramContaminationFilter.from_sqlite_texts([benchmark], tmp_path / "index.sqlite", ngram_size=13)

    assert sqlite.screen("prefix " + benchmark + " suffix").rejected == memory.screen("prefix " + benchmark + " suffix").rejected
    assert not sqlite.screen("new clean independent text with no matching sequence at all").rejected
