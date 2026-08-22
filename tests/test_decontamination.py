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
