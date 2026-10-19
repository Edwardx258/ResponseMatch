from responsematch.retrieval import HashingEmbedder, chunk_text


def test_chunk_text_respects_size_and_ids():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = chunk_text(text, source="guide.md", max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 + 400 for c in chunks)
    assert chunks[0].id == "guide#0"


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=64)
    a, b = e.embed(["first aid kit", "first aid kit"])
    assert a == b
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9


def test_search_finds_the_relevant_handbook(store):
    top = store.search("injured person needs first aid certification", k=1)[0]
    assert top.source == "first-aid.md"
