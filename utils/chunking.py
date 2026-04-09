def chunk_text(text, size=300, overlap=50):
    chunks = []

    step = size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + size]
        chunks.append(chunk)

    return chunks
