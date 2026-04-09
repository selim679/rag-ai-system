def chunk_text(text, size=300, overlap=50):

    sentences = text.split(". ")

    chunks = []
    chunk = ""

    for s in sentences:
        if len(chunk) + len(s) < size:
            chunk += s + ". "
        else:
            chunks.append(chunk.strip())
            chunk = s + ". "

    if chunk:
        chunks.append(chunk.strip())

    return chunks
