"""Parse a document with Docling, cut it into heading-aware chunks, search and rerank them."""
import re

MAX_TOK = 300  # Laya reads ~512 tokens: question + chunk must fit
EMB = "BAAI/bge-small-en-v1.5"
RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def parse(path):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    # ponytail: OCR off. The book has a real text layer (~1,900 chars/page, 0 of 40 sampled pages empty);
    # OCR on CPU is hours slow and only adds noise. Turn it on for a scanned PDF.
    opts = PdfPipelineOptions(do_ocr=False)
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return conv.convert(path).document


def load(path):
    from docling_core.types.doc import DoclingDocument
    return DoclingDocument.load_from_json(path)


def chapter_starts(pdf):
    """[(start_page, 'Chapter N. Title'), ..., (first_appendix_page, 'back')] from the PDF bookmarks.
    Docling's headings are a flat list of section titles, so chapters come from the bookmarks instead."""
    import pymupdf
    toc = pymupdf.open(pdf).get_toc(simple=True)
    out = [(p, t) for lvl, t, p in toc if lvl == 2 and re.match(r"Chapter \d+\. ", t)]
    back = next((p for lvl, t, p in toc if lvl == 1 and t.startswith("Appendix")), None)
    return sorted(out + ([(back, "back")] if back else []))


def chapter_of(page, starts):
    name = "front"
    for start, title in starts:
        if page >= start:
            name = title
    return name


def chunk(doc, max_tokens=MAX_TOK, tag=None, starts=None):
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    tok = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(EMB), max_tokens=max_tokens)
    ch = HybridChunker(tokenizer=tok)
    out, chapter = [], "front"
    for i, c in enumerate(ch.chunk(dl_doc=doc)):
        h = list(c.meta.headings or [])
        pages = [pv.page_no for it in c.meta.doc_items for pv in it.prov]
        if starts and pages:
            chapter = chapter_of(min(pages), starts)  # a chunk with no page info keeps the previous chapter
        out.append({"id": i, "text": c.text, "ctx": ch.contextualize(c),
                    "chapter": tag or (chapter if starts else (h[0] if h else "front")),
                    "section": " > ".join(h), "page": min(pages) if pages else None})
    return out


def merge_chunks(sources):
    """sources = [(label, chunks)] -> one list for one Index: ids are positions again, and each passage keeps its source label."""
    merged = []
    for label, chunks in sources:
        merged += [{**c, "id": len(merged) + i, "source": label} for i, c in enumerate(chunks)]
    return merged


class Index:
    def __init__(self, chunks, device=None):
        """device=None picks the GPU when there is one; the web app passes "cpu" to leave the 4 GB GPU to the Laya gates."""
        import faiss
        import numpy as np
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.chunks = chunks
        self.emb = SentenceTransformer(EMB, device=device)
        self.rr = CrossEncoder(RERANK, device=device)
        v = self.emb.encode([c["ctx"] for c in chunks], normalize_embeddings=True, batch_size=64)
        self.ix = faiss.IndexFlatIP(v.shape[1])
        self.ix.add(np.asarray(v, dtype="float32"))
        self._np = np

    def search(self, q, k=20):
        v = self.emb.encode([q], normalize_embeddings=True)
        _, ids = self.ix.search(self._np.asarray(v, dtype="float32"), k)
        return [self.chunks[i] for i in ids[0] if i >= 0]  # chunk id == list position

    def rerank(self, q, hits, k=5):
        scores = self.rr.predict([(q, h["text"]) for h in hits])
        return [h for _, h in sorted(zip(scores, hits), key=lambda t: -t[0])][:k]
