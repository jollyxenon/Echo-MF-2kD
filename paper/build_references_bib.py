#!/usr/bin/env python3
"""从 paper2-ncs-zh.md 的参考文献章节解析 APA 7th 文献条目，生成纯 ASCII 的 references.bib。"""

import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZH_MD = HERE / "paper2-ncs-zh.md"
OUT_BIB = HERE / "references.bib"

PROTECTED_TERMS = [
    "QAOA", "Ising", "MWIS", "GNN", "GNNs", "MLP", "MLPs", "TQA", "NISQ", "VQE",
    "QSP", "QSVT", "CPMG", "Daubechies", "Pontryagin", "Pontryagin's", "Hahn", "Herrmann", "Bloch",
    "L-BFGS-B", "CVaR", "PI-GNN", "Erdős", "Trotter", "Trotterized", "Boolean",
    "Max-Cut", "MaxCut", "Maximum", "Cut", "Max-SAT", "Bohmian", "Hamiltonian", "Pauli", "Pauli-X",
    "Rydberg", "Fourier", "Beta", "NP-hard", "NP-complete", "NP", "SciPy", "PyTorch",
    "IEEE", "ACM", "SIAM", "APS", "Nature", "Science", "PMLR", "PRX",
    "GPU", "GPUs", "QPU", "QPUs", "OpenMP", "CUDA", "Echo-QAOA", "QSeer",
    "Goemans-Williamson's", "Goemans-Williamson"
]

LATEX_ACCENT_MAP = {
    "ä": '{\\"a}',
    "ö": '{\\"o}',
    "ü": '{\\"u}',
    "Ä": '{\\"A}',
    "Ö": '{\\"O}',
    "Ü": '{\\"U}',
    "é": "{\\'e}",
    "á": "{\\'a}",
    "í": "{\\'\\i}",
    "ó": "{\\'o}",
    "ú": "{\\'u}",
    "Á": "{\\'A}",
    "É": "{\\'E}",
    "ć": "{\\'c}",
    "č": "{\\v{c}}",
    "ž": "{\\v{z}}",
    "š": "{\\v{s}}",
    "ő": "{\\H{o}}",
    "ű": "{\\H{u}}",
    "ñ": "{\\~n}",
    "ß": "{\\ss}",
}


def to_latex_ascii(text: str) -> str:
    """将含重音字符的字符串转换为纯 ASCII LaTeX 转义序列。"""
    for ch, repl in LATEX_ACCENT_MAP.items():
        text = text.replace(ch, repl)
    return text


def strip_accents_ascii(text: str) -> str:
    """将重音字符规范化为无重音纯 ASCII 字母（用于生成 BibTeX key）。"""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def protect_title(title: str) -> str:
    """保护标题中的专业缩略词与人名首字母大写。"""
    words = title.split()
    protected_words = []
    for w in words:
        match_punct = re.search(r"([,\\.:;\\?\\!]+)$", w)
        punct = match_punct.group(1) if match_punct else ""
        clean_w = w[:-len(punct)] if punct else w

        matched = False
        for pt in PROTECTED_TERMS:
            if clean_w.lower() == pt.lower():
                protected_words.append(f"{{{to_latex_ascii(pt)}}}" + punct)
                matched = True
                break
        if not matched:
            if clean_w.isupper() and len(clean_w) >= 2:
                protected_words.append(f"{{{to_latex_ascii(clean_w)}}}" + punct)
            else:
                protected_words.append(to_latex_ascii(w))
    return " ".join(protected_words)


def parse_apa_entry(entry: str):
    """解析单条 APA 7th 参考文献段落（支持 *...* 与 _..._ 斜体）。"""
    m = re.match(
        r"^(.+?)\s*\(([12][09]\d\d[a-z]?)\)\.\s*(.+?[\.\?])\s+(?=(?:In\s*:?\s*[*_]|[*_]))(.+)$",
        entry.strip(),
        re.DOTALL,
    )
    if not m:
        return None
    raw_authors, year, title, rest = m.groups()
    title = title.strip()
    if title.endswith("."):
        title = title[:-1].strip()

    raw_authors = re.sub(r"\.\.\.\s*", "", raw_authors)
    parts = raw_authors.replace("&", ",").split(",")
    names = []
    i = 0
    while i < len(parts):
        p = parts[i].strip()
        if not p:
            i += 1
            continue
        if i + 1 < len(parts):
            p2 = parts[i + 1].strip()
            if re.match(r"^[A-Z](\.\s*[A-Z])*\.?$", p2) or len(p2) <= 4:
                names.append(f"{p}, {p2}")
                i += 2
                continue
        names.append(p)
        i += 1

    first_author = strip_accents_ascii(names[0].split(",")[0].strip()).lower()
    first_author = re.sub(r"[^a-z]", "", first_author)

    ascii_title = strip_accents_ascii(title)
    title_words = [
        w.lower()
        for w in re.findall(r"[A-Za-z0-9]+", ascii_title)
        if w.lower() not in ("a", "an", "the", "on", "for", "in", "of", "to", "with", "from", "and")
    ]
    first_word = title_words[0] if title_words else "ref"
    key = f"{first_author}{year}{first_word}"

    entry_type = "article"
    doi = ""
    doi_match = re.search(r"https?://doi\.org/([^\s]+)", rest)
    if doi_match:
        doi = doi_match.group(1).rstrip(".")

    arxiv_match = re.search(r"arXiv\.(\d{4}\.\d{4,5})", rest)
    eprint = arxiv_match.group(1) if arxiv_match else ""

    journal = ""
    booktitle = ""
    volume = ""
    number = ""
    pages = ""
    publisher = ""

    if rest.startswith(("In *", "In: *", "In _", "In: _")):
        entry_type = "inproceedings"
        bt_match = re.search(r"In:?\s*[*_]([^*_]+)[*_]", rest)
        if bt_match:
            booktitle = bt_match.group(1).strip()
    else:
        if "*arXiv*" in rest or "_arXiv_" in rest or "arXiv preprint" in rest:
            journal = f"arXiv preprint arXiv:{eprint}" if eprint else "arXiv preprint"
        elif "Research Square preprint" in rest:
            journal = f"Research Square preprint https://doi.org/{doi}" if doi else "Research Square preprint"
        else:
            j_match = re.search(r"[*_]([^*_]+)[*_]", rest)
            if j_match:
                journal = j_match.group(1).strip()

    vp_match = re.search(r"[*_],\s*[*_]?(\d+)[*_]?(?:\(([0-9–—\-]+)\))?,\s*([0-9–—\-]+|[A-Z]?[0-9]+)\.", rest)
    if vp_match:
        volume = vp_match.group(1)
        number = (vp_match.group(2) or "").replace("–", "--").replace("—", "--")
        pages = vp_match.group(3).replace("–", "--").replace("—", "--")

    pp_match = re.search(r"\bpp\.\s*([0-9–—\-]+)\)", rest)
    if pp_match:
        pages = pp_match.group(1).replace("–", "--").replace("—", "--")

    vol_match = re.search(r"\(Vol\.\s*(\d+)", rest)
    if vol_match:
        volume = vol_match.group(1)

    pub_match = re.search(r"\)\.\s*([A-Za-z\s]+)\.\s*https?://", rest)
    if pub_match:
        publisher = pub_match.group(1).strip()

    return {
        "key": key,
        "type": entry_type,
        "author": to_latex_ascii(" and ".join(names)),
        "title": protect_title(title),
        "year": year,
        "journal": to_latex_ascii(journal),
        "booktitle": to_latex_ascii(booktitle),
        "volume": volume,
        "number": number,
        "pages": pages,
        "publisher": to_latex_ascii(publisher),
        "doi": doi,
        "eprint": eprint,
    }


def format_bibtex(e: dict) -> str:
    """将解析后的文献字典格式化为 BibTeX 块。"""
    lines = [f"@{e['type']}{{{e['key']},"]
    lines.append(f"  author    = {{{e['author']}}},")
    lines.append(f"  title     = {{{e['title']}}},")
    lines.append(f"  year      = {{{e['year']}}},")
    if e["type"] == "article":
        if e["journal"]:
            lines.append(f"  journal   = {{{e['journal']}}},")
        if e["volume"]:
            lines.append(f"  volume    = {{{e['volume']}}},")
        if e["number"]:
            lines.append(f"  number    = {{{e['number']}}},")
    elif e["type"] == "inproceedings":
        if e["booktitle"]:
            lines.append(f"  booktitle = {{{e['booktitle']}}},")
        if e["volume"]:
            lines.append(f"  volume    = {{{e['volume']}}},")
        if e["publisher"]:
            lines.append(f"  publisher = {{{e['publisher']}}},")
    if e["pages"]:
        lines.append(f"  pages     = {{{e['pages']}}},")
    if e["doi"]:
        lines.append(f"  doi       = {{{e['doi']}}},")
    if e["eprint"]:
        lines.append(f"  eprint    = {{{e['eprint']}}},")
        lines.append("  archiveprefix = {arXiv},")
    lines.append("}\n")
    return "\n".join(lines)


def main():
    """提取参考文献并写入 references.bib。"""
    text = ZH_MD.read_text(encoding="utf-8")
    pos_ref = text.find("## 参考文献")
    pos_app = text.find("## 附录", pos_ref)
    ref_text = text[pos_ref:pos_app] if pos_app != -1 else text[pos_ref:]
    paras = [p.strip() for p in ref_text.split("\n\n") if p.strip()][1:]

    entries = []
    seen_keys = set()
    for p in paras:
        if p.startswith("---") or p.startswith("#"):
            continue
        parsed = parse_apa_entry(p)
        if not parsed:
            raise ValueError(f"Failed to parse: {p}")
        key = parsed["key"]
        orig_key = key
        counter = 1
        while key in seen_keys:
            key = f"{orig_key}_{counter}"
            counter += 1
        parsed["key"] = key
        seen_keys.add(key)
        entries.append(parsed)

    bib_content = (
        f"% Auto-generated verified BibTeX library for Paper 2\n"
        f"% Total entries: {len(entries)}\n\n"
        + "\n".join(format_bibtex(e) for e in entries)
        + "\n"
    )
    non_ascii = [c for c in bib_content if ord(c) >= 128]
    assert not non_ascii, f"Found non-ASCII chars in BibTeX: {non_ascii}"

    OUT_BIB.write_text(bib_content, encoding="utf-8")
    print(f"Saved {len(entries)} pure-ASCII BibTeX entries to {OUT_BIB}")


if __name__ == "__main__":
    main()
