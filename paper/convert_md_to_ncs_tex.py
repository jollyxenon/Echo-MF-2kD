#!/usr/bin/env python3
"""
Deterministically convert paper/paper2/paper2-ncs-en.md into the publication-grade
Nature Computational Science (Springer Nature / Nature Portfolio) LaTeX manuscript
paper/paper2/main.tex.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EN_MD = HERE / "paper2-ncs-en.md"
BIB_FILE = HERE / "references.bib"
OUT_TEX = HERE / "main.tex"

LATEX_TO_UTF8 = {
    '{\\"a}': "ä",
    '{\\"o}': "ö",
    '{\\"u}': "ü",
    '{\\"A}': "Ä",
    '{\\"O}': "Ö",
    '{\\"U}': "Ü",
    "{\\'e}": "é",
    "{\\'a}": "á",
    "{\\'\\i}": "í",
    "{\\'o}": "ó",
    "{\\'u}": "ú",
    "{\\'A}": "Á",
    "{\\'E}": "É",
    "{\\'c}": "ć",
    "{\\v{c}}": "č",
    "{\\v{z}}": "ž",
    "{\\v{s}}": "š",
    "{\\H{o}}": "ő",
    "{\\H{u}}": "ű",
    "{\\~n}": "ñ",
    "{\\ss}": "ß",
}


def build_bib_map(bib_text: str) -> dict[tuple[str, str], str]:
    """Build (author_lastname_lower, year) -> bib_key mapping from references.bib."""
    bib_map = {}
    for m in re.finditer(
        r"@\w+\{([^,]+),\s*author\s*=\s*\{(.+?)\},\s*title.*?year\s*=\s*\{(\d{4}[a-z]?)\}",
        bib_text,
        re.DOTALL,
    ):
        key, authors, year = m.groups()
        first_last = authors.split(",")[0].strip()
        for k_tex, v_utf in LATEX_TO_UTF8.items():
            first_last = first_last.replace(k_tex, v_utf)
        first_last = first_last.lower()
        bib_map[(first_last, year)] = key
        ascii_last = (
            first_last.replace("é", "e")
            .replace("á", "a")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ö", "o")
            .replace("ü", "u")
            .replace("ä", "a")
            .replace("ß", "ss")
            .replace("ő", "o")
            .replace("č", "c")
            .replace("ć", "c")
            .replace("ž", "z")
            .replace("ñ", "n")
        )
        bib_map[(ascii_last, year)] = key
    return bib_map


def convert() -> None:
    """Convert paper2-ncs-en.md to Nature Computational Science main.tex."""
    bib_text = BIB_FILE.read_text(encoding="utf-8")
    bib_map = build_bib_map(bib_text)
    md_text = EN_MD.read_text(encoding="utf-8")

    # 1. Split off ## Extended Data, ## Code and Data Availability, ## References
    part_main_and_methods, rem_ext = md_text.split("\n## Extended Data\n", 1)
    part_ext, _ = rem_ext.split("\n## Code and Data Availability\n", 1)

    full_md = (
        part_main_and_methods.rstrip()
        + "\n\n@@END_MATTER_AND_BIB@@\n\n## Extended Data\n"
        + part_ext.rstrip()
        + "\n"
    )

    # 2. Replace Mermaid Figure 1 block and caption
    fig1_pattern = re.compile(
        r"```mermaid\n.*?\n```\s*\n+\*\*Figure 1\s*\|\s*(.+?)\*\*\s*(.+?)(?=\n\n)",
        re.DOTALL,
    )

    def repl_fig1(m: re.Match) -> str:
        bold_title, rest_cap = m.groups()
        return (
            "@@FIGURE_START:1:fig1_echo_mf_2kd_overview@@\n"
            f"\\textbf{{{bold_title.strip()}}} {rest_cap.strip()}\n"
            "@@FIGURE_END@@"
        )

    full_md = fig1_pattern.sub(repl_fig1, full_md)

    # 3. Replace HTML <img> + **Figure X | ...** for Figures 2 and 3
    fig_img_pattern = re.compile(
        r'<img\s+src="figures/([^"]+)\.png"[^>]*>\s*\n+\*\*Figure\s+(\d+)\s*\|\s*(.+?)\*\*\s*(.+?)(?=\n\n)',
        re.DOTALL,
    )

    def repl_fig(m: re.Match) -> str:
        stem, fnum, bold_title, rest_cap = m.groups()
        return (
            f"@@FIGURE_START:{fnum}:{stem}@@\n"
            f"\\textbf{{{bold_title.strip()}}} {rest_cap.strip()}\n"
            "@@FIGURE_END@@"
        )

    full_md = fig_img_pattern.sub(repl_fig, full_md)

    # 4. Protect math blocks $$ ... $$ and inline math $ ... $
    math_store: list[str] = []

    def save_math(tex_str: str) -> str:
        idx = len(math_store)
        math_store.append(tex_str)
        return f"@@MATH{idx}@@"

    def repl_display_math(m: re.Match) -> str:
        inner = m.group(1).strip()
        return save_math(f"\\[\n{inner}\n\\]")

    full_md = re.sub(r"\$\$(.+?)\$\$", repl_display_math, full_md, flags=re.DOTALL)

    def repl_inline_math(m: re.Match) -> str:
        inner = m.group(1).replace("\\text{--}+", "\\mbox{--}{+}").replace("\\text{--}", "\\mbox{--}")
        return save_math(f"${inner}$")

    full_md = re.sub(r"(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)", repl_inline_math, full_md)

    # 5. Convert Citations (Parenthetical & Narrative)
    full_md = full_md.replace(
        "(`MWIS`, `Dense Ising`, `Sparse Ising`, `Grid Ising`, and `Pinned MaxCut`; Yao, 2026)",
        "(`MWIS`, `Dense Ising`, `Sparse Ising`, `Grid Ising`, and `Pinned MaxCut`\\citep{yao2026echo})",
    )

    cite_item_re = re.compile(
        r"^([A-Z][a-zA-Zéöüäß\-\x27]+)(?:\s+(?:et al\.|(?:&|and)\s+[A-Z][a-zA-Zéöüäß\-\x27]+))?,\s*((?:19|20)\d{2}[a-z]?)$"
    )

    def repl_paren_cite(m: re.Match) -> str:
        content = m.group(1)
        parts = [p.strip() for p in content.split(";")]
        keys = []
        for p in parts:
            cm = cite_item_re.match(p)
            if not cm:
                return m.group(0)
            last, yr = cm.group(1).lower(), cm.group(2)
            key = bib_map.get((last, yr))
            if not key:
                raise ValueError(f"Unmapped citation: {p}")
            keys.append(key)
        return f"\\citep{{{','.join(keys)}}}"

    full_md = re.sub(r"\(([^()]*?\b(?:19|20)\d{2}[a-z]?[^()]*?)\)", repl_paren_cite, full_md)

    narr_re = re.compile(
        r"\b([A-Z][a-zA-Zéöüäß\-\x27]+)(?:\s+et al\.|\s+(?:and|&)\s+[A-Z][a-zA-Zéöüäß\-\x27]+)?\s*\(((?:19|20)\d{2}[a-z]?)\)"
    )

    def repl_narr_cite(m: re.Match) -> str:
        raw_last, yr = m.group(1), m.group(2)
        last = raw_last.lower()
        key = bib_map.get((last, yr))
        if not key:
            raise ValueError(f"Unmapped narrative citation: {m.group(0)}")
        return f"{raw_last}~\\citep{{{key}}}"

    full_md = narr_re.sub(repl_narr_cite, full_md)

    # 6. Convert Markdown links [text](url) -> \url{url}
    full_md = re.sub(r"\[https?://[^\]]+\]\((https?://[^)]+)\)", r"\\url{\1}", full_md)

    # 7. Convert inline code `...` -> \texttt{...}
    code_store: list[str] = []

    def repl_inline_code(m: re.Match) -> str:
        c = m.group(1)
        c = (
            c.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_\\allowbreak{}")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
            .replace("*", "{*}")
            .replace(", ", ",\\allowbreak\\ ")
        )
        idx = len(code_store)
        code_store.append(f"\\texttt{{{c}}}")
        return f"@@CODE{idx}@@"

    full_md = re.sub(r"`([^`\n]+)`", repl_inline_code, full_md)

    # 8. Extract and convert Tables (**Table X | ...** + | ... |)
    table_pattern = re.compile(
        r"(?:###\s+Extended Data Table\s+\d+:[^\n]+\n+)?\*\*Table\s+([0-9A-Za-z\.]+)\s*\|\s*(.+?)\*\*\s*(.+?)\n\n((?:\|[^\n]+\n)+)",
        re.DOTALL,
    )

    table_store: list[str] = []

    def format_inline_md(s: str) -> str:
        s = s.replace("<br>", " ").replace("\\*", "@@AST@@")
        s = s.replace("%", "\\%").replace("&", "\\&")
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
        s = re.sub(r"_([^_]+)_", r"\\emph{\1}", s)
        s = s.replace("@@AST@@", "*")
        s = s.replace("—", "---").replace("–", "--").replace("→", "$\\rightarrow$")
        s = (
            s.replace("é", "\\'e")
            .replace("ő", "\\H{o}")
            .replace("ü", '\\"u')
            .replace("ä", '\\"a')
            .replace("č", "\\v{c}")
        )
        return s

    def format_heading_md(s: str) -> str:
        res = format_inline_md(s)

        def repl_hmath(m: re.Match) -> str:
            midx = int(m.group(1))
            mtex = math_store[midx]
            plain = mtex.strip("$").replace("\\text{--}", "--").replace("\\", "")
            return f"\\texorpdfstring{{{mtex}}}{{{plain}}}"

        res = re.sub(r"@@MATH(\d+)@@", repl_hmath, res)
        return res

    def repl_table(m: re.Match) -> str:
        t_id, t_bold, t_rest, t_body = m.groups()
        t_cap_tex = f"\\textbf{{{format_inline_md(t_bold.strip())}}} {format_inline_md(t_rest.strip())}"
        lines = [l.strip() for l in t_body.strip().split("\n") if l.strip()]
        header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
        align_cells = [c.strip() for c in lines[1].strip("|").split("|")]
        data_rows = [
            [c.strip() for c in r.strip("|").split("|")] for r in lines[2:]
        ]

        col_specs = []
        for a in align_cells:
            if a.startswith(":") and a.endswith(":"):
                col_specs.append("c")
            elif a.endswith(":"):
                col_specs.append("r")
            else:
                col_specs.append("l")

        label_id = t_id.lower().replace(".", "_")
        col_spec_str = "".join(col_specs)

        def format_header_cell(raw_c: str, col_align: str) -> str:
            # Split long header cells into two balanced lines so tables render larger
            words = raw_c.split(" ")
            if len(raw_c) <= 20 or len(words) < 2:
                return f"\\textbf{{{format_inline_md(raw_c)}}}"
            # Find split point near middle outside of math/code tokens
            mid = len(raw_c) // 2
            best_idx = None
            best_dist = 10**9
            pos = 0
            for idx in range(1, len(words)):
                pos += len(words[idx - 1]) + 1
                dist = abs(pos - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            line1 = format_inline_md(" ".join(words[:best_idx]))
            line2 = format_inline_md(" ".join(words[best_idx:]))
            align_char = "l" if col_align == "l" else "c"
            return f"\\shortstack[{align_char}]{{\\textbf{{{line1}}} \\\\ \\textbf{{{line2}}}}}"

        tex_lines = [
            "\\begin{table}[!htbp]",
            f"\\caption{{{t_cap_tex}}}",
            f"\\label{{tab:{label_id}}}",
            "\\vspace{2pt}",
            "\\centering",
            "\\renewcommand{\\arraystretch}{1.15}",
            "\\setlength{\\tabcolsep}{3.5pt}",
            "\\resizebox{\\textwidth}{!}{%",
            f"\\begin{{tabular}}{{{col_spec_str}}}",
            "\\toprule",
        ]
        h_tex = " & ".join(
            format_header_cell(c, col_specs[idx]) for idx, c in enumerate(header_cells)
        ) + " \\\\"
        tex_lines.append(h_tex)
        tex_lines.append("\\midrule")
        for row in data_rows:
            r_tex = " & ".join(format_inline_md(c) for c in row) + " \\\\"
            tex_lines.append(r_tex)
        tex_lines.append("\\bottomrule")
        tex_lines.append("\\end{tabular}%")
        tex_lines.append("}")
        tex_lines.append("\\end{table}")

        idx = len(table_store)
        table_store.append("\n".join(tex_lines))
        return f"@@TABLE{idx}@@"

    full_md = table_pattern.sub(repl_table, full_md)

    # 9. Process Line-by-Line Structure
    lines = full_md.split("\n")
    out_lines: list[str] = []
    in_abstract = False
    list_stack: list[tuple[str, int]] = []

    def close_lists_to(target_indent: int) -> None:
        while list_stack and list_stack[-1][1] >= target_indent:
            env, _ = list_stack.pop()
            out_lines.append(f"\\end{{{env}}}")

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        s = raw_line.strip()

        if s.startswith("# "):
            i += 1
            continue

        if s == "---":
            close_lists_to(0)
            i += 1
            continue

        if s == "## Abstract":
            close_lists_to(0)
            out_lines.append("\\begin{ncsabstract}")
            in_abstract = True
            i += 1
            continue

        if s.startswith("## "):
            close_lists_to(0)
            if in_abstract:
                out_lines.append("\\end{ncsabstract}\n")
                in_abstract = False

            sec_title = s[3:].strip()
            if sec_title == "Extended Data":
                out_lines.append("\\clearpage")
                out_lines.append("\\section*{Extended Data}")
                out_lines.append("\\renewcommand{\\tablename}{Extended Data Table}")
                out_lines.append("\\setcounter{table}{0}")
            else:
                out_lines.append(f"\\section*{{{format_heading_md(sec_title)}}}")
            i += 1
            continue

        if s.startswith("### "):
            close_lists_to(0)
            sub_title = s[4:].strip()
            out_lines.append(f"\\subsection*{{{format_heading_md(sub_title)}}}")
            i += 1
            continue

        if s == "@@END_MATTER_AND_BIB@@":
            close_lists_to(0)
            out_lines.append(
                "\\section*{Data Availability}\n"
                "The five-family Ising benchmark datasets (\\texttt{data/benchmark}), the Maximum Weight Independent Set learning and cross-scale benchmark datasets (\\texttt{data/mwis}, spanning $n \\in \\{6, 8, 10, 12, 14\\}$ with 5,120 test instances), and all source data CSV files underlying Figures 1--3 and Extended Data Tables 1--2 are publicly available at \\url{https://github.com/jollyxenon/Echo-MF-2kD}.\n\n"
                "\\section*{Code Availability}\n"
                "All core source code---including the \\texttt{Echo-MF-2kD} and \\texttt{Echo-Flat-2kD} $\\mathrm{SU}(2)$-adjoint manifold refinement engine (\\texttt{src/echo/multiscalar.py}), Daubechies maximally flat filter synthesis (\\texttt{src/echo/core.py}), and deterministic evaluation and plotting scripts---is open-sourced under the MIT License at \\url{https://github.com/jollyxenon/Echo-MF-2kD}.\n\n"
                "\\section*{Acknowledgments}\n"
                "The author thanks colleagues at ShanghaiTech University for helpful discussions on variational quantum algorithms and statistical mechanics of disordered spin systems.\n\n"
                "\\section*{Author Contributions}\n"
                "J.Y. conceived the mathematical theory, designed the \\texttt{Echo-MF-2kD} algorithm, implemented the software and benchmark suites, performed the numerical experiments, analyzed the data, and wrote the manuscript.\n\n"
                "\\section*{Competing Interests}\n"
                "The author declares no competing interests.\n\n"
                "\\bibliographystyle{unsrtnat}\n"
                "\\bibliography{references}"
            )
            i += 1
            continue

        if s.startswith("@@FIGURE_START:"):
            close_lists_to(0)
            _, fnum, stem = s.strip("@").split(":")
            cap_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "@@FIGURE_END@@":
                cap_lines.append(lines[i].strip())
                i += 1
            cap_tex = format_inline_md(" ".join(cap_lines))
            fig_w = "0.96\\textwidth" if fnum == "1" else "0.92\\textwidth"
            out_lines.append(
                "\\begin{figure}[!htbp]\n"
                "\\centering\n"
                f"\\includegraphics[width={fig_w}]{{figures/{stem}.pdf}}\n"
                "\\vspace{-2pt}\n"
                f"\\caption{{{cap_tex}}}\n"
                f"\\label{{fig:{fnum}}}\n"
                "\\end{figure}"
            )
            i += 1
            continue

        if re.match(r"^@@TABLE\d+@@$", s):
            close_lists_to(0)
            out_lines.append(s)
            i += 1
            continue

        m_enum = re.match(r"^(\s*)\d+\.\s+(.+)$", raw_line)
        m_item = re.match(r"^(\s*)-\s+(.+)$", raw_line)
        if m_enum or m_item:
            indent = len((m_enum or m_item).group(1))
            content = (m_enum or m_item).group(2)
            env_type = "enumerate" if m_enum else "itemize"

            while list_stack and list_stack[-1][1] > indent:
                env, _ = list_stack.pop()
                out_lines.append(f"\\end{{{env}}}")
            if not list_stack or list_stack[-1][1] < indent:
                list_stack.append((env_type, indent))
                out_lines.append(
                    f"\\begin{{{env_type}}}\\setlength{{\\itemsep}}{{1pt}}\\setlength{{\\parskip}}{{1pt}}\\setlength{{\\topsep}}{{2pt}}"
                )
            elif list_stack[-1][0] != env_type:
                env, _ = list_stack.pop()
                out_lines.append(f"\\end{{{env}}}")
                list_stack.append((env_type, indent))
                out_lines.append(
                    f"\\begin{{{env_type}}}\\setlength{{\\itemsep}}{{1pt}}\\setlength{{\\parskip}}{{1pt}}\\setlength{{\\topsep}}{{2pt}}"
                )

            out_lines.append(f"\\item {format_inline_md(content)}")
            i += 1
            continue

        if not s:
            out_lines.append("")
            i += 1
            continue

        if list_stack and raw_line.startswith("   "):
            out_lines.append(format_inline_md(s))
            i += 1
            continue

        if list_stack:
            close_lists_to(0)

        out_lines.append(format_inline_md(s))
        i += 1

    close_lists_to(0)
    body_tex = "\n".join(out_lines)

    # 10. Restore Tables, Code, and Math tokens
    for idx, tbl_tex in enumerate(table_store):
        body_tex = body_tex.replace(f"@@TABLE{idx}@@", tbl_tex)

    for idx, c_tex in enumerate(code_store):
        body_tex = body_tex.replace(f"@@CODE{idx}@@", c_tex)

    for idx, m_tex in enumerate(math_store):
        if m_tex.startswith("\\[\n"):
            body_tex = re.sub(
                rf"\n*@@MATH{idx}@@\n*",
                lambda _, s=m_tex: f"\n{s}\n",
                body_tex,
            )
        else:
            body_tex = body_tex.replace(f"@@MATH{idx}@@", m_tex)

    preamble = r"""%%%%%%%% NATURE COMPUTATIONAL SCIENCE LATEX MANUSCRIPT %%%%%%%%%%%%%%%%
%
% Deterministic conversion of paper/paper2/paper2-ncs-en.md
% Style: Nature Portfolio / Nature Computational Science (sn-ncs.sty)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass[10pt,a4paper]{article}

\usepackage{sn-ncs}

\ncstitle{Training-free mean-field adjoint manifold initialization for constrained variational quantum optimization}
\ncsauthor{Jianxuan Yao}
\ncsaffiliation{ShanghaiTech University, Shanghai, China}
\ncsemail{yaojx2025@shanghaitech.edu.cn}

\begin{document}

\makencstitle
"""

    full_tex = preamble + "\n" + body_tex + "\n\n\\end{document}\n"
    non_ascii = [c for c in full_tex if ord(c) >= 128]
    assert not non_ascii, f"Found non-ASCII chars in main.tex: {non_ascii}"
    OUT_TEX.write_text(full_tex, encoding="utf-8")
    print(f"Successfully wrote {len(full_tex.splitlines())} pure-ASCII lines to {OUT_TEX}")


if __name__ == "__main__":
    convert()
