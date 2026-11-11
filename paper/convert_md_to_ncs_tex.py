#!/usr/bin/env python3
"""
Deterministically convert paper/paper2-ncs-en.md into the publication-grade
single-column LaTeX manuscript paper/paper2_preprint.tex.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EN_MD = HERE / "paper2-ncs-en.md"
BIB_FILE = HERE / "references.bib"
OUT_TEX = HERE / "paper2_preprint.tex"

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
    """Convert paper2-ncs-en.md to Nature Computational Science paper2_preprint.tex."""
    bib_text = BIB_FILE.read_text(encoding="utf-8")
    bib_map = build_bib_map(bib_text)
    md_text = EN_MD.read_text(encoding="utf-8")

    # 1. Split off ## Extended Data, End Matter declarations, and ## References
    part_main_and_methods, rem_ext = md_text.split("\n## Extended Data\n", 1)
    part_ext, rem_decl = rem_ext.split("\n---\n\n## Data Availability\n", 1)
    part_decl, _ = rem_decl.split("\n## References\n", 1)

    full_md = (
        part_main_and_methods.rstrip()
        + "\n\n## Data Availability\n"
        + part_decl.rstrip()
        + "\n\n@@BIBLIOGRAPHY@@\n\n## Extended Data\n"
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
        "(`MWIS`, `Dense Ising`, `Sparse Ising`, `Grid Ising`, and `Pinned MaxCut`~\\citep{yao2026echo})",
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
            .replace("/", "/\\allowbreak{}")
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

    # 8. Extract and convert Tables (**Table X | ...** + single or a/b dual tabular blocks)
    table_pattern = re.compile(
        r"(?:###\s+Extended Data Table\s+\d+:[^\n]+\n+)?\*\*Table\s+([0-9A-Za-z\.]+)\s*\|\s*(.+?)\*\*\s*(.+?)\n\n"
        r"((?:\*\*[ab]\.\s+[^\n]+\*\*\n\n)?(?:\|[^\n]+\n)+(?:\n\*\*[ab]\.\s+[^\n]+\*\*\n\n(?:\|[^\n]+\n)+)?)",
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
        s = re.sub(r'"([^"]+)"', r"``\1''", s)
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

    def format_header_cell(raw_c: str, col_align: str) -> str:
        """Render header cell cleanly, using explicit <br> breaks when provided."""
        if "<br>" in raw_c:
            parts = [format_inline_md(p.strip()) for p in raw_c.split("<br>")]
            align_char = "l" if col_align == "l" else "c"
            inner = " \\\\ ".join(f"\\textbf{{{p}}}" for p in parts)
            return f"\\shortstack[{align_char}]{{{inner}}}"
        return f"\\textbf{{{format_inline_md(raw_c.strip())}}}"

    def build_single_tabular(t_id: str, md_grid: str, panel_header: str = "") -> list[str]:
        """Convert one Markdown pipe-table grid into booktabs LaTeX lines."""
        lines = [l.strip() for l in md_grid.strip().split("\n") if l.strip()]
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
        n_cols = len(col_specs)
        col_spec_str = "".join(col_specs)

        if panel_header != "":
            open_tab = f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{col_spec_str}@{{}}}}"
            close_tab = "\\end{tabular*}"
        else:
            open_tab = f"\\resizebox{{\\textwidth}}{{!}}{{%\n\\begin{{tabular}}{{@{{}}{col_spec_str}@{{}}}}"
            close_tab = "\\end{tabular}%\n}"

        out = [open_tab]
        if panel_header:
            m_ph = re.match(r"^([ab])\.\s+(.+)$", panel_header)
            if m_ph:
                p_let, p_txt = m_ph.groups()
                ph_tex = f"\\textbf{{{p_let}.}} \\textbf{{{format_inline_md(p_txt)}}}"
            else:
                ph_tex = f"\\textbf{{{format_inline_md(panel_header)}}}"
            out.append(f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{{ph_tex}}} \\\\[2pt]")
        out.append("\\toprule")
        h_tex = " & ".join(
            format_header_cell(c, col_specs[idx]) for idx, c in enumerate(header_cells)
        ) + " \\\\"
        out.append(h_tex)
        out.append("\\midrule")
        for r_idx, row in enumerate(data_rows):
            if t_id in ("1", "ED1") and r_idx > 0 and row[0]:
                out.append("\\midrule")
            elif t_id == "2" and r_idx in (4, 7):
                out.append("\\midrule")
            r_tex = " & ".join(format_inline_md(c) for c in row) + " \\\\"
            out.append(r_tex)
        out.append("\\bottomrule")
        out.append(close_tab)
        return out

    def repl_table(m: re.Match) -> str:
        t_id, t_bold, t_rest, t_body = m.groups()
        t_cap_tex = f"\\textbf{{{format_inline_md(t_bold.strip())}}} {format_inline_md(t_rest.strip())}"
        label_id = t_id.lower().replace(".", "_")

        if t_id in ("2", "ED1"):
            tabcol_pt = "3.8pt"
        elif t_id in ("1", "ED2"):
            tabcol_pt = "4.2pt"
        else:
            tabcol_pt = "4.0pt"
        tex_lines = [
            "\\begin{table}[!htbp]",
            f"\\caption{{{t_cap_tex}}}",
            f"\\label{{tab:{label_id}}}",
            "\\vspace{2pt}",
            "\\centering",
            "\\footnotesize",
            "\\renewcommand{\\arraystretch}{1.12}",
            f"\\setlength{{\\tabcolsep}}{{{tabcol_pt}}}",
        ]

        panel_matches = list(
            re.finditer(
                r"\*\*([ab]\.\s+[^\n]+?)\*\*\n\n((?:\|[^\n]+\n?)+)",
                t_body.strip() + "\n",
            )
        )
        if panel_matches:
            for p_idx, pm in enumerate(panel_matches):
                if p_idx > 0:
                    tex_lines.append("\\vspace{6pt}")
                p_head, p_grid = pm.groups()
                tex_lines.extend(build_single_tabular(t_id, p_grid, panel_header=p_head.strip()))
        else:
            tex_lines.extend(build_single_tabular(t_id, t_body))

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

        if s == "@@BIBLIOGRAPHY@@":
            close_lists_to(0)
            out_lines.append("\\bibliographystyle{unsrtnat}")
            out_lines.append("\\bibliography{references}")
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
            fig_w = "\\textwidth" if fnum == "1" else "0.94\\textwidth"
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
% Deterministic conversion of paper/paper2-ncs-en.md
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
    assert not non_ascii, f"Found non-ASCII chars in paper2_preprint.tex: {non_ascii}"
    OUT_TEX.write_text(full_tex, encoding="utf-8")
    print(f"Successfully wrote {len(full_tex.splitlines())} pure-ASCII lines to {OUT_TEX}")


if __name__ == "__main__":
    convert()
