#!/usr/bin/env python
"""
Assemble a styled results workbook from make_table.py TSVs, in the same layout
as results_tables_dn2.xlsx: a merged title bar per block, an italic header row,
one row per method, and a merged note row underneath.

    python make_xlsx.py \
        --block results_table_all.tsv "Our Results on Cam16  (BIHN OC-SVM - Bank A)" \
                "Complete test set. 129 slides (49 tumour + 80 normal), 1,097,356 patches, 6.67% prevalence." \
        --block results_table_macro.tsv "Our Results on Cam16  (BIHN OC-SVM - Bank A - without micro)" \
                "27 micro slides dropped, leaving 102 slides (22 macro + 80 normal), 868,877 patches, 8.30% prevalence." \
        --out results_tables_bihn.xlsx

--reference adds a DN2 comparison row to a block, given as label plus nine numbers.
Green marks the best value in each slide-level AUROC column of a block.
"""
import argparse
import csv

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

PLUM, PINK, DARK = '8E4A6F', 'F2D9E6', '6D2B4F'
GREEN = '4DFD4D'
FONT = 'Arial'
COLS = ['AUROC_max', 'F1_max', 'Recall_max', 'AUROC_top1', 'F1_top1', 'Recall_top1',
        'AUROC_patch', 'F1_patch', 'Recall_patch']
HILITE = ['AUROC_max', 'AUROC_top1', 'AUROC_patch']

thin = Side(style='thin', color=DARK)
med = Side(style='medium', color=PLUM)
BOX = Border(left=med, right=med, top=med, bottom=med)


def read_tsv(p):
    with open(p, newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    return [(r['method'], [float(r[c]) for c in COLS]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', nargs=3, action='append', required=True,
                    metavar=('TSV', 'TITLE', 'NOTE'))
    ap.add_argument('--reference', nargs=10, action='append', default=[],
                    help='LABEL followed by the 9 column values, in header order')
    ap.add_argument('--sheet', default='BIHN Results')
    ap.add_argument('--out', default='results_tables_bihn.xlsx')
    args = ap.parse_args()

    ncol = len(COLS) + 1
    wb = Workbook()
    ws = wb.active
    ws.title = args.sheet
    ws.column_dimensions['A'].width = 34
    for i in range(2, ncol + 1):
        ws.column_dimensions[get_column_letter(i)].width = 13.5

    refs = [(r[0], [float(x) for x in r[1:]]) for r in args.reference]
    r = 1
    for tsv, title, note in args.block:
        data = read_tsv(tsv)
        if refs:
            data = refs + data                              # DN2 row first, for comparison

        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=ncol)
        c = ws.cell(r, 1, title)
        c.font = Font(name=FONT, bold=True, italic=True, color='FFFFFF', size=11)
        c.fill = PatternFill('solid', fgColor=PLUM)
        c.alignment = Alignment(horizontal='center', vertical='center')
        for i in range(1, ncol + 1):
            ws.cell(r, i).border = BOX
        r += 1

        ws.cell(r, 1, 'Method')
        for i, name in enumerate(COLS, start=2):
            ws.cell(r, i, name)
        for i in range(1, ncol + 1):
            c = ws.cell(r, i)
            c.font = Font(name=FONT, bold=True, italic=True, color=PLUM, size=10)
            c.fill = PatternFill('solid', fgColor=PINK)
            c.alignment = Alignment(horizontal='center')
            c.border = BOX
        r += 1

        best = {j: max(v[j] for _, v in data) for j in range(len(COLS))}
        first = r
        for label, vals in data:
            ws.cell(r, 1, label).font = Font(name=FONT, bold=True, size=10)
            ws.cell(r, 1).alignment = Alignment(horizontal='left')
            for j, v in enumerate(vals):
                c = ws.cell(r, j + 2, round(v, 2))
                c.number_format = '0.00'
                c.font = Font(name=FONT, size=10)
                c.alignment = Alignment(horizontal='center')
                if COLS[j] in HILITE and abs(v - best[j]) < 1e-9:
                    c.fill = PatternFill('solid', fgColor=GREEN)
                    c.font = Font(name=FONT, size=10, bold=True)
            for i in range(1, ncol + 1):
                ws.cell(r, i).border = BOX
            r += 1

        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=ncol)
        c = ws.cell(r, 1, note)
        c.font = Font(name=FONT, size=9)
        c.alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
        for i in range(1, ncol + 1):
            ws.cell(r, i).border = BOX
        ws.row_dimensions[r].height = 46
        r += 2
        print(f'{title}: {len(data)} rows (data rows {first}-{r - 3})')

    ws.freeze_panes = 'B1'
    wb.save(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
