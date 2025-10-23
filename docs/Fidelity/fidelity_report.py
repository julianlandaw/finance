#!/usr/bin/env python3
"""
fidelity_report.py

Reads a Fidelity CSV (columns described by the user), computes a concise but detailed
summary (overall totals, gains, top holdings, distributions), and writes:
 - an HTML file (self-contained, with inline base64 charts)
 - a PDF file (using reportlab, with charts embedded)

Usage:
    python fidelity_report.py --csv fidelity_positions.csv --out-dir reports --top-n 10

Author: ChatGPT (GPT-5 Thinking mini)
"""
import yfinance as yf
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------------------------
# Helpers for parsing values
# ---------------------------
def parse_currency_series(s: pd.Series) -> pd.Series:
    """Strip common currency formatting and convert to float. Accepts $, commas, parentheses for negatives."""
    def parse(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, (int, float, np.number)):
            return float(x)
        x = str(x).strip()
        # parentheses negative
        if x.startswith('(') and x.endswith(')'):
            x = '-' + x[1:-1]
        # remove dollar signs, commas
        x = x.replace('$', '').replace(',', '').replace('%', '')
        try:
            return float(x)
        except:
            return 0.0
    return s.map(parse).astype(float)

def parse_percent_series(s: pd.Series) -> pd.Series:
    """Strip percent symbols and convert to float fraction (e.g. '1.2%' -> 1.2). We'll keep percent as percent number."""
    def parse(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, (int, float, np.number)):
            return float(x)
        x = str(x).strip().replace('%', '').replace(',', '')
        try:
            return float(x)
        except:
            return 0.0
    return s.map(parse).astype(float)

# ---------------------------
# Chart helpers
# ---------------------------
def fig_to_base64_png(fig, close=True, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    if close:
        plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{b64}"

def make_pie_chart(labels, sizes, title="", autopct='%1.1f%%'):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.pie(sizes, labels=labels, autopct=autopct, startangle=90)
    ax.axis('equal')
    ax.set_title(title)
    return fig

def make_bar_chart(names, values, title="", xlabel="", ylabel="", rotate_xticks=45):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(names, rotation=rotate_xticks, ha='right')
    ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

# ---------------------------
# Report generation
# ---------------------------
def generate_summary(df, top_n=10):
    """
    Returns a dict with summary statistics and DataFrames for tables
    """
    # ensure needed numeric columns exist
    expected_numeric = {
        'Quantity': 0.0,
        'Last Price': 0.0,
        'Current Value': 0.0,
        "Today's Gain/Loss Dollar": 0.0,
        "Today's Gain/Loss Percent": 0.0,
        'Total Gain/Loss Dollar': 0.0,
        'Total Gain/Loss Percent': 0.0,
        'Cost Basis Total': 0.0,
        'Average Cost Basis': 0.0,
        'Percent of Account': 0.0
    }
    # map common column name variants to the canonical names
    colmap = {}
    # build canonical presence map
    for c in df.columns:
        c_clean = c.strip()
        colmap[c_clean] = c_clean
    # attempt to rename to canonical keys if exact matches
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Parse numeric columns if present
    for key in expected_numeric.keys():
        if key in df.columns:
            if 'Percent' in key or 'Percent of Account' in key:
                df[key + '_num'] = parse_percent_series(df[key])
            else:
                df[key + '_num'] = parse_currency_series(df[key])
        else:
            df[key + '_num'] = 0.0

    total_current_value = df['Current Value_num'].sum()
    total_today_gain = df["Today's Gain/Loss Dollar_num"].sum()
    total_total_gain = df['Total Gain/Loss Dollar_num'].sum()

    overall = {
        'total_current_value': total_current_value,
        'total_today_gain': total_today_gain,
        'total_total_gain': total_total_gain,
        'total_positions': len(df),
    }

    # Top holdings by Current Value
    top_by_value = df.sort_values('Current Value_num', ascending=False).head(top_n)
    top_by_value_table = top_by_value.loc[:, ['Account Number','Account Name','Sleeve Name','Symbol','Description','Quantity','Last Price','Current Value','Total Gain/Loss Dollar']]
    # clean display: replace NaN with ''
    top_by_value_table = top_by_value_table.fillna('')

    # Top movers today
    top_by_today = df.sort_values("Today's Gain/Loss Dollar_num", ascending=False).head(top_n)
    top_by_today_table = top_by_today.loc[:, ['Symbol','Description','Quantity','Last Price',"Today's Gain/Loss Dollar","Today's Gain/Loss Percent","Current Value"]].fillna('')

    # Distribution by Type
    if 'Type' in df.columns:
        dist_type = df.groupby('Type').agg({'Current Value_num':'sum'}).reset_index()
        dist_type['pct'] = dist_type['Current Value_num'] / total_current_value * 100
        dist_type = dist_type.sort_values('Current Value_num', ascending=False)
    else:
        dist_type = pd.DataFrame(columns=['Type','Current Value_num','pct'])

    # Distribution by Sleeve Name (useful for sleeves)
    if 'Sleeve Name' in df.columns:
        dist_sleeve = df.groupby('Sleeve Name').agg({'Current Value_num':'sum'}).reset_index()
        dist_sleeve['pct'] = dist_sleeve['Current Value_num'] / total_current_value * 100
        dist_sleeve = dist_sleeve.sort_values('Current Value_num', ascending=False)
    else:
        dist_sleeve = pd.DataFrame(columns=['Sleeve Name','Current Value_num','pct'])

    # small stats: weighted avg total gain percent
    if df['Current Value_num'].sum() > 0:
        weighted_total_gain_pct = (df['Total Gain/Loss Percent_num'] * df['Current Value_num']).sum() / df['Current Value_num'].sum()
    else:
        weighted_total_gain_pct = 0.0

    summary = {
        'overall': overall,
        'top_by_value_table': top_by_value_table,
        'top_by_today_table': top_by_today_table,
        'dist_type': dist_type,
        'dist_sleeve': dist_sleeve,
        'weighted_total_gain_pct': weighted_total_gain_pct
    }
    return summary

def build_html_report(summary, df, charts_b64, out_path):
    """Creates a simple self-contained HTML file with inline charts (base64)"""
    overall = summary['overall']
    dist_type = summary['dist_type']
    dist_sleeve = summary['dist_sleeve']
    top_by_value_html = summary['top_by_value_table'].to_html(index=False, float_format='{:,.2f}'.format)
    top_by_today_html = summary['top_by_today_table'].to_html(index=False, float_format='{:,.2f}'.format)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Fidelity Positions Report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; }}
    h1,h2,h3 {{ color: #222; }}
    .summary {{ display:flex; gap:24px; flex-wrap:wrap; }}
    .card {{ border:1px solid #ddd; padding:12px; border-radius:6px; min-width:200px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
    table {{ border-collapse: collapse; width:100%; margin-bottom:18px; }}
    table th, table td {{ border:1px solid #e6e6e6; padding:6px; text-align:left; font-size:13px; }}
    .charts {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:12px; }}
    .chart {{ max-width:600px; }}
  </style>
</head>
<body>
  <h1>Fidelity Positions — Report</h1>
  <p>Generated summary for <strong>{int(overall['total_positions'])}</strong> positions. Total current value: <strong>${overall['total_current_value']:,.2f}</strong>.</p>

  <div class="summary">
    <div class="card">
      <h3>Totals</h3>
      <p><strong>Portfolio value:</strong> ${overall['total_current_value']:,.2f}</p>
      <p><strong>Today's gain/loss:</strong> ${overall['total_today_gain']:,.2f}</p>
      <p><strong>Total gain/loss:</strong> ${overall['total_total_gain']:,.2f}</p>
    </div>

    <div class="card">
      <h3>Weighted gain%</h3>
      <p><strong>{summary['weighted_total_gain_pct']:.2f}%</strong> (weighted by current value)</p>
    </div>
  </div>

  <h2>Charts</h2>
  <div class="charts">
    <div class="chart"><img src="{charts_b64.get('by_type')}" alt="by_type" style="max-width:100%"></div>
    <div class="chart"><img src="{charts_b64.get('top_holdings')}" alt="top_holdings" style="max-width:100%"></div>
    <div class="chart"><img src="{charts_b64.get('by_sleeve')}" alt="by_sleeve" style="max-width:100%"></div>
  </div>

  <h2>Top {len(summary['top_by_value_table'])} Holdings by Value</h2>
  {top_by_value_html}

  <h2>Top Movers Today</h2>
  {top_by_today_html}

  <h2>Distributions</h2>
  <h3>By Type</h3>
  {dist_type.to_html(index=False, float_format='{:,.2f}'.format)}
  <h3>By Sleeve Name</h3>
  {dist_sleeve.to_html(index=False, float_format='{:,.2f}'.format)}

  <footer style="margin-top:24px; font-size:12px; color:#666;">
    Report created by fidelity_report.py
  </footer>
</body>
</html>
"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return out_path

def build_pdf_report(summary, charts_b64, out_path_pdf):
    """Create a PDF with text summary, top tables and embedded charts using reportlab."""
    styles = getSampleStyleSheet()
    story = []
    overall = summary['overall']

    story.append(Paragraph("Fidelity Positions — Report", styles['Title']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Total positions: <b>{int(overall['total_positions'])}</b>", styles['Normal']))
    story.append(Paragraph(f"Portfolio value: <b>${overall['total_current_value']:,.2f}</b>", styles['Normal']))
    story.append(Paragraph(f"Today's gain/loss: <b>${overall['total_today_gain']:,.2f}</b>", styles['Normal']))
    story.append(Paragraph(f"Total gain/loss: <b>${overall['total_total_gain']:,.2f}</b>", styles['Normal']))
    story.append(Spacer(1, 12))

    # Add charts
    for key, title in [('by_type', 'Distribution by Type'), ('top_holdings', 'Top Holdings (by value)'), ('by_sleeve', 'Distribution by Sleeve')]:
        if charts_b64.get(key):
            story.append(Paragraph(title, styles['Heading3']))
            # convert base64 to bytes and wrap in Image
            header, data = charts_b64[key].split(',', 1)
            imgbytes = base64.b64decode(data)
            img_buf = BytesIO(imgbytes)
            try:
                im = Image(img_buf, width=450, height=250)  # reasonable default size
                story.append(im)
                story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph("Could not embed chart image: " + str(e), styles['Normal']))

    # Top holdings table (simple)
    story.append(Paragraph("Top holdings (by current value)", styles['Heading3']))
    top = summary['top_by_value_table'].head(15)
    # build table data
    table_data = [top.columns.tolist()] + top.values.tolist()
    # convert all values to strings with formatting
    formatted = []
    for row in table_data:
        newrow = []
        for v in row:
            if isinstance(v, float):
                newrow.append(f"{v:,.2f}")
            else:
                newrow.append(str(v))
        formatted.append(newrow)
    t = Table(formatted, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t)

    # write
    doc = SimpleDocTemplate(out_path_pdf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)
    return out_path_pdf

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Produce HTML and PDF summary reports from a Fidelity positions CSV.")
    parser.add_argument('--csv', '-c', required=True, help='Path to Fidelity CSV file')
    parser.add_argument('--out-dir', '-o', default='reports', help='Output directory for reports')
    parser.add_argument('--top-n', '-n', type=int, default=10, help='How many top holdings/movers to show')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = args.csv

    # read CSV (try common encodings)
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception:
        df = pd.read_csv(csv_path, dtype=str, encoding='latin1')

    # Standardize column names
    df.columns = [col.strip() for col in df.columns]

    # If the CSV contains embedded commas, pandas will still handle if quoted; otherwise we try again
    # Parse numeric columns using helper functions inside generate_summary
    summary = generate_summary(df, top_n=args.top_n)

    # Charts creation
    charts_b64 = {}

    # by type pie chart
    if not summary['dist_type'].empty:
        labels = summary['dist_type']['Type'].astype(str).tolist()
        sizes = summary['dist_type']['Current Value_num'].tolist()
        fig = make_pie_chart(labels, sizes, title="By Type (% of portfolio)")
        charts_b64['by_type'] = fig_to_base64_png(fig)
    else:
        charts_b64['by_type'] = ''

    # top holdings bar chart
    topvals = summary['top_by_value_table'].head(args.top_n)
    if len(topvals) > 0:
        names = (topvals['Symbol'].astype(str) + " (" + topvals['Description'].astype(str).str.slice(0,20) + ")").tolist()
        values = parse_currency_series(topvals['Current Value']).tolist() if 'Current Value' in topvals.columns else topvals.get('Current Value_num', pd.Series([0]*len(topvals))).tolist()
        fig = make_bar_chart(names, values, title=f"Top {len(names)} Holdings by Current Value", ylabel="Current Value ($)")
        charts_b64['top_holdings'] = fig_to_base64_png(fig)
    else:
        charts_b64['top_holdings'] = ''

    # by sleeve
    if not summary['dist_sleeve'].empty:
        labels = summary['dist_sleeve']['Sleeve Name'].astype(str).tolist()
        sizes = summary['dist_sleeve']['Current Value_num'].tolist()
        fig = make_pie_chart(labels, sizes, title="By Sleeve (% of portfolio)")
        charts_b64['by_sleeve'] = fig_to_base64_png(fig)
    else:
        charts_b64['by_sleeve'] = ''

    # Build files
    html_path = os.path.join(args.out_dir, 'fidelity_report.html')
    pdf_path = os.path.join(args.out_dir, 'fidelity_report.pdf')

    build_html_report(summary, df, charts_b64, html_path)
    build_pdf_report(summary, charts_b64, pdf_path)

    print(f"Reports written:\n  HTML -> {html_path}\n  PDF  -> {pdf_path}")

    symbols = df["Symbol"].unique().tolist()
    #print(float(np.array(df[df["Symbol"] == 'AAPL']["Quantity"])[0]) + 1)

    # List to store all the data
    records = []
    
    for symbol in symbols:
        print(symbol)
        shares = np.array(df[df["Symbol"] == symbol]["Quantity"])
            totshares = 0
            for i in range(len(shares)):
                totshares = totshares + float(shares[i])
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info  # Fetch metadata
            shares = np.array(df[df["Symbol"] == symbol]["Quantity"])
            currentPrice = info.get("currentPrice", "NA")
            totalprice = totshares*float(currentPrice)
            
            record = {
                "Symbol": symbol,
                "shortName": info.get("shortName", "NA"),
                "currentPrice": np.round(currentPrice,2),
                "Shares": np.round(totshares,2),
                "Total_Amount": np.round(totalprice, 2),
                "PE_Ratio": info.get("trailingPE", "NA"),
                "Beta": info.get("beta", "NA"),
                "MarketCap": info.get("marketCap", "NA"),
                "DividendYield": info.get("dividendYield", "NA"),
                "52WeekLow": info.get("fiftyTwoWeekLow", "NA"),
                "52WeekHigh": info.get("fiftyTwoWeekHigh", "NA"),
                "Sector": info.get("sector", "NA"),
                "Industry": info.get("industry", "NA")
            }
    
        except Exception as e:
            # Handle cases where ticker is invalid or missing
            record = {
                "Symbol": symbol,
                "shortName": "NA",
                "currentPrice": "NA",
                "Shares": np.round(totshares,2),
                "Total_Amount": "NA",
                "PE_Ratio": "NA",
                "Beta": "NA",
                "MarketCap": "NA",
                "DividendYield": "NA",
                "52WeekLow": "NA",
                "52WeekHigh": "NA",
                "Sector": "NA",
                "Industry": "NA"
            }
    
        records.append(record)
    
    # Convert to DataFrame
    stock_df = pd.DataFrame(records)
    
    # Save to CSV
    output_csv = "fidelity_stock_info.csv"
    stock_df.to_csv(output_csv, index=False)
    print(f"✅ Stock data saved to {output_csv}")

if __name__ == '__main__':
    main()
