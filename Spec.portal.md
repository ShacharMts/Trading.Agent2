generate a web ste to support :
generate recomandation base on the following filter :
how many symboles
how long to keep 
how much profit epected
over moving everage : select : nonm20,50,100,
symbol %|olatility/Velocity|
generate a table and graph of the recomended symboles 
table :
| Rank | Symbol | Current Price | Score | Target Price | Stop-Loss | Expected Profit |Y2D %|Last Month %|olatility/Velocity|
graph : selection , y2d,1 month, 1 week,
add option to keep the recomandation more recomandation folder


A simple HTML + vanilla JS + Chart.js approach since it avoids build tooling and can be served directly by FastAPI. OK?

use interactive graphs.

by default the latest data on the data files 
add an option to select prev date.

save as json file 

for the graphs use existing hourly candle .

up to 10 symbols.

plan to deploy it first localhost later on we will work on remote deployment

recommendation update base on user requests.

When the user selects a previous date show a calendar picker


Graph show all symbols at once or selected on the table.

volatility be a single dropdown (None / Low / Medium / High) o


Default Form Values

defaults be when the page loads:
Symbols: 10
Hold period: 14 days
Profit target: 5%
Moving average: 100
Volatility: Low


the portal have a "History" tab/section to list and view previously saved JSON recommendation files

ask questions to generate the architecture 

save file name format: rec_2026-03-24_symbolscount_seq.json

page format :
┌─────────────────────────────┐
│  Header / Logo              │
├─────────────────────────────┤
│  Filters (horizontal row)   │  ← Desktop: single row of inputs
│  [Generate] button          │    Mobile: stacks vertically
├─────────────────────────────┤
│  Results Table (full width) │  ← Mobile: horizontal scroll
├─────────────────────────────┤
│  Chart (full width)         │  ← Fills available width
├─────────────────────────────┤
│  Saved Recommendations      │
└─────────────────────────────┘