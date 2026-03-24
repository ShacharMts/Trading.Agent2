"""Recommendation engine for the portal — refactored from scripts/recommend_mar23.py.

Supports configurable:
  - Data cutoff (default: latest available)
  - SMA filter (none, 20, 50, 100)
  - Volatility filter (None, Low, Medium, High)
  - Profit target, hold period, number of stocks
"""

import json
import numpy as np
import pandas as pd
import joblib

from src.data.loader import load_all_data
from src.features.pipeline import engineer_features
from src.utils.config import BEST_MODEL_PATH, FEATURE_COLUMNS_PATH, SCALER_PATH

BLACKLIST = {"INTC", "MU", "LRCX", "NEM"}

SYMBOL_NAMES = {
    "AAPL": "Apple", "ABBV": "AbbVie", "ABT": "Abbott Labs", "ADBE": "Adobe",
    "AIG": "AIG", "AMD": "AMD", "AMGN": "Amgen", "AMZN": "Amazon",
    "AVGO": "Broadcom", "AXP": "American Express", "BA": "Boeing",
    "BAC": "Bank of America", "BK": "BNY Mellon", "BKNG": "Booking Holdings",
    "BLK": "BlackRock", "BMY": "Bristol-Myers", "BRK.B": "Berkshire B",
    "C": "Citigroup", "CAT": "Caterpillar", "CHTR": "Charter Comm",
    "CL": "Colgate-Palmolive", "CMCSA": "Comcast", "COF": "Capital One",
    "COP": "ConocoPhillips", "COST": "Costco", "CRM": "Salesforce",
    "CSCO": "Cisco", "CVS": "CVS Health", "CVX": "Chevron",
    "DE": "John Deere", "DHR": "Danaher", "DIS": "Disney",
    "DUK": "Duke Energy", "EMR": "Emerson", "EXC": "Exelon",
    "FDX": "FedEx", "GD": "General Dynamics", "GE": "GE Aerospace",
    "GILD": "Gilead", "GS": "Goldman Sachs", "HD": "Home Depot",
    "HON": "Honeywell", "IBM": "IBM", "INTC": "Intel",
    "INTU": "Intuit", "ISRG": "Intuitive Surgical", "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase", "KO": "Coca-Cola", "LLY": "Eli Lilly",
    "LMT": "Lockheed Martin", "LOW": "Lowe's", "MA": "Mastercard",
    "MCD": "McDonald's", "MDLZ": "Mondelez", "MDT": "Medtronic",
    "MET": "MetLife", "META": "Meta", "MMM": "3M",
    "MO": "Altria", "MRK": "Merck", "MS": "Morgan Stanley",
    "MSFT": "Microsoft", "MU": "Micron", "NEE": "NextEra Energy",
    "NEM": "Newmont", "NFLX": "Netflix", "NKE": "Nike",
    "NVDA": "NVIDIA", "ORCL": "Oracle", "PEP": "PepsiCo",
    "PFE": "Pfizer", "PG": "Procter & Gamble", "PM": "Philip Morris",
    "PYPL": "PayPal", "QCOM": "Qualcomm", "RTX": "RTX",
    "SBUX": "Starbucks", "SCHW": "Schwab", "SO": "Southern Co",
    "SPG": "Simon Property", "T": "AT&T", "TGT": "Target",
    "TMO": "Thermo Fisher", "TMUS": "T-Mobile", "TSLA": "Tesla",
    "TXN": "Texas Instruments", "UNH": "UnitedHealth", "UNP": "Union Pacific",
    "UPS": "UPS", "USB": "US Bancorp", "V": "Visa",
    "VZ": "Verizon", "WBA": "Walgreens", "WFC": "Wells Fargo",
    "WMT": "Walmart", "XOM": "Exxon Mobil",
    # S&P 500 extras
    "A": "Agilent", "AAL": "American Airlines", "ABNB": "Airbnb",
    "ACGL": "Arch Capital", "ACN": "Accenture", "ADSK": "Autodesk",
    "AEE": "Ameren", "AEP": "AEP", "AFL": "Aflac",
    "ALGN": "Align Tech", "ALK": "Alaska Air", "ALL": "Allstate",
    "AMAT": "Applied Materials", "AMCR": "Amcor", "AMP": "Ameriprise",
    "AMT": "American Tower", "ANET": "Arista Networks", "ANSS": "Ansys",
    "AON": "Aon", "AOS": "A.O. Smith", "APA": "APA Corp",
    "APD": "Air Products", "APH": "Amphenol", "ARE": "Alexandria RE",
    "ATO": "Atmos Energy", "ATVI": "Activision", "AWK": "American Water",
    "AZO": "AutoZone", "BALL": "Ball Corp", "BAX": "Baxter",
    "BBY": "Best Buy", "BDX": "Becton Dickinson", "BEN": "Franklin Templeton",
    "BIIB": "Biogen", "BR": "Broadridge", "BRO": "Brown & Brown",
    "BSX": "Boston Scientific", "BWA": "BorgWarner", "BXP": "BXP",
    "CAG": "Conagra", "CAH": "Cardinal Health", "CARR": "Carrier",
    "CB": "Chubb", "CBOE": "Cboe", "CBRE": "CBRE",
    "CCI": "Crown Castle", "CCL": "Carnival", "CDNS": "Cadence",
    "CDW": "CDW", "CE": "Celanese", "CEG": "Constellation Energy",
    "CF": "CF Industries", "CFG": "Citizens Financial", "CHD": "Church & Dwight",
    "CHRW": "C.H. Robinson", "CI": "Cigna", "CINF": "Cincinnati Financial",
    "CLX": "Clorox", "CMA": "Comerica", "CME": "CME Group",
    "CMG": "Chipotle", "CMI": "Cummins", "CMS": "CMS Energy",
    "CNC": "Centene", "CNP": "CenterPoint", "COIN": "Coinbase",
    "CPAY": "Corpay", "CPRT": "Copart", "CPT": "Camden Property",
    "CRL": "Charles River", "CRWD": "CrowdStrike", "CSGP": "CoStar",
    "CTAS": "Cintas", "CTLT": "Catalent", "CTRA": "Coterra Energy",
    "CTSH": "Cognizant", "CTVA": "Corteva", "D": "Dominion Energy",
    "DAL": "Delta Air Lines", "DASH": "DoorDash", "DD": "DuPont",
    "DECK": "Deckers", "DFS": "Discover Financial", "DG": "Dollar General",
    "DGX": "Quest Diagnostics", "DLTR": "Dollar Tree", "DOV": "Dover",
    "DOW": "Dow Inc", "DPZ": "Domino's", "DRI": "Darden",
    "DTE": "DTE Energy", "DVA": "DaVita", "DVN": "Devon Energy",
    "DXCM": "DexCom", "EA": "Electronic Arts", "EBAY": "eBay",
    "ECL": "Ecolab", "ED": "Con Edison", "EFX": "Equifax",
    "EIX": "Edison Intl", "EL": "Estee Lauder", "EMN": "Eastman Chemical",
    "ENPH": "Enphase", "EOG": "EOG Resources", "EPAM": "EPAM Systems",
    "EQIX": "Equinix", "EQR": "Equity Residential", "EQT": "EQT Corp",
    "ES": "Eversource", "ESS": "Essex Property", "ETN": "Eaton",
    "ETR": "Entergy", "EVRG": "Evergy", "EW": "Edwards Lifesciences",
    "EXPD": "Expeditors", "EXPE": "Expedia", "F": "Ford",
    "FANG": "Diamondback", "FAST": "Fastenal", "FCNCA": "First Citizens",
    "FCX": "Freeport-McMoRan", "FDS": "FactSet", "FI": "Fiserv",
    "FICO": "Fair Isaac", "FIS": "FIS", "FITB": "Fifth Third",
    "FLT": "Fleetcor", "FMC": "FMC Corp", "FOX": "Fox Corp",
    "FOXA": "Fox Corp A", "FRT": "Federal Realty", "FSLR": "First Solar",
    "FTNT": "Fortinet", "FTV": "Fortive",
    "GEN": "Gen Digital", "GOOG": "Alphabet", "GOOGL": "Alphabet A",
    "GPC": "Genuine Parts", "GPN": "Global Payments", "GRMN": "Garmin",
    "GWW": "Grainger", "HAL": "Halliburton", "HAS": "Hasbro",
    "HBAN": "Huntington", "HCA": "HCA Healthcare", "HOLX": "Hologic",
    "HPE": "HPE", "HPQ": "HP Inc", "HRL": "Hormel",
    "HSIC": "Henry Schein", "HST": "Host Hotels", "HSY": "Hershey",
    "HUBB": "Hubbell", "HUM": "Humana", "HWM": "Howmet",
    "ICE": "ICE", "IDXX": "IDEXX Labs", "IEX": "IDEX",
    "IFF": "IFF", "ILMN": "Illumina", "INVH": "Invitation Homes",
    "IP": "Intl Paper", "IPG": "IPG", "IQV": "IQVIA",
    "IR": "Ingersoll Rand", "IRM": "Iron Mountain", "IT": "Gartner",
    "ITW": "Illinois Tool Works", "J": "Jacobs", "JBHT": "J.B. Hunt",
    "JCI": "Johnson Controls", "JKHY": "Jack Henry", "JNPR": "Juniper",
    "K": "Kellanova", "KDP": "Keurig Dr Pepper", "KEY": "KeyCorp",
    "KEYS": "Keysight", "KHC": "Kraft Heinz", "KIM": "Kimco Realty",
    "KLAC": "KLA Corp", "KMB": "Kimberly-Clark", "KMI": "Kinder Morgan",
    "KMX": "CarMax", "KR": "Kroger",
    "L": "Loews", "LDOS": "Leidos", "LEN": "Lennar",
    "LH": "Labcorp", "LHX": "L3Harris", "LIN": "Linde",
    "LKQ": "LKQ Corp", "LRCX": "Lam Research", "LULU": "Lululemon",
    "LUV": "Southwest", "LVS": "Las Vegas Sands", "LW": "Lamb Weston",
    "LYB": "LyondellBasell", "LYV": "Live Nation",
    "MAA": "Mid-America Apt", "MAR": "Marriott", "MCHP": "Microchip",
    "MCK": "McKesson", "MCO": "Moody's", "MDLZ": "Mondelez",
    "MOH": "Molina", "MPWR": "Monolithic Power", "MRNA": "Moderna",
    "MRVL": "Marvell", "MSCI": "MSCI", "MSI": "Motorola Solutions",
    "MTB": "M&T Bank", "MTCH": "Match Group", "MTD": "Mettler-Toledo",
    "MPC": "Marathon Petroleum", "NDAQ": "Nasdaq", "NDSN": "Nordson",
    "NOC": "Northrop Grumman", "NOW": "ServiceNow", "NRG": "NRG Energy",
    "NSC": "Norfolk Southern", "NTAP": "NetApp", "NTRS": "Northern Trust",
    "NUE": "Nucor", "NVR": "NVR", "NWS": "News Corp",
    "O": "Realty Income", "ODFL": "Old Dominion", "OKE": "ONEOK",
    "OMC": "Omnicom", "ON": "ON Semi", "ORLY": "O'Reilly Auto",
    "OTIS": "Otis", "OXY": "Occidental",
    "PANW": "Palo Alto", "PARA": "Paramount", "PAYC": "Paycom",
    "PAYX": "Paychex", "PCAR": "PACCAR", "PCG": "PG&E",
    "PFG": "Principal", "PHM": "PulteGroup", "PKG": "Packaging Corp",
    "PLD": "Prologis", "PLTR": "Palantir", "PNC": "PNC Financial",
    "PNR": "Pentair", "PNW": "Pinnacle West", "POOL": "Pool Corp",
    "PPG": "PPG Industries", "PPL": "PPL Corp", "PRU": "Prudential",
    "PSA": "Public Storage", "PSX": "Phillips 66", "PTC": "PTC",
    "PVH": "PVH Corp", "PWR": "Quanta Services",
    "RE": "Everest Re", "REG": "Regency Centers", "REGN": "Regeneron",
    "RF": "Regions Financial", "RHI": "Robert Half", "RJF": "Raymond James",
    "RL": "Ralph Lauren", "RMD": "ResMed", "ROK": "Rockwell",
    "ROL": "Rollins", "ROP": "Roper Tech", "ROST": "Ross Stores",
    "RSG": "Republic Services",
    "SBAC": "SBA Comm", "SHW": "Sherwin-Williams", "SJM": "J.M. Smucker",
    "SLB": "SLB", "SMCI": "Super Micro", "SNA": "Snap-on",
    "SNPS": "Synopsys", "SPG": "Simon Property", "SPGI": "S&P Global",
    "SRE": "Sempra", "STE": "Steris", "STLD": "Steel Dynamics",
    "STT": "State Street", "STX": "Seagate", "STZ": "Constellation Brands",
    "SWK": "Stanley B&D", "SWKS": "Skyworks", "SYF": "Synchrony",
    "SYK": "Stryker", "SYY": "Sysco",
    "TDG": "TransDigm", "TDY": "Teledyne", "TECH": "Bio-Techne",
    "TEL": "TE Connectivity", "TER": "Teradyne", "TFC": "Truist",
    "TJX": "TJX Companies", "TPR": "Tapestry", "TRGP": "Targa Resources",
    "TRMB": "Trimble", "TROW": "T. Rowe Price", "TRV": "Travelers",
    "TSCO": "Tractor Supply", "TSN": "Tyson Foods", "TT": "Trane Tech",
    "TTWO": "Take-Two", "TYL": "Tyler Tech", "TXT": "Textron",
    "UAL": "United Airlines", "UBER": "Uber", "UDR": "UDR",
    "ULTA": "Ulta Beauty", "URI": "United Rentals",
    "VFC": "VF Corp", "VICI": "VICI Properties", "VLO": "Valero",
    "VMC": "Vulcan Materials", "VRSK": "Verisk", "VRSN": "VeriSign",
    "VRTX": "Vertex Pharma", "VTR": "Ventas",
    "WAB": "Wabtec", "WAT": "Waters", "WBD": "Warner Bros",
    "WDC": "Western Digital", "WELL": "Welltower", "WEC": "WEC Energy",
    "WM": "Waste Management", "WMB": "Williams Cos", "WRB": "Berkley",
    "WRK": "WestRock", "WST": "West Pharma", "WTW": "WTW",
    "WY": "Weyerhaeuser", "WYNN": "Wynn",
    "XEL": "Xcel Energy", "XYL": "Xylem",
    "YUM": "Yum! Brands",
    "ZBH": "Zimmer Biomet", "ZBRA": "Zebra Tech", "ZION": "Zions", "ZTS": "Zoetis",
    # ETFs
    "HACK": "Cybersecurity ETF", "IGV": "Software ETF", "MAGS": "Mag 7 ETF",
    "VOO": "S&P 500 ETF", "XAR": "Aerospace & Defense ETF",
    "XLC": "Communication ETF", "XLE": "Energy ETF", "XLF": "Financial ETF",
    "XLI": "Industrial ETF", "XLK": "Technology ETF", "XLP": "Consumer Staples ETF",
    "XLRE": "Real Estate ETF", "XLU": "Utilities ETF", "XLV": "Healthcare ETF",
    "XLY": "Consumer Disc ETF",
    # Merchandise
    "ETHA": "Ethereum ETF", "GLD": "Gold ETF", "IBIT": "Bitcoin ETF",
    "SLVR": "Silver ETF", "USO": "Oil ETF",
}


def classify_volatility(atr_pct: float, daily_vol: float) -> str:
    composite = atr_pct * 0.5 + daily_vol * 0.5
    if composite < 1.8:
        return "Low"
    elif composite < 3.5:
        return "Medium"
    else:
        return "High"


class RecommendEngine:
    """Singleton-style engine that caches model and raw data."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self._raw_data = None

    def load(self):
        self.model = joblib.load(BEST_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        with open(FEATURE_COLUMNS_PATH) as f:
            self.feature_columns = json.load(f)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_raw_data(self) -> pd.DataFrame:
        if self._raw_data is None:
            self._raw_data = load_all_data()
            self._raw_data["DateTime"] = pd.to_datetime(self._raw_data["DateTime"])
        return self._raw_data

    def get_available_dates(self) -> list[str]:
        """Return sorted unique dates available in the data."""
        raw = self._get_raw_data()
        dates = raw["DateTime"].dt.date.unique()
        return sorted([str(d) for d in dates])

    def get_chart_data(self, symbol: str, period: str, cutoff: str | None = None, sma: int | None = None) -> dict:
        """Return OHLCV candle data with optional SMA for a symbol.

        period: 'ytd', '1m', '1w'
        sma: SMA window to compute (7, 20, 50, 100) or None
        Returns dict with 'candles' list and 'name' string.
        """
        raw = self._get_raw_data()
        sym = raw[raw["Symbol"] == symbol].sort_values("DateTime").copy()
        if sym.empty:
            return {"candles": [], "name": SYMBOL_NAMES.get(symbol, symbol)}

        if cutoff:
            cutoff_ts = pd.Timestamp(cutoff)
            sym = sym[sym["DateTime"] <= cutoff_ts]

        # Compute SMA on full data before slicing for period
        if sma and sma > 0:
            sym["sma"] = sym["Close"].rolling(window=sma, min_periods=sma).mean()
        else:
            sym["sma"] = np.nan

        if period == "1w":
            sym = sym.tail(7 * 7)  # ~7 bars/day * 7 days
        elif period == "1m":
            sym = sym.tail(7 * 22)  # ~22 trading days
        elif period == "ytd":
            current_year = sym["DateTime"].dt.year.max()
            sym = sym[sym["DateTime"].dt.year == current_year]

        candles = []
        for _, row in sym.iterrows():
            c = {
                "time": row["DateTime"].isoformat(),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            }
            if pd.notna(row["sma"]):
                c["sma"] = round(float(row["sma"]), 2)
            candles.append(c)

        return {"candles": candles, "name": SYMBOL_NAMES.get(symbol, symbol)}

    def recommend(
        self,
        num_stocks: int = 10,
        expected_profit_pct: float = 5.0,
        period_days: int = 14,
        sma_filter: int | None = 100,
        vol_filter: str | None = "Low",
        cutoff: str | None = None,
    ) -> list[dict]:
        if not self.is_loaded:
            self.load()

        raw = self._get_raw_data()

        # Apply cutoff
        if cutoff:
            cutoff_ts = pd.Timestamp(cutoff)
        else:
            cutoff_ts = raw["DateTime"].max()
        data = raw[raw["DateTime"] <= cutoff_ts].copy()
        if data.empty:
            return []

        featured = engineer_features(data)
        if featured.empty:
            return []

        # Per-symbol volatility stats
        symbol_vol = {}
        symbol_vol_class = {}
        for symbol in data["Symbol"].unique():
            sym_raw = data[data["Symbol"] == symbol].sort_values("DateTime")
            if len(sym_raw) < 20:
                continue
            sym_raw = sym_raw.copy()
            sym_raw["date"] = sym_raw["DateTime"].dt.date
            daily = sym_raw.groupby("date")["Close"].last()
            daily_returns = daily.pct_change().dropna() * 100
            if len(daily_returns) < 5:
                continue

            max_gains = []
            closes = daily.values
            window = min(period_days, len(closes) - 1)
            for i in range(len(closes) - window):
                future_max = np.max(closes[i + 1 : i + window + 1])
                gain = (future_max - closes[i]) / closes[i] * 100
                max_gains.append(gain)

            current_close = sym_raw.iloc[-1]["Close"]
            atr_series = sym_raw["Close"].diff().abs().rolling(14).mean()
            atr_val = atr_series.iloc[-1] if len(atr_series) > 14 else 0
            atr_pct = (atr_val / current_close * 100) if current_close > 0 else 0
            daily_vol = daily_returns.std() if len(daily_returns) > 1 else 0

            vol_label = classify_volatility(atr_pct, daily_vol)
            symbol_vol_class[symbol] = {"label": vol_label, "atr_pct": atr_pct, "daily_vol": daily_vol}

            if max_gains:
                symbol_vol[symbol] = {
                    "avg_max_gain": np.mean(max_gains),
                    "pct_achieving": np.mean([g >= expected_profit_pct for g in max_gains]) * 100,
                }

        # Per-symbol YTD/month stats
        as_of_date = cutoff_ts.date()
        symbol_stats = {}
        for symbol in data["Symbol"].unique():
            sym_raw = data[data["Symbol"] == symbol].sort_values("DateTime")
            if sym_raw.empty:
                continue
            current_close = sym_raw.iloc[-1]["Close"]
            year_data = sym_raw[sym_raw["DateTime"].dt.year == as_of_date.year]
            first_close = year_data.iloc[0]["Close"] if not year_data.empty else current_close
            ytd_pct = (current_close - first_close) / first_close * 100
            daily_closes = sym_raw.groupby(sym_raw["DateTime"].dt.date)["Close"].last()
            month_ago_close = daily_closes.iloc[-20] if len(daily_closes) >= 20 else daily_closes.iloc[0]
            month_pct = (current_close - month_ago_close) / month_ago_close * 100
            symbol_stats[symbol] = {"ytd_pct": ytd_pct, "month_pct": month_pct}

        # Latest bar per symbol
        latest_rows = []
        for symbol in featured["Symbol"].unique():
            sym_data = featured[featured["Symbol"] == symbol].sort_values("DateTime")
            last_row = sym_data.iloc[-1:]
            if last_row[self.feature_columns].isna().any(axis=1).iloc[0]:
                if len(sym_data) > 1:
                    last_row = sym_data.iloc[-2:-1]
            latest_rows.append(last_row)

        if not latest_rows:
            return []

        latest_df = pd.concat(latest_rows, ignore_index=True)
        latest_df[self.feature_columns] = latest_df[self.feature_columns].fillna(0)
        if latest_df.empty:
            return []

        X = latest_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        if isinstance(self.model, dict):
            probas = [m.predict_proba(X_scaled)[:, 1] for m in self.model.values()]
            proba = np.mean(probas, axis=0)
        else:
            proba = self.model.predict_proba(X_scaled)[:, 1]
        latest_df["buy_probability"] = proba

        market_regime_val = 0.0
        if "market_regime" in latest_df.columns:
            market_regime_val = latest_df["market_regime"].median()

        # Scoring
        feasibility_scores = []
        for _, row in latest_df.iterrows():
            symbol = row["Symbol"]
            if symbol in BLACKLIST:
                feasibility_scores.append(0.0)
                continue

            vol_data = symbol_vol.get(symbol, {})
            model_conf = proba[row.name]
            hist_pct = vol_data.get("pct_achieving", 0.0)
            avg_gain = vol_data.get("avg_max_gain", 0.0)

            min_feasibility = max(15.0, 35.0 - expected_profit_pct * 2.0)
            if hist_pct < min_feasibility:
                feasibility_scores.append(0.0)
                continue

            atr_val = row.get("atr_14", None)
            close = row["Close"]
            if atr_val is not None and pd.notna(atr_val) and close > 0:
                atr_pct_of_price = atr_val / close * 100
                if atr_pct_of_price > expected_profit_pct * 0.8:
                    feasibility_scores.append(0.0)
                    continue

            gain_ratio = min(avg_gain / expected_profit_pct, 2.0) if expected_profit_pct > 0 else 1.0

            returns_5d = row.get("returns_5d", 0.0)
            overext_penalty = (
                max(0.5, 1.0 - (returns_5d - expected_profit_pct) / 20.0)
                if (pd.notna(returns_5d) and returns_5d > expected_profit_pct)
                else 1.0
            )

            momentum_decel = row.get("momentum_decel", 0.0)
            decel_penalty = (
                max(0.7, 1.0 + momentum_decel / 30.0)
                if (pd.notna(momentum_decel) and momentum_decel < -3.0)
                else 1.0
            )

            sma_20 = row.get("sma_20", None)
            mean_rev_factor = 1.0
            if sma_20 is not None and pd.notna(sma_20) and sma_20 > 0:
                if close < sma_20 and pd.notna(momentum_decel) and momentum_decel > 0:
                    mean_rev_factor = 1.15
                elif close > sma_20 * 1.05:
                    mean_rev_factor = 0.85
            if sma_20 is not None and pd.notna(sma_20) and close > sma_20:
                if hist_pct < 50.0:
                    mean_rev_factor *= 0.75

            est_ret = avg_gain * model_conf
            max_reasonable = expected_profit_pct * 2.0
            profit_cap_penalty = (
                max(0.5, 1.0 - (est_ret - max_reasonable) / 20.0) if est_ret > max_reasonable else 1.0
            )

            regime_penalty = 1.0
            if market_regime_val < -0.02:
                if model_conf < 0.6:
                    regime_penalty = 0.7
                elif model_conf < 0.7:
                    regime_penalty = 0.85

            combined = (
                0.25 * model_conf + 0.40 * (hist_pct / 100.0) + 0.35 * min(gain_ratio, 1.0)
            ) * overext_penalty * decel_penalty * mean_rev_factor * profit_cap_penalty * regime_penalty
            feasibility_scores.append(combined)

        latest_df["combined_score"] = feasibility_scores
        latest_df["score"] = np.clip((np.array(feasibility_scores) * 100).astype(int), 1, 100)
        latest_df["est_return"] = latest_df.apply(
            lambda r: symbol_vol.get(r["Symbol"], {}).get("avg_max_gain", 0.0) * proba[r.name], axis=1
        )
        latest_df["est_target_price"] = latest_df["Close"] * (1 + latest_df["est_return"] / 100)

        if "atr_14" in latest_df.columns:
            atr = latest_df["atr_14"]
            atr_pct = atr / latest_df["Close"]
            median_atr_pct = atr_pct.median()
            multiplier = np.where(
                atr_pct > median_atr_pct,
                np.clip(2.0 + (atr_pct - median_atr_pct) / median_atr_pct, 2.0, 3.0),
                np.clip(2.0 - (median_atr_pct - atr_pct) / median_atr_pct * 0.5, 1.5, 2.0),
            )
            latest_df["est_stop_loss"] = latest_df["Close"] - multiplier * atr
        else:
            latest_df["est_stop_loss"] = latest_df["Close"] * 0.97

        # Sort and filter
        latest_df = latest_df.sort_values("combined_score", ascending=False).reset_index(drop=True)

        # SMA filter
        if sma_filter:
            sma_col = f"sma_{sma_filter}"
            above_sma_mask = latest_df.apply(
                lambda r: r.get(sma_col) is not None
                and pd.notna(r.get(sma_col))
                and r["Close"] > r[sma_col],
                axis=1,
            )
            latest_df = latest_df[above_sma_mask].reset_index(drop=True)

        # Volatility filter
        if vol_filter and vol_filter != "None":
            vol_mask = latest_df["Symbol"].apply(
                lambda s: symbol_vol_class.get(s, {}).get("label", "Unknown") == vol_filter
            )
            latest_df = latest_df[vol_mask].reset_index(drop=True)

        # Diverse selection
        selected_indices = []
        category_counts = {}
        for idx, row in latest_df.iterrows():
            if len(selected_indices) >= num_stocks:
                break
            cat = row.get("category", "unknown")
            count = category_counts.get(cat, 0)
            max_per_cat = 2 if cat in ("merchandise", "etfs") else 3
            if count >= max_per_cat:
                continue
            penalty = 0.70**count
            adj_score = row["combined_score"] * penalty
            if len(selected_indices) == 0 or adj_score >= latest_df.loc[selected_indices[-1], "combined_score"] * 0.5:
                selected_indices.append(idx)
                category_counts[cat] = count + 1

        if len(selected_indices) < num_stocks:
            remaining = latest_df[~latest_df.index.isin(selected_indices)]
            needed = num_stocks - len(selected_indices)
            selected_indices.extend(remaining.head(needed).index.tolist())

        top = latest_df.loc[selected_indices] if selected_indices else latest_df.head(num_stocks)

        results = []
        for _, row in top.iterrows():
            symbol = row["Symbol"]
            close = row["Close"]
            stats = symbol_stats.get(symbol, {})
            vc = symbol_vol_class.get(symbol, {})
            results.append(
                {
                    "symbol": symbol,
                    "name": SYMBOL_NAMES.get(symbol, symbol),
                    "current_price": round(float(close), 2),
                    "score": int(row["score"]),
                    "target_price": round(float(row["est_target_price"]), 2),
                    "stop_loss": round(float(row["est_stop_loss"]), 2),
                    "expected_profit": round(float(row["est_return"]), 1),
                    "ytd_pct": round(stats.get("ytd_pct", 0.0), 1),
                    "month_pct": round(stats.get("month_pct", 0.0), 1),
                    "vol_label": vc.get("label", "?"),
                    "atr_pct": round(vc.get("atr_pct", 0), 2),
                    "daily_vol": round(vc.get("daily_vol", 0), 2),
                }
            )
        return results
