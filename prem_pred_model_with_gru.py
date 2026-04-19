import pandas as pd
import understatapi
import numpy as np
import os
import re
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Soft torch import: GRU support is optional. If torch isn't installed,
# Footy_Predictor still works for the XGB/RF/ET ensemble — `use_gru` will
# silently be ignored.
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

client = understatapi.UnderstatClient()


# =========================================================================
# GRU model + sklearn-style wrapper
# =========================================================================

if TORCH_AVAILABLE:

    class GRUNet(nn.Module):
        """
        Two-branch GRU. One branch reads the home team's recent match
        sequence, the other reads the away team's. Their final hidden states
        are concatenated and passed through a dropout + linear layer to
        produce 3-class logits (away win, draw, home win).
        """
        def __init__(self, n_features, hidden_size=32, n_classes=3, dropout=0.2):
            super().__init__()
            self.home_gru = nn.GRU(n_features, hidden_size, batch_first=True)
            self.away_gru = nn.GRU(n_features, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size * 2, n_classes)

        def forward(self, home_seq, away_seq):
            _, h_home = self.home_gru(home_seq)
            _, h_away = self.away_gru(away_seq)
            h = torch.cat([h_home.squeeze(0), h_away.squeeze(0)], dim=-1)
            h = self.dropout(h)
            return self.fc(h)


    class GRUClassifier:
        """
        Lightweight wrapper that mimics the bits of the sklearn API the
        Footy_Predictor pipeline uses (`fit`, `predict_proba`), but takes
        sequence inputs (home_seqs, away_seqs) instead of a flat DataFrame.
        """
        def __init__(self, n_features, hidden_size=32, n_classes=3, lr=1e-3,
                     epochs=30, batch_size=32, dropout=0.2, draw_class_weight=1.5,
                     device=None, seed=42):
            self.n_features = n_features
            self.hidden_size = hidden_size
            self.n_classes = n_classes
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.dropout = dropout
            self.draw_class_weight = draw_class_weight
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.seed = seed
            self.model = None

        def fit(self, home_seqs, away_seqs, y, sample_weight=None):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            self.model = GRUNet(
                n_features=self.n_features,
                hidden_size=self.hidden_size,
                n_classes=self.n_classes,
                dropout=self.dropout,
            ).to(self.device)
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            home_t = torch.tensor(home_seqs, dtype=torch.float32, device=self.device)
            away_t = torch.tensor(away_seqs, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(y, dtype=torch.long, device=self.device)
            if sample_weight is None:
                sw_t = torch.ones(len(y), dtype=torch.float32, device=self.device)
            else:
                sw_t = torch.tensor(sample_weight, dtype=torch.float32, device=self.device)

            n = len(y)
            self.model.train()
            for _ in range(self.epochs):
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    logits = self.model(home_t[idx], away_t[idx])
                    loss = nn.functional.cross_entropy(logits, y_t[idx], reduction="none")
                    loss = (loss * sw_t[idx]).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return self

        def predict_proba(self, home_seqs, away_seqs):
            self.model.eval()
            with torch.no_grad():
                home_t = torch.tensor(home_seqs, dtype=torch.float32, device=self.device)
                away_t = torch.tensor(away_seqs, dtype=torch.float32, device=self.device)
                logits = self.model(home_t, away_t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs


# =========================================================================
# Main predictor
# =========================================================================


class Footy_Predictor:
    def __init__(self, roll_n=5, min_n=1, draw_threshold=0.35,
                 draw_class_weight=1.5, use_gru=True, gru_seq_len=8):
        self.roll_n = roll_n
        self.min_n = min_n
        self.draw_threshold = draw_threshold
        self.draw_class_weight = draw_class_weight
        self.eps = 1 * 10**(-6)

        # GRU config
        self.use_gru = use_gru and TORCH_AVAILABLE
        if use_gru and not TORCH_AVAILABLE:
            print("[Footy_Predictor] torch not installed — GRU disabled, "
                  "running tree ensemble only.")
        self.gru_seq_len = gru_seq_len
        # Raw per-match features fed to the GRU (one row per match per team)
        self.gru_features = [
            "xG", "xGA", "npxG", "npxGA", "scored", "missed", "pts",
            "h_a_num", "deep", "deep_allowed", "ppda_att", "ppda_def",
        ]

        self.team_map = {
            "Man City": "Manchester City",
            "Man United": "Manchester United",
            "Nott'm Forest": "Nottingham Forest",
            "Newcastle": "Newcastle United",
            "Wolves": "Wolverhampton Wanderers"
        }

        years = sorted(
            [
                fname.replace(".csv", "")
                for fname in os.listdir(".")
                if re.fullmatch(r"\d{4}\.csv", fname)
            ]
        )
        if not years:
            raise ValueError("No season CSV files found. Expected files like 2021.csv, 2022.csv, ...")
        self.years = years
        self.latest_year = years[-1]

        seasons = []
        for y in years:
            season_data = client.league("EPL").get_team_data(season=y)
            seasons.append(season_data)

        season_dfs = []
        for s in seasons:
            rows = []
            for team_id, team_data in s.items():
                team_name = team_data["title"]
                for match in team_data["history"]:
                    row = match.copy()
                    row["team"] = team_name
                    row["team_id"] = team_id
                    rows.append(row)
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.sort_values(["team", "date"])
            df = df.set_index(["team", "date"])

            df['deep_ratio'] = round(df['deep'] / (df['deep'] + df['deep_allowed'] + self.eps), 2)
            df = self.extract_ppda(df, col="ppda", prefix="ppda")
            df = self.extract_ppda(df, col="ppda_allowed", prefix="ppda_allowed")

            season_dfs.append(df)

        mean_metrics = ["xG", "xGA", "npxGD", "npxG", "npxGA", "xpts", "deep_ratio"]
        momentum_metrics = ["xG", "xGA", "npxGD", "xpts"]
        venue_metrics = ["xG", "xGA", "npxGD", "xpts"]
        slope_metrics = ["xG", "xGA", "npxGD", "xpts"]

        base_key_stats = [
            "ppda_att", "ppda_def", "ppda_allowed_att", "ppda_allowed_def",
            "rolling_xG_5", "rolling_xGA_5", "rolling_npxGD_5", "rolling_npxG_5",
            "rolling_npxGA_5", "rolling_xpts_5", "rolling_deep_ratio_5", "rolling_xG_std_5",
            "rolling_xGA_std_5", "rolling_npxGD_std_5"
        ]
        momentum_key_stats = [f"momentum_{m}_3_10" for m in momentum_metrics]
        schedule_key_stats = ["days_rest", "matches_last_7d"]
        venue_key_stats = [
            f"{venue}_form_{metric}_{self.roll_n}"
            for metric in venue_metrics
            for venue in ("home", "away")
        ]
        strength_balance_key_stats = ["attack_strength_5", "defense_strength_5"]
        consistency_key_stats = ["cv_xG_5", "cv_xGA_5", "cv_npxGD_5"]
        result_form_key_stats = ["ppg_last_5", "win_rate_last_5", "draw_rate_last_5"]
        finishing_key_stats = ["finishing_overperf_5", "defensive_overperf_5"]
        slope_key_stats = [f"slope_{metric}_{self.roll_n}" for metric in slope_metrics]
        venue_dependence_key_stats = [
            f"home_edge_{metric}_{self.roll_n}" for metric in venue_metrics
        ] + [
            f"away_dropoff_{metric}_{self.roll_n}" for metric in venue_metrics
        ]
        table_proxy_key_stats = [
            "cum_points_pre", "games_played_pre",
            "points_per_game_pre", "league_rank_pre",
            "rank_form_5", "points_gap_top_pre"]
        self.key_stats = (
            base_key_stats
            + momentum_key_stats
            + schedule_key_stats
            + venue_key_stats
            + strength_balance_key_stats
            + consistency_key_stats
            + result_form_key_stats
            + finishing_key_stats
            + slope_key_stats
            + venue_dependence_key_stats
            + table_proxy_key_stats
        )

        # ---- NEW: capture raw per-match stats for GRU sequences ----
        # Built BEFORE the feature-engineering loop modifies each season df.
        seasons_raw_dict = {}

        for i, s in enumerate(season_dfs):
            # ---- NEW: capture raw per-match stats before feature engineering ----
            raw = s.copy()
            if "h_a" in raw.columns:
                raw["h_a_num"] = (raw["h_a"] == "h").astype(float)
            else:
                raw["h_a_num"] = 0.0
            for col in ["xG", "xGA", "npxG", "npxGA", "scored", "missed", "pts", "deep", "deep_allowed"]:
                if col in raw.columns:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce")
            available = [c for c in self.gru_features if c in raw.columns]
            raw = raw[available].dropna()
            seasons_raw_dict[self.years[i]] = raw
            # --------------------------------------------------------------------

            for m_col in mean_metrics:
                s = self.rolling_mean(s, m_col)

            for sd_col in ["xG", "xGA", "npxGD"]:
                s = self.rolling_std(s, sd_col)

            for metric in momentum_metrics:
                s[f"rolling_{metric}_3"] = self.rolling_mean_n(s, metric, 3)
                s[f"rolling_{metric}_10"] = self.rolling_mean_n(s, metric, 10)
                s[f"momentum_{metric}_3_10"] = s[f"rolling_{metric}_3"] - s[f"rolling_{metric}_10"]

            s["days_rest"] = self.days_rest(s)
            s["matches_last_7d"] = self.matches_in_last_days(s, days=7)

            for metric in venue_metrics:
                s[f"home_form_{metric}_{self.roll_n}"] = self.rolling_mean_by_venue(s, metric, venue="h")
                s[f"away_form_{metric}_{self.roll_n}"] = self.rolling_mean_by_venue(s, metric, venue="a")

            s = self.add_strength_balance_features(s)
            s = self.add_consistency_features(s)
            s = self.add_result_form_features(s)
            s = self.add_finishing_features(s)
            for metric in slope_metrics:
                s[f"slope_{metric}_{self.roll_n}"] = self.rolling_slope(s, metric, window=self.roll_n)
            s = self.add_venue_dependence_features(s, venue_metrics)
            s = self.add_table_proxy_features(s)

            s = s.reset_index()[["team", "date"] + self.key_stats]
            s_clean = s.set_index(["team", "date"])
            s_clean = s_clean.dropna()

            season_dfs[i] = s_clean
        self.seasons = dict(zip(years, season_dfs))
        self.seasons_raw = seasons_raw_dict  # ---- NEW ----

        game_results = []
        keep_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        for y in years:
            df = pd.read_csv(f'{y}.csv')[keep_cols]

            df["HomeTeam"] = df["HomeTeam"].replace(self.team_map)
            df = df.apply(pd.to_numeric, errors="ignore")
            df["FTR"] = np.select(
                [df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]],
                [1, 0],
                default=-1
            )
            drop_cols = ["FTHG", "FTAG"]
            df = df.drop(columns=drop_cols)
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise").dt.date
            df = df.sort_values("Date")

            first_game_mask = ((df["HomeTeam"] == "Arsenal") | (df["AwayTeam"] == "Arsenal"))
            first_ars_index = df.loc[first_game_mask].index[:2]
            df = df.drop(index=first_ars_index)

            game_results.append(df)
        self.results = dict(zip(years, game_results))
        self.results_history = (
            pd.concat(game_results, ignore_index=True)
            .sort_values("Date")
            .reset_index(drop=True)
        )

    """
    Now it's time to build the methods that we'll use to clean our dfs,
    engineer new features, merging our dataframes,
    and train some models based on these new features
    """

    def extract_ppda(self, df, col, prefix):
        """
        Extract att/def from Understat PPDA dict column
        and compute the PPDA ratio.
        """
        df[f"{prefix}_att"] = df[col].apply(
            lambda x: x.get("att") if isinstance(x, dict) else np.nan
        )
        df[f"{prefix}_def"] = df[col].apply(
            lambda x: x.get("def") if isinstance(x, dict) else np.nan
        )
        df[prefix] = df[f"{prefix}_att"] / (df[f"{prefix}_def"] + self.eps)
        return df

    def rolling_mean(self, df, col):
        out_col = f'rolling_{col}_{self.roll_n}'
        df[out_col] = (
            df.groupby("team")[f'{col}'].shift(1)
            .rolling(window=self.roll_n, min_periods=self.min_n).mean()
        )
        return df

    def rolling_std(self, df, col):
        out_col = f'rolling_{col}_std_{self.roll_n}'
        df[out_col] = (
            df.groupby("team")[f'{col}'].shift(1)
            .rolling(window=self.roll_n, min_periods=self.min_n).std()
        )
        return df

    def rolling_mean_n(self, df, col, n):
        return (
            df.groupby("team")[col].shift(1)
            .rolling(window=n, min_periods=self.min_n).mean()
        )

    def rolling_mean_by_venue(self, df, col, venue):
        out = pd.Series(index=df.index, dtype="float64")

        for _, group in df.groupby("team", sort=False):
            group = group.sort_index(level="date")
            venue_series = group.loc[group["h_a"] == venue, col]
            rolled = venue_series.shift(1).rolling(
                window=self.roll_n, min_periods=self.min_n
            ).mean()
            out.loc[group.index] = rolled.reindex(group.index).ffill()

        return out

    def days_rest(self, df):
        dates = pd.to_datetime(df.index.get_level_values("date"))
        series = pd.Series(dates, index=df.index)
        return series.groupby(level="team").diff().dt.days

    def matches_in_last_days(self, df, days=7):
        out = pd.Series(index=df.index, dtype="float64")
        window = f"{days}D"

        for _, group in df.groupby("team", sort=False):
            group = group.sort_index(level="date")
            dt_index = pd.to_datetime(group.index.get_level_values("date"))
            ones = pd.Series(1.0, index=dt_index)
            counts = ones.shift(1).rolling(window).sum().fillna(0.0).to_numpy()
            out.loc[group.index] = counts

        return out

    def add_strength_balance_features(self, df):
        tmp = df.reset_index()
        league_xg = tmp.groupby("date")["rolling_xG_5"].transform("mean")
        league_xga = tmp.groupby("date")["rolling_xGA_5"].transform("mean")
        tmp["attack_strength_5"] = tmp["rolling_xG_5"] / (league_xg + self.eps)
        tmp["defense_strength_5"] = league_xga / (tmp["rolling_xGA_5"] + self.eps)
        return tmp.set_index(["team", "date"])

    def add_consistency_features(self, df):
        df["cv_xG_5"] = df["rolling_xG_std_5"] / (df["rolling_xG_5"].abs() + self.eps)
        df["cv_xGA_5"] = df["rolling_xGA_std_5"] / (df["rolling_xGA_5"].abs() + self.eps)
        df["cv_npxGD_5"] = df["rolling_npxGD_std_5"] / (df["rolling_npxGD_5"].abs() + self.eps)
        return df

    def add_result_form_features(self, df):
        pts_col = "pts" if "pts" in df.columns else None
        if pts_col is None:
            return df

        out_ppg = pd.Series(index=df.index, dtype="float64")
        out_win = pd.Series(index=df.index, dtype="float64")
        out_draw = pd.Series(index=df.index, dtype="float64")

        for _, group in df.groupby("team", sort=False):
            group = group.sort_index(level="date")
            shifted_pts = group[pts_col].shift(1)
            out_ppg.loc[group.index] = shifted_pts.rolling(window=5, min_periods=1).mean().to_numpy()
            out_win.loc[group.index] = (shifted_pts == 3).astype(float).rolling(window=5, min_periods=1).mean().to_numpy()
            out_draw.loc[group.index] = (shifted_pts == 1).astype(float).rolling(window=5, min_periods=1).mean().to_numpy()

        df["ppg_last_5"] = out_ppg
        df["win_rate_last_5"] = out_win
        df["draw_rate_last_5"] = out_draw
        return df

    def add_finishing_features(self, df):
        if not {"scored", "xG", "xGA", "missed"}.issubset(df.columns):
            return df

        df["finishing_delta"] = df["scored"] - df["xG"]
        df["defensive_delta"] = df["xGA"] - df["missed"]
        out_finish = pd.Series(index=df.index, dtype="float64")
        out_def = pd.Series(index=df.index, dtype="float64")

        for _, group in df.groupby("team", sort=False):
            group = group.sort_index(level="date")
            out_finish.loc[group.index] = (
                group["finishing_delta"].shift(1).rolling(window=5, min_periods=1).mean().to_numpy()
            )
            out_def.loc[group.index] = (
                group["defensive_delta"].shift(1).rolling(window=5, min_periods=1).mean().to_numpy()
            )

        df["finishing_overperf_5"] = out_finish
        df["defensive_overperf_5"] = out_def
        return df

    def rolling_slope(self, df, col, window):
        out = pd.Series(index=df.index, dtype="float64")

        def _slope(vals):
            if len(vals) < 2:
                return np.nan
            x = np.arange(len(vals), dtype=float)
            return float(np.polyfit(x, vals, 1)[0])

        for _, group in df.groupby("team", sort=False):
            group = group.sort_index(level="date")
            series = group[col].shift(1).rolling(window=window, min_periods=2).apply(_slope, raw=True)
            out.loc[group.index] = series.to_numpy()

        return out

    def add_venue_dependence_features(self, df, venue_metrics):
        for metric in venue_metrics:
            home_form_col = f"home_form_{metric}_{self.roll_n}"
            away_form_col = f"away_form_{metric}_{self.roll_n}"
            rolling_col = f"rolling_{metric}_{self.roll_n}"
            if home_form_col in df.columns and rolling_col in df.columns:
                df[f"home_edge_{metric}_{self.roll_n}"] = df[home_form_col] - df[rolling_col]
            if away_form_col in df.columns and rolling_col in df.columns:
                df[f"away_dropoff_{metric}_{self.roll_n}"] = df[rolling_col] - df[away_form_col]
        return df

    def add_table_proxy_features(self, df):
        if "pts" not in df.columns:
            return df

        cum_points = df.groupby("team")["pts"].cumsum()
        df["cum_points_pre"] = cum_points.groupby(level="team").shift(1).fillna(0.0)
        df["games_played_pre"] = df.groupby("team").cumcount()
        df["points_per_game_pre"] = df["cum_points_pre"] / (df["games_played_pre"] + self.eps)

        rank_df = df.reset_index()[["team", "date", "cum_points_pre"]].copy()
        rank_df["league_rank_pre"] = rank_df.groupby("date")["cum_points_pre"].rank(
            method="average",
            ascending=False
        )
        rank_df = rank_df.set_index(["team", "date"])
        df["league_rank_pre"] = rank_df["league_rank_pre"]

        out_rank_form = pd.Series(index=df.index, dtype="float64")
        for _, group in df.groupby("team", sort=False):
            group = group.sort_index(level="date")
            out_rank_form.loc[group.index] = (
                group["league_rank_pre"].shift(1).rolling(window=5, min_periods=1).mean().to_numpy()
            )
        df["rank_form_5"] = out_rank_form

        top_points = rank_df.reset_index().groupby("date")["cum_points_pre"].max()
        max_pts = df.index.get_level_values("date").map(top_points.to_dict())
        df["points_gap_top_pre"] = pd.Series(max_pts, index=df.index) - df["cum_points_pre"]
        return df

    def h2h_snapshot(self, home_team, away_team, match_date, max_matches=6, decay=0.8):
        history = self.results_history
        date_cutoff = pd.to_datetime(match_date).date()
        mask = (
            (history["Date"] < date_cutoff)
            & (
                ((history["HomeTeam"] == home_team) & (history["AwayTeam"] == away_team))
                | ((history["HomeTeam"] == away_team) & (history["AwayTeam"] == home_team))
            )
        )
        past = history.loc[mask].tail(max_matches)
        if past.empty:
            return {
                "h2h_points_diff_decay": 0.0,
                "h2h_draw_rate_decay": 0.0,
                "h2h_recent_count": 0.0,
            }

        weights = np.array([decay ** i for i in range(len(past) - 1, -1, -1)], dtype=float)
        weights = weights / weights.sum()

        pts_diff = []
        draw_ind = []
        for row in past.itertuples(index=False):
            if row.FTR == 0:
                home_pts, away_pts = 1.0, 1.0
            elif (row.HomeTeam == home_team and row.FTR == 1) or (row.AwayTeam == home_team and row.FTR == -1):
                home_pts, away_pts = 3.0, 0.0
            else:
                home_pts, away_pts = 0.0, 3.0
            pts_diff.append(home_pts - away_pts)
            draw_ind.append(1.0 if row.FTR == 0 else 0.0)

        return {
            "h2h_points_diff_decay": float(np.dot(weights, np.array(pts_diff, dtype=float))),
            "h2h_draw_rate_decay": float(np.dot(weights, np.array(draw_ind, dtype=float))),
            "h2h_recent_count": float(len(past)),
        }

    def add_match_context_features(self, df):
        out = df.copy()
        dates = pd.to_datetime(out["Date"])
        month_num = dates.dt.month.astype(float)
        out["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)
        out["is_festive_period"] = (
            ((dates.dt.month == 12) & (dates.dt.day >= 20))
            | ((dates.dt.month == 1) & (dates.dt.day <= 5))
        ).astype(float)
        season_start = pd.to_datetime(out.groupby("season_year")["Date"].transform("min"))
        out["days_since_season_start"] = (dates - season_start).dt.days.astype(float)

        out["home_attack_x_away_def_weakness"] = out["home_rolling_xG_5"] * out["away_rolling_xGA_5"]
        out["away_attack_x_home_def_weakness"] = out["away_rolling_xG_5"] * out["home_rolling_xGA_5"]
        out["abs_npxgd_gap"] = (out["home_rolling_npxGD_5"] - out["away_rolling_npxGD_5"]).abs()
        out["min_xg_matchup"] = out[["home_rolling_xG_5", "away_rolling_xG_5"]].min(axis=1)
        total_xg_matchup = out["home_rolling_xG_5"] + out["away_rolling_xG_5"]
        out["low_total_xg_flag"] = (total_xg_matchup < 2.2).astype(float)
        out["draw_balance_signal"] = (
            (out["home_rolling_xG_5"] - out["away_rolling_xG_5"]).abs()
            + (out["home_rolling_xGA_5"] - out["away_rolling_xGA_5"]).abs()
        )

        h2h_rows = [
            self.h2h_snapshot(h, a, d)
            for h, a, d in zip(out["HomeTeam"], out["AwayTeam"], out["Date"])
        ]
        h2h_df = pd.DataFrame(h2h_rows, index=out.index)
        out[h2h_df.columns] = h2h_df
        return out

    def add_matchup_features(self, df):
        metrics = [
            "rolling_xG_5", "rolling_xGA_5", "rolling_npxGD_5", "rolling_npxG_5",
            "rolling_npxGA_5", "rolling_xpts_5", "rolling_deep_ratio_5",
            "rolling_xG_std_5", "rolling_xGA_std_5", "rolling_npxGD_std_5",
            "momentum_xG_3_10", "momentum_xGA_3_10", "momentum_npxGD_3_10", "momentum_xpts_3_10",
            "days_rest", "matches_last_7d",
            "attack_strength_5", "defense_strength_5",
            "cv_xG_5", "cv_xGA_5", "cv_npxGD_5",
            "ppg_last_5", "win_rate_last_5", "draw_rate_last_5",
            "finishing_overperf_5", "defensive_overperf_5",
            "cum_points_pre", "games_played_pre", "points_per_game_pre", "league_rank_pre", "rank_form_5", "points_gap_top_pre",
            f"slope_xG_{self.roll_n}", f"slope_xGA_{self.roll_n}", f"slope_npxGD_{self.roll_n}", f"slope_xpts_{self.roll_n}",
            f"home_edge_xG_{self.roll_n}", f"home_edge_xGA_{self.roll_n}", f"home_edge_npxGD_{self.roll_n}", f"home_edge_xpts_{self.roll_n}",
            f"away_dropoff_xG_{self.roll_n}", f"away_dropoff_xGA_{self.roll_n}", f"away_dropoff_npxGD_{self.roll_n}", f"away_dropoff_xpts_{self.roll_n}",
        ]

        for metric in metrics:
            home_col = f"home_{metric}"
            away_col = f"away_{metric}"
            if home_col in df.columns and away_col in df.columns:
                df[f"delta_{metric}"] = df[home_col] - df[away_col]

        for metric in ["xG", "xGA", "npxGD", "xpts"]:
            home_col = f"home_home_form_{metric}_{self.roll_n}"
            away_col = f"away_away_form_{metric}_{self.roll_n}"
            if home_col in df.columns and away_col in df.columns:
                df[f"delta_venue_form_{metric}_{self.roll_n}"] = df[home_col] - df[away_col]

        return df

    def build_match_df(self, year):
        """
        Merging our dataframes in order to have a match ready dataframe that we can use our
        engineered features to train a model and make predictions
        """
        stats = self.seasons[year].reset_index()
        results = self.results[year].copy()

        home = stats.rename(columns={c: f"home_{c}" for c in self.key_stats})
        away = stats.rename(columns={c: f"away_{c}" for c in self.key_stats})

        df = (
            results.merge(
                home,
                left_on=["HomeTeam", "Date"],
                right_on=["team", "date"],
                how="inner"
            ).merge(
                away,
                left_on=["AwayTeam", "Date"],
                right_on=["team", "date"],
                how="inner"
            )
        )

        df = df.drop(columns=["team_x", "date_x", "team_y", "date_y"])
        df = self.add_matchup_features(df)
        df["season_year"] = year
        df = self.add_match_context_features(df)
        return df

    def split_xy(self, df):
        drop_cols = ["FTR", "Date", "HomeTeam", "AwayTeam", "season_year"]
        X = df.drop(columns=drop_cols)
        y = df["FTR"].map({-1: 0, 0: 1, 1: 2})
        return X, y

    def walk_forward_splits(self, year_dfs):
        cv_years = [y for y in self.years if y != self.latest_year]
        splits = []
        for i in range(1, len(cv_years)):
            train_years = cv_years[:i]
            valid_year = cv_years[i]
            train_df = pd.concat([year_dfs[y] for y in train_years], ignore_index=True)
            valid_df = year_dfs[valid_year]
            splits.append((train_df, valid_df, train_years, valid_year))
        return splits

    def walk_forward_logloss(self, model_builder, splits):
        return self.walk_forward_logloss_subset(model_builder, splits, feature_subset=None)

    def sample_weights_from_y(self, y):
        y_arr = np.asarray(y)
        return np.where(y_arr == 1, self.draw_class_weight, 1.0).astype(float)

    def fit_model_with_weights(self, model, X, y):
        sample_weight = self.sample_weights_from_y(y)
        try:
            model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            model.fit(X, y)
        return model

    def walk_forward_logloss_subset(self, model_builder, splits, feature_subset=None):
        fold_losses = []
        for train_df, valid_df, _, _ in splits:
            X_tr, y_tr = self.split_xy(train_df)
            X_val, y_val = self.split_xy(valid_df)
            if feature_subset is not None:
                X_tr = X_tr[feature_subset]
                X_val = X_val[feature_subset]
            model = model_builder()
            self.fit_model_with_weights(model, X_tr, y_tr)
            probs = model.predict_proba(X_val)
            fold_losses.append(log_loss(y_val, probs, labels=[0, 1, 2]))
        return float(np.mean(fold_losses)), fold_losses

    # ---- NEW: GRU-specific walk-forward CV ----
    def walk_forward_logloss_gru(self, splits, gru_kwargs=None):
        """
        Walk-forward CV for the GRU. Mirrors `walk_forward_logloss_subset`
        but operates on sequence inputs instead of flat DataFrames.
        """
        if not self.use_gru:
            return float("inf"), []
        gru_kwargs = gru_kwargs or {}
        fold_losses = []
        for train_df, valid_df, _, _ in splits:
            home_tr, away_tr, y_tr = self.build_gru_dataset(train_df)
            home_val, away_val, y_val = self.build_gru_dataset(valid_df)
            sw = self.sample_weights_from_y(y_tr)
            clf = GRUClassifier(
                n_features=len(self.gru_features),
                draw_class_weight=self.draw_class_weight,
                **gru_kwargs,
            )
            clf.fit(home_tr, away_tr, y_tr, sample_weight=sw)
            probs = clf.predict_proba(home_val, away_val)
            fold_losses.append(log_loss(y_val, probs, labels=[0, 1, 2]))
        return float(np.mean(fold_losses)), fold_losses

    def labels_from_proba(self, probs):
        preds = np.argmax(probs, axis=1).astype(int)
        draw_mask = probs[:, 1] >= self.draw_threshold
        preds[draw_mask] = 1
        return preds

    def build_pruning_batches(self, feature_cols):
        batches = {}
        cols = list(feature_cols)
        prefix_groups = {
            "calendar": ["month_sin", "month_cos", "is_festive_period", "days_since_season_start"],
            "h2h": ["h2h_"],
            "nonlinear_matchup": [
                "home_attack_x_away_def_weakness", "away_attack_x_home_def_weakness",
                "abs_npxgd_gap", "min_xg_matchup", "low_total_xg_flag", "draw_balance_signal"
            ],
            "ppda_raw": ["home_ppda_att", "home_ppda_def", "home_ppda_allowed_att", "home_ppda_allowed_def",
                         "away_ppda_att", "away_ppda_def", "away_ppda_allowed_att", "away_ppda_allowed_def"],
            "table_gap_delta": ["delta_games_played_pre", "delta_points_gap_top_pre", "delta_rank_form_5"],
            "consistency_cv_delta": ["delta_cv_xG_5", "delta_cv_xGA_5", "delta_cv_npxGD_5"],
        }

        for name, patterns in prefix_groups.items():
            group = []
            for c in cols:
                for p in patterns:
                    if c == p or c.startswith(p):
                        group.append(c)
                        break
            group = sorted(set(group))
            if group:
                batches[name] = group
        return batches

    def prune_features_in_batches(self, feature_cols, wf_splits, xgb_params, improvement_tol=0.001):
        current = list(feature_cols)
        history = []

        def builder():
            return XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                tree_method="hist",
                **xgb_params
            )

        baseline_ll, _ = self.walk_forward_logloss_subset(builder, wf_splits, feature_subset=current)
        for batch_name, batch_cols in self.build_pruning_batches(current).items():
            trial_cols = [c for c in current if c not in batch_cols]
            if len(trial_cols) == len(current) or not trial_cols:
                continue
            trial_ll, _ = self.walk_forward_logloss_subset(builder, wf_splits, feature_subset=trial_cols)
            improved = trial_ll < (baseline_ll - improvement_tol)
            history.append({
                "batch": batch_name,
                "n_cols": len(batch_cols),
                "baseline_ll": baseline_ll,
                "trial_ll": trial_ll,
                "accepted": improved,
            })
            if improved:
                current = trial_cols
                baseline_ll = trial_ll

        return current, baseline_ll, history

    # ---- NEW: build per-team match-history sequence ----
    def get_sequence(self, team, date, year, seq_len=None):
        """
        Return the team's last `seq_len` matches (raw stats) prior to `date`,
        as a (seq_len, n_features) numpy array. Pads with leading zeros if
        fewer matches exist.
        """
        seq_len = seq_len or self.gru_seq_len
        year = str(year)
        n_features = len(self.gru_features)

        if year not in self.seasons_raw:
            return np.zeros((seq_len, n_features), dtype=np.float32)

        raw_df = self.seasons_raw[year]
        cutoff = pd.to_datetime(date)

        try:
            team_df = raw_df.xs(team, level="team")
        except KeyError:
            return np.zeros((seq_len, n_features), dtype=np.float32)

        team_df = team_df.copy()
        team_df.index = pd.to_datetime(team_df.index)
        team_df = team_df.sort_index()
        past = team_df.loc[team_df.index < cutoff].tail(seq_len)

        # ensure column order
        cols_present = [c for c in self.gru_features if c in past.columns]
        seq = past[cols_present].to_numpy(dtype=np.float32)

        # if some features missing, pad those columns as zeros
        if seq.shape[1] < n_features:
            full = np.zeros((seq.shape[0], n_features), dtype=np.float32)
            for j, c in enumerate(self.gru_features):
                if c in cols_present:
                    full[:, j] = seq[:, cols_present.index(c)]
            seq = full

        # pad rows at the start if fewer matches than seq_len
        if len(seq) < seq_len:
            pad = np.zeros((seq_len - len(seq), n_features), dtype=np.float32)
            seq = np.vstack([pad, seq])

        return seq

    # ---- NEW: turn a match dataframe into GRU inputs ----
    def build_gru_dataset(self, match_df, seq_len=None):
        """
        For each row in the match df, build the home-team and away-team
        sequence tensors and the integer label.
        """
        seq_len = seq_len or self.gru_seq_len
        home_seqs = []
        away_seqs = []
        for _, row in match_df.iterrows():
            year = str(row["season_year"])
            date = row["Date"]
            home_seqs.append(self.get_sequence(row["HomeTeam"], date, year, seq_len=seq_len))
            away_seqs.append(self.get_sequence(row["AwayTeam"], date, year, seq_len=seq_len))

        y = match_df["FTR"].map({-1: 0, 0: 1, 1: 2}).to_numpy(dtype=np.int64)
        return (
            np.array(home_seqs, dtype=np.float32),
            np.array(away_seqs, dtype=np.float32),
            y,
        )

    def train_model(self):
        # building the merged dataframes that we'll use to train
        year_dfs = {year: self.build_match_df(year) for year in self.years}
        latest_df = year_dfs[self.latest_year]

        # set a cutoff date that we can use to split how far into this season
        # we'll train our model on
        cutoff_date = latest_df["Date"].quantile(0.5)

        latest_train = latest_df[latest_df["Date"] <= cutoff_date]
        latest_test = latest_df[latest_df["Date"] > cutoff_date]

        # combine all of our training dataframes
        prior_year_dfs = [year_dfs[y] for y in self.years if y != self.latest_year]
        train_parts = prior_year_dfs + [latest_train]
        train_df = pd.concat(train_parts, ignore_index=True)

        X_train, y_train = self.split_xy(train_df)
        X_test, y_test = self.split_xy(latest_test)
        self.feature_columns = X_train.columns.tolist()
        wf_splits = self.walk_forward_splits(year_dfs)
        if len(wf_splits) < 1:
            raise ValueError("Need at least 3 seasons for walk-forward validation.")

        param_dist = {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.3, 1.0],
            "reg_lambda": [1, 3, 10],
        }
        candidates = list(ParameterSampler(param_dist, n_iter=45, random_state=42))

        best_params = None
        best_cv_ll = float("inf")
        for params in candidates:
            builder = lambda p=params: XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                tree_method="hist",
                **p
            )
            mean_ll, _ = self.walk_forward_logloss(builder, wf_splits)
            if mean_ll < best_cv_ll:
                best_cv_ll = mean_ll
                best_params = params

        pruned_features, pruned_cv_ll, prune_history = self.prune_features_in_batches(
            self.feature_columns, wf_splits, best_params, improvement_tol=0.001
        )
        self.feature_columns = pruned_features
        X_train = X_train[self.feature_columns]
        X_test = X_test[self.feature_columns]

        xgb_model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
            **best_params
        )
        rf_model = RandomForestClassifier(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2
        )
        et_model = ExtraTreesClassifier(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2
        )

        model_builders = {
            "xgb": lambda: XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                tree_method="hist",
                **best_params
            ),
            "rf": lambda: RandomForestClassifier(
                n_estimators=600,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=2
            ),
            "et": lambda: ExtraTreesClassifier(
                n_estimators=600,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=2
            ),
        }

        cv_logloss = {}
        for name, builder in model_builders.items():
            mean_ll, _ = self.walk_forward_logloss_subset(builder, wf_splits, feature_subset=self.feature_columns)
            cv_logloss[name] = mean_ll

        # ---- NEW: GRU walk-forward CV ----
        gru_model = None
        if self.use_gru:
            gru_cv_ll, _ = self.walk_forward_logloss_gru(wf_splits)
            cv_logloss["gru"] = gru_cv_ll

        inv_scores = {name: 1.0 / (score + self.eps) for name, score in cv_logloss.items()}
        inv_total = sum(inv_scores.values())
        ensemble_weights = {name: score / inv_total for name, score in inv_scores.items()}

        self.fit_model_with_weights(xgb_model, X_train, y_train)
        self.fit_model_with_weights(rf_model, X_train, y_train)
        self.fit_model_with_weights(et_model, X_train, y_train)

        # ---- NEW: fit GRU on the full training set ----
        if self.use_gru:
            home_tr_seqs, away_tr_seqs, y_tr_seqs = self.build_gru_dataset(train_df)
            sw_tr = self.sample_weights_from_y(y_tr_seqs)
            gru_model = GRUClassifier(
                n_features=len(self.gru_features),
                draw_class_weight=self.draw_class_weight,
            )
            gru_model.fit(home_tr_seqs, away_tr_seqs, y_tr_seqs, sample_weight=sw_tr)

        probs = (
            ensemble_weights["xgb"] * xgb_model.predict_proba(X_test)
            + ensemble_weights["rf"] * rf_model.predict_proba(X_test)
            + ensemble_weights["et"] * et_model.predict_proba(X_test)
        )
        if self.use_gru and gru_model is not None:
            home_te_seqs, away_te_seqs, _ = self.build_gru_dataset(latest_test)
            probs = probs + ensemble_weights["gru"] * gru_model.predict_proba(home_te_seqs, away_te_seqs)

        preds = self.labels_from_proba(probs)

        print(probs.mean(axis=0))

        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)

        print("Accuracy:", acc)
        print("Log loss:", ll)
        print("Draw class weight:", self.draw_class_weight)
        print("Walk-forward CV log loss:", cv_logloss)
        print("Pruned feature CV log loss (xgb):", pruned_cv_ll)
        print("Feature pruning decisions:", prune_history)
        print("Feature count after pruning:", len(self.feature_columns))
        print("Ensemble weights:", ensemble_weights)
        print("Draw threshold:", self.draw_threshold)
        print("\nClassification report:\n", classification_report(y_test, preds, labels=[0, 1, 2], zero_division=0))
        print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))

        # store stuff on self so you can reuse later
        self.best_model = xgb_model
        ensemble = {"xgb": xgb_model, "rf": rf_model, "et": et_model}
        if self.use_gru and gru_model is not None:
            ensemble["gru"] = gru_model
        self.ensemble_models = ensemble
        self.ensemble_weights = ensemble_weights
        self.best_xgb_params = best_params
        self.feature_prune_history = prune_history
        self.cutoff_date_latest = cutoff_date

        return self.ensemble_models

    def get_stats(self, team, date, year):
        stats_df = self.seasons[year]
        stats_df.index = stats_df.index.set_levels(
            pd.to_datetime(stats_df.index.levels[1]),
            level='date')

        cutoff = pd.to_datetime(date)

        team_df = (
            stats_df
            .xs(team, level="team")
            .sort_index()
        )

        team_stats = team_df.loc[:cutoff]

        if team_stats.empty:
            raise ValueError(f"No stats available for {team} before {cutoff}")

        return team_stats.iloc[-1]

    def predict_game(self, home_team, away_team, date, year, home_bonus=False, rivalry=False):
        if not hasattr(self, "best_model"):
            raise ValueError("Model not trained. Call train_model() first.")

        # get pre-match stats
        home_stats = self.get_stats(home_team, date, year)
        away_stats = self.get_stats(away_team, date, year)

        # build feature row
        row = pd.concat([
            home_stats.add_prefix("home_"),
            away_stats.add_prefix("away_")
        ])

        # drop non-feature columns (safety)
        drop_cols = ["FTR", "Date", "HomeTeam", "AwayTeam", "season_year"]
        row = row.drop([c for c in drop_cols if c in row.index])

        X = row.to_frame().T
        X["Date"] = pd.to_datetime(date).date()
        X["HomeTeam"] = home_team
        X["AwayTeam"] = away_team
        X["season_year"] = str(year)
        X = self.add_matchup_features(X)
        X = self.add_match_context_features(X)
        if hasattr(self, "feature_columns"):
            X = X.reindex(columns=self.feature_columns, fill_value=np.nan)

        # predict
        if hasattr(self, "ensemble_models") and hasattr(self, "ensemble_weights"):
            probs = np.zeros(3, dtype=float)
            for name, model in self.ensemble_models.items():
                if name == "gru":
                    # ---- NEW: GRU branch uses sequence inputs ----
                    home_seq = self.get_sequence(home_team, date, year)[None, :, :]
                    away_seq = self.get_sequence(away_team, date, year)[None, :, :]
                    probs += self.ensemble_weights[name] * model.predict_proba(home_seq, away_seq)[0]
                else:
                    probs += self.ensemble_weights[name] * model.predict_proba(X)[0]
            pred = int(self.labels_from_proba(probs.reshape(1, -1))[0])
        else:
            probs = self.best_model.predict_proba(X)[0]
            pred = int(self.labels_from_proba(probs.reshape(1, -1))[0])

        if home_bonus:
            probs[0] += probs[0] * 0.10
            probs[1] -= probs[1] * 0.03
            probs[2] -= probs[2] * 0.07

        if rivalry:
            probs[1] += probs[1] * 0.05
            probs[0] -= probs[0] * 0.025
            probs[2] -= probs[2] * 0.025

        inv_map = {0: f'{away_team} wins🫣', 1: 'A draw is on the cards😴', 2: f'{home_team} wins🥳'}
        result = inv_map[pred]

        return {
            "prediction": result,
            f"Chance of {home_team} winning": f"{round(float(probs[2]) * 100, 2)}%",
            f"Chance of a draw": f"{round(float(probs[1]) * 100, 2)}%",
            f"Chance of {away_team} winning": f"{round(float(probs[0]) * 100, 2)}%"
        }
