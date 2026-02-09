import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="CoffeeShop SMB Prototype", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def safe_div(n, d):
    return (float(n) / float(d)) if d not in (0, 0.0, None) else np.nan

def fmt_money(x):
    return "—" if pd.isna(x) else f"${x:,.0f}"

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x*100:.1f}%"

def fmt_pct_delta(x):
    return "—" if pd.isna(x) else f"{x*100:.1f}%"

def fmt_pts(x):
    # expects points already (e.g., 1.2 means 1.2 pts)
    return "—" if pd.isna(x) else f"{x:.1f} pts"

def top_attr(df, col, weight_col="sold_units_2025"):
    if df.empty or col not in df.columns or weight_col not in df.columns:
        return "—"
    s = df.groupby(col, dropna=False)[weight_col].sum().sort_values(ascending=False)
    return "—" if s.empty else str(s.index[0])

def multiselect_with_all(label, options, default_all=True, key_prefix="flt"):
    all_key = f"{key_prefix}_{label}_all"
    sel_key = f"{key_prefix}_{label}_sel"

    if all_key not in st.session_state:
        st.session_state[all_key] = default_all

    select_all = st.checkbox(f"Select all {label}", value=st.session_state[all_key], key=all_key)

    if select_all:
        st.session_state[sel_key] = options
        return options

    if sel_key not in st.session_state:
        st.session_state[sel_key] = options if default_all else []

    return st.multiselect(label, options, default=st.session_state[sel_key], key=sel_key)

def _as_records(df_small: pd.DataFrame, n=8):
    if df_small is None or df_small.empty:
        return []
    return df_small.head(n).to_dict(orient="records")

def _filters_summary(sel_product, sel_size, sel_flavor, sel_heat, sel_item):
    return {
        "Product Type": sel_product,
        "Size": sel_size,
        "Flavor": sel_flavor,
        "Heat": sel_heat,
        "Item ID": sel_item,
    }

def _join(items):
    items = [x for x in items if x and str(x).strip() and str(x).strip() != "—"]
    return "—" if not items else ", ".join(items)

# ----------------------------
# Scope payload builder (ONLY AI input)
# ----------------------------
def build_scope_payload(
    filtered: pd.DataFrame,
    view: pd.DataFrame,
    kpis: dict,
    filters_applied: dict,
    n_list: int = 8,
):
    payload = {
        "meta": {
            "prototype": "AI-Powered Merchant Analytics Copilot (CoffeeShop SMB Prototype)",
            "data_note": "Synthetic demo dataset",
            "rows_in_scope": int(len(filtered)),
        },
        "filters_applied": filters_applied,
        "kpis": kpis,
        "top_selling_attributes": {
            "product_type": top_attr(filtered, "product_type"),
            "size": top_attr(filtered, "size"),
            "flavor": top_attr(filtered, "flavor"),
            "heat": top_attr(filtered, "heat"),
        },
    }

    # ---------- item-level summary ----------
    item_dims = ["item_id", "product_type", "size", "flavor", "heat"]
    base_cols = [c for c in item_dims if c in view.columns]

    needed = [
        "revenue_2025", "revenue_2024", "revenue_yoy_pct",
        "gross_profit_2025", "gross_profit_2024",
        "gross_margin_2025",
        "subscription_share_2025",
        "new_customer_rate_2025",
        "sold_units_2025", "sold_units_2024", "units_yoy_pct",
    ]
    cols = base_cols + [c for c in needed if c in view.columns]
    item = view[cols].copy() if cols else view.copy()

    def wavg(s, w):
        wsum = w.sum()
        return np.nan if wsum == 0 else (s * w).sum() / wsum

    if "item_id" in item.columns and base_cols:
        agg_map = {
            "revenue_2025": "sum",
            "revenue_2024": "sum",
            "gross_profit_2025": "sum",
            "gross_profit_2024": "sum",
            "sold_units_2025": "sum",
            "sold_units_2024": "sum",
        }
        g = item.groupby(base_cols, dropna=False).agg({k: v for k, v in agg_map.items() if k in item.columns}).reset_index()

        if "revenue_2025" in g.columns and "revenue_2024" in g.columns:
            g["revenue_yoy_pct"] = np.where(
                g["revenue_2024"] > 0,
                (g["revenue_2025"] - g["revenue_2024"]) / g["revenue_2024"],
                np.nan
            )

        if "gross_profit_2025" in g.columns and "revenue_2025" in g.columns:
            g["gross_margin_2025"] = np.where(g["revenue_2025"] > 0, g["gross_profit_2025"] / g["revenue_2025"], np.nan)

        # weighted rates from original item rows
        if "subscription_share_2025" in item.columns and "revenue_2025" in item.columns:
            tmp = item[base_cols + ["subscription_share_2025", "revenue_2025"]].copy()
            tmp["revenue_2025"] = tmp["revenue_2025"].fillna(0)
            sub = tmp.groupby(base_cols, dropna=False).apply(lambda x: wavg(x["subscription_share_2025"], x["revenue_2025"])).reset_index(name="subscription_share_2025")
            g = g.merge(sub, on=base_cols, how="left")

        if "new_customer_rate_2025" in item.columns and "revenue_2025" in item.columns:
            tmp = item[base_cols + ["new_customer_rate_2025", "revenue_2025"]].copy()
            tmp["revenue_2025"] = tmp["revenue_2025"].fillna(0)
            ncr = tmp.groupby(base_cols, dropna=False).apply(lambda x: wavg(x["new_customer_rate_2025"], x["revenue_2025"])).reset_index(name="new_customer_rate_2025")
            g = g.merge(ncr, on=base_cols, how="left")

        if "units_yoy_pct" in item.columns and "sold_units_2025" in item.columns:
            tmp = item[base_cols + ["units_yoy_pct", "sold_units_2025"]].copy()
            tmp["sold_units_2025"] = tmp["sold_units_2025"].fillna(0)
            uy = tmp.groupby(base_cols, dropna=False).apply(lambda x: wavg(x["units_yoy_pct"], x["sold_units_2025"])).reset_index(name="units_yoy_pct")
            g = g.merge(uy, on=base_cols, how="left")

        item_summary = g
    else:
        item_summary = item.copy()

    def top_list(df_in, sort_col, ascending=False, cols_keep=None):
        if df_in.empty or sort_col not in df_in.columns:
            return []
        cols_keep = cols_keep or [c for c in base_cols if c in df_in.columns] + [sort_col]
        cols_keep = [c for c in cols_keep if c in df_in.columns]
        d = df_in.sort_values(sort_col, ascending=ascending)[cols_keep]
        return _as_records(d, n=n_list)

    payload["ranked_lists"] = {
        "top_products_by_revenue": top_list(
            item_summary, "revenue_2025", ascending=False,
            cols_keep=base_cols + ["revenue_2025", "revenue_yoy_pct", "gross_margin_2025", "subscription_share_2025", "new_customer_rate_2025", "units_yoy_pct"]
        ),
        "top_products_by_revenue_yoy": top_list(
            item_summary, "revenue_yoy_pct", ascending=False,
            cols_keep=base_cols + ["revenue_2025", "revenue_yoy_pct", "gross_margin_2025"]
        ),
        "bottom_products_by_revenue_yoy": top_list(
            item_summary, "revenue_yoy_pct", ascending=True,
            cols_keep=base_cols + ["revenue_2025", "revenue_yoy_pct", "gross_margin_2025"]
        ),
        "top_products_by_gross_margin": top_list(
            item_summary, "gross_margin_2025", ascending=False,
            cols_keep=base_cols + ["revenue_2025", "gross_margin_2025", "revenue_yoy_pct"]
        ),
        "bottom_products_by_gross_margin": top_list(
            item_summary, "gross_margin_2025", ascending=True,
            cols_keep=base_cols + ["revenue_2025", "gross_margin_2025", "revenue_yoy_pct"]
        ),
        "top_products_by_subscription_share": top_list(
            item_summary, "subscription_share_2025", ascending=False,
            cols_keep=base_cols + ["revenue_2025", "subscription_share_2025", "gross_margin_2025"]
        ),
        "top_products_by_new_customer_rate": top_list(
            item_summary, "new_customer_rate_2025", ascending=False,
            cols_keep=base_cols + ["revenue_2025", "new_customer_rate_2025", "gross_margin_2025"]
        ),
    }

    # ---------- attribute breakdowns ----------
    def attr_breakdown(attr):
        if attr not in filtered.columns:
            return []
        g = filtered.groupby(attr, dropna=False).agg(
            revenue_2025=("revenue_2025", "sum"),
            revenue_2024=("revenue_2024", "sum"),
            gross_profit_2025=("gross_profit_2025", "sum"),
            units_2025=("sold_units_2025", "sum"),
            subs_rev_2025=("subscription_revenue_2025", "sum"),
            new_cust_2025=("new_customers_2025", "sum"),
            total_cust_2025=("total_customers_2025", "sum"),
        ).reset_index()

        g["revenue_yoy_pct"] = np.where(g["revenue_2024"] > 0, (g["revenue_2025"] - g["revenue_2024"]) / g["revenue_2024"], np.nan)
        g["gross_margin_2025"] = np.where(g["revenue_2025"] > 0, g["gross_profit_2025"] / g["revenue_2025"], np.nan)
        g["subscription_share_2025"] = np.where(g["revenue_2025"] > 0, g["subs_rev_2025"] / g["revenue_2025"], np.nan)
        g["new_customer_rate_2025"] = np.where(g["total_cust_2025"] > 0, g["new_cust_2025"] / g["total_cust_2025"], np.nan)
        g["revenue_share_2025"] = np.where(g["revenue_2025"].sum() > 0, g["revenue_2025"] / g["revenue_2025"].sum(), np.nan)

        g = g.sort_values("revenue_2025", ascending=False)
        out = g[[attr, "revenue_2025", "revenue_share_2025", "revenue_yoy_pct", "gross_margin_2025", "subscription_share_2025", "new_customer_rate_2025", "units_2025"]]
        return _as_records(out, n=20)

    payload["breakdowns"] = {
        "by_product_type": attr_breakdown("product_type"),
        "by_size": attr_breakdown("size"),
        "by_flavor": attr_breakdown("flavor"),
        "by_heat": attr_breakdown("heat"),
    }

    return payload

def render_structured_response(title: str, content_md: str):
    st.markdown(f"**{title}**")
    st.markdown(content_md)

# ----------------------------
# Mock copilot (owner-friendly, metric-labeled, non-generic)
# ----------------------------
def mock_copilot_response(workflow_name: str, payload: dict, custom_question: str = "") -> str:
    k = payload.get("kpis", {})
    rows = payload.get("meta", {}).get("rows_in_scope", 0)
    top_mix = payload.get("top_selling_attributes", {})

    rev = k.get("rev_25", np.nan)
    rev_yoy = k.get("rev_yoy", np.nan)
    units = k.get("units_25", np.nan)
    units_yoy = k.get("units_yoy", np.nan)

    gm = k.get("gm_25", np.nan)
    gm_delta_pts = k.get("gm_delta_pts", np.nan)
    sub_share = k.get("sub_share_25", np.nan)
    sub_delta_pts = k.get("sub_share_delta_pts", np.nan)
    new_rate = k.get("new_rate_25", np.nan)
    new_delta_pts = k.get("new_rate_delta_pts", np.nan)

    rl = payload.get("ranked_lists", {})
    bd = payload.get("breakdowns", {})

    def mix_label(r):
        parts = []
        for key in ["product_type", "size", "flavor", "heat"]:
            v = r.get(key, "—")
            if v and str(v).strip() and str(v).strip() != "—":
                parts.append(str(v))
        return " · ".join(parts) if parts else (r.get("item_id", "—") or "—")

    # UPDATED: explicit metric names (no GM/Sub/New abbreviations)
    def format_item_line(r, include_margin=False, include_sub=False, include_new=False):
        label = mix_label(r)
        parts = [f"**{label}**"]

        if "revenue_2025" in r:
            parts.append(f"Revenue (2025): {fmt_money(r.get('revenue_2025', np.nan))}")

        if "revenue_yoy_pct" in r:
            parts.append(f"Revenue YoY %: {fmt_pct_delta(r.get('revenue_yoy_pct', np.nan))}")

        if include_margin and "gross_margin_2025" in r:
            parts.append(f"Gross Margin % (2025): {fmt_pct(r.get('gross_margin_2025', np.nan))}")

        if include_sub and "subscription_share_2025" in r:
            parts.append(f"Subscription Revenue % (2025): {fmt_pct(r.get('subscription_share_2025', np.nan))}")

        if include_new and "new_customer_rate_2025" in r:
            parts.append(f"New Customer % (2025): {fmt_pct(r.get('new_customer_rate_2025', np.nan))}")

        return " · ".join(parts)

    def top_by_yoy(breakdown_records, dim_name, n=3):
        if not breakdown_records:
            return []
        # sort by revenue_yoy_pct desc, but require non-trivial revenue share
        recs = [r for r in breakdown_records if not pd.isna(r.get("revenue_yoy_pct", np.nan))]
        recs = sorted(recs, key=lambda r: (r.get("revenue_yoy_pct", -999), r.get("revenue_2025", 0)), reverse=True)
        out = []
        for r in recs:
            share = r.get("revenue_share_2025", np.nan)
            # keep small slices from taking over (still allow if we don't have many)
            if (pd.isna(share) or share >= 0.05) or len(out) < 1:
                out.append(r)
            if len(out) >= n:
                break
        return out

    def bottom_by_yoy(breakdown_records, dim_name, n=3):
        if not breakdown_records:
            return []
        recs = [r for r in breakdown_records if not pd.isna(r.get("revenue_yoy_pct", np.nan))]
        recs = sorted(recs, key=lambda r: (r.get("revenue_yoy_pct", 999), r.get("revenue_2025", 0)))
        out = []
        for r in recs:
            share = r.get("revenue_share_2025", np.nan)
            if (pd.isna(share) or share >= 0.05) or len(out) < 1:
                out.append(r)
            if len(out) >= n:
                break
        return out

    def theme_line(dim_label, r):
        return (
            f"- **{dim_label}: {r.get(dim_label.lower().replace(' ', '_'), r.get(dim_label, '—'))}** "
            f"(Revenue share: {fmt_pct(r.get('revenue_share_2025', np.nan))} · "
            f"Revenue YoY: {fmt_pct_delta(r.get('revenue_yoy_pct', np.nan))} · "
            f"GM: {fmt_pct(r.get('gross_margin_2025', np.nan))})."
        )

    md = []

    # --------------------------------
    # Shared snapshot (always labeled)
    # --------------------------------
    md.append("### Snapshot (current filters)")
    md.append(f"- Rows in scope: **{rows:,}**")
    md.append(f"- 2025 Revenue: **{fmt_money(rev)}** (Revenue YoY: **{fmt_pct_delta(rev_yoy)}**)")
    md.append(f"- Units: **{int(units):,}** (Units YoY: **{fmt_pct_delta(units_yoy)}**)" if not pd.isna(units) else "- Units: —")
    md.append(f"- Gross Margin % (2025): **{fmt_pct(gm)}** (YoY Δ: **{fmt_pts(gm_delta_pts)}**)")
    md.append(f"- Subscription Revenue % (2025): **{fmt_pct(sub_share)}** (YoY Δ: **{fmt_pts(sub_delta_pts)}**)")
    md.append(f"- New Customer % (2025): **{fmt_pct(new_rate)}** (YoY Δ: **{fmt_pts(new_delta_pts)}**)")

    scope_fingerprint = f"{top_mix.get('product_type','—')} · {top_mix.get('size','—')} · {top_mix.get('flavor','—')} · {top_mix.get('heat','—')}"
    md.append(f"- Scope fingerprint: **{scope_fingerprint}** (top-selling mix in this slice)")
    md.append("")

    # --------------------------------
    # Workflow 2: What Changed This Year
    # --------------------------------
    if workflow_name == "What Changed This Year":
        md.append("### What changed this year (so you know where to focus next year)")
        md.append("This section summarizes **year-over-year change in revenue** by category, then translates it into clear actions.")
        md.append("")

        # Trend themes from breakdowns
        heat_bd = bd.get("by_heat", [])
        flavor_bd = bd.get("by_flavor", [])
        size_bd = bd.get("by_size", [])
        type_bd = bd.get("by_product_type", [])

        # winners/losers by category (not item spam)
        md.append("#### 1) Category shifts (Revenue YoY by category)")
        for label, records in [
            ("Drink type", type_bd),
            ("Size", size_bd),
            ("Flavor", flavor_bd),
            ("Hot vs Iced", heat_bd),
        ]:
            top3 = top_by_yoy(records, label, n=2)
            bot2 = bottom_by_yoy(records, label, n=1)
            if not records:
                continue

            def cat_name(r):
                # record contains the attribute value under its real key
                # try common keys
                for key in ["product_type", "size", "flavor", "heat"]:
                    if key in r:
                        return str(r.get(key))
                return "—"

            winners = [f"{cat_name(r)} ({fmt_pct_delta(r.get('revenue_yoy_pct', np.nan))} YoY · share {fmt_pct(r.get('revenue_share_2025', np.nan))})" for r in top3]
            losers = [f"{cat_name(r)} ({fmt_pct_delta(r.get('revenue_yoy_pct', np.nan))} YoY · share {fmt_pct(r.get('revenue_share_2025', np.nan))})" for r in bot2]

            md.append(f"- **{label} winners (Revenue YoY):** {_join(winners)}")
            md.append(f"  - **{label} laggards (Revenue YoY):** {_join(losers)}")

        md.append("")
        md.append("#### 2) What it means (owner decisions you can make)")
        # Convert to concrete decisions
        # Heat decision:
        if heat_bd:
            top_heat = top_by_yoy(heat_bd, "Hot vs Iced", n=2)
            if len(top_heat) >= 1:
                best = top_heat[0]
                # identify key name:
                heat_name = best.get("heat", "—")
                md.append(
                    f"- **Menu emphasis:** {heat_name.title()} drinks are one of the faster movers in this slice "
                    f"(Revenue YoY: {fmt_pct_delta(best.get('revenue_yoy_pct', np.nan))}, share: {fmt_pct(best.get('revenue_share_2025', np.nan))}). "
                    "Next-year move: give this format a **permanent menu position** and rotate seasonal variants *inside the winner*."
                )

        # Flavor decision:
        if flavor_bd:
            top_flavors = top_by_yoy(flavor_bd, "Flavor", n=2)
            if top_flavors:
                fl_names = [r.get("flavor", "—") for r in top_flavors]
                md.append(
                    f"- **Flavor direction:** fastest growing flavors by **Revenue YoY** are: {_join(fl_names)}. "
                    "Next-year move: promote these as **paid add-ons** (not discounts) to increase average ticket."
                )

        # Size decision:
        if size_bd:
            top_sizes = top_by_yoy(size_bd, "Size", n=2)
            if top_sizes:
                sz_names = [r.get("size", "—") for r in top_sizes]
                md.append(
                    f"- **Size laddering:** growth is strongest in: {_join(sz_names)} (Revenue YoY by size). "
                    "Next-year move: train one consistent upsell line (e.g., “Want to size up for $X?”) and track size mix monthly."
                )

        md.append("")
        md.append("#### 3) Item-level winners/losers (clearly labeled)")
        winners = rl.get("top_products_by_revenue_yoy", [])[:5]
        losers = rl.get("bottom_products_by_revenue_yoy", [])[:5]
        if winners:
            md.append("**Top 5 items by Revenue YoY %:**")
            for r in winners:
                md.append(f"- {format_item_line(r, include_margin=True)}")
        if losers:
            md.append("")
            md.append("**Bottom 5 items by Revenue YoY %:**")
            for r in losers:
                md.append(f"- {format_item_line(r, include_margin=True)}")

        md.append("")
        md.append("#### 4) Next-year focus checklist")
        md.append("- Keep **winners** permanently visible (menu board position, default recommendations).")
        md.append("- Use **seasonal specials** as twists on winners (adds novelty without operational chaos).")
        md.append("- Don’t promo laggards until you understand *why* they’re down (stockouts, taste feedback, pricing, visibility).")

        return "\n".join(md)

    # --------------------------------
    # Workflow 3: Profit Plan
    # --------------------------------
    if workflow_name == "Profit Plan":
        md.append("### Profit plan (fix margin leaks + grow revenue without discounting)")
        md.append("This section is a **12-month plan**: what to fix first, what to test, and how to tell if it worked.")
        md.append("")

        # Build a margin leak list: low margin + meaningful revenue
        low_margin = rl.get("bottom_products_by_gross_margin", [])
        # revenue threshold (avoid tiny items)
        revs = [r.get("revenue_2025", 0) for r in low_margin if r.get("revenue_2025", 0) is not None]
        rev_thresh = np.percentile(revs, 50) if len(revs) >= 5 else (np.mean(revs) if revs else 0)

        leaks = []
        for r in low_margin:
            rrev = r.get("revenue_2025", 0) or 0
            if rrev >= rev_thresh:
                leaks.append(r)
            if len(leaks) >= 5:
                break

        md.append("#### 1) Where profit is leaking (fix these before you promote them)")
        if leaks:
            for r in leaks:
                md.append(f"- {format_item_line(r, include_margin=True)}")
        else:
            md.append("- —")

        md.append("")
        md.append("#### 2) Concrete profit levers (what to do next year)")

        # Price test candidates: pick 2 items from leaks (or fallback to top revenue items with lower margin)
        price_candidates = leaks[:2] if leaks else rl.get("top_products_by_revenue", [])[:2]

        md.append("**A) Price tests (small, safe, measurable)**")
        if price_candidates:
            for r in price_candidates:
                label = mix_label(r)
                md.append(
                    f"- Test item: **{label}** → run a **+1% to +3%** price test.\n"
                    f"  - Why: it has meaningful revenue in this slice (Rev: {fmt_money(r.get('revenue_2025', np.nan))}).\n"
                    "  - Pass criteria (monthly): revenue stays flat or up, and gross margin improves **≥ +0.5 pts**.\n"
                    "  - Stop rule: units drop sharply and revenue declines."
                )
        else:
            md.append("- —")

        md.append("")
        md.append("**B) Grow revenue without discounting (increase average ticket)**")

        # Use flavor breakdown to recommend add-on direction
        flavor_bd = bd.get("by_flavor", [])
        top_flavors = top_by_yoy(flavor_bd, "Flavor", n=3)
        fl_names = [r.get("flavor", "—") for r in top_flavors if r.get("flavor") is not None]

        md.append(
            "- **Paid add-ons (flavor shots / syrups):** if flavored demand is growing, charge for customization instead of discounting drinks.\n"
            f"  - Data signal: fastest-growing flavors by **Revenue YoY** in this slice: {_join(fl_names)}.\n"
            "  - Test: add a clearly priced add-on button; measure average ticket and gross margin."
        )

        # Size laddering from size breakdown
        size_bd = bd.get("by_size", [])
        top_sizes = top_by_yoy(size_bd, "Size", n=2)
        sz_names = [r.get("size", "—") for r in top_sizes if r.get("size") is not None]
        md.append(
            "- **Size laddering:** train one consistent upsell script.\n"
            f"  - Data signal: growth is strongest in sizes: {_join(sz_names)} (Revenue YoY).\n"
            "  - Test: track size mix monthly; target a +2–4 pt increase in the higher size mix without margin loss."
        )

        md.append("")
        md.append("**C) Subscription growth (increase recurring revenue without killing margin)**")
        sub_list = rl.get("top_products_by_subscription_share", [])
        sub_anchor = sub_list[0] if sub_list else None
        if sub_anchor:
            md.append(
                f"- Best subscription anchor (in this slice): {format_item_line(sub_anchor, include_margin=True, include_sub=True)}\n"
                "  - Offer design: use **perks**, not deep discounts (e.g., 1 free add-on per week, priority upgrade, member-only seasonal).\n"
                "  - Pass criteria (quarterly): subscription share improves **+1 to +2 pts** while gross margin stays within **-0.5 pts**."
            )
        else:
            md.append("- —")

        md.append("")
        md.append("#### 3) What to feature vs. what to stop pushing (rules)")
        md.append("- **Feature** items that are: high revenue AND healthy margin (use as your “always visible” anchors).")
        md.append("- **Do not heavily promote** low-margin items until you fix margin (price / cost / recipe / portioning).")
        md.append("- Use discounting only for **new-customer acquisition**, not for everyone (limits cannibalization).")

        md.append("")
        md.append("#### 4) Next data to add (so this becomes even more specific)")
        md.append("- Average ticket / orders (so we can quantify add-on lift)")
        md.append("- Promo history (so we can recommend exact discount depth that actually moved units)")
        md.append("- COGS detail (so we can identify which ingredients are driving margin drops)")

        return "\n".join(md)

    # --------------------------------
    # Workflow 1: Next-Year Growth Plan (selection/expansion + why)
    # --------------------------------
    if workflow_name == "Next-Year Growth Plan":
        md.append("### Next-year growth plan (what to do + why it grows sales)")
        md.append("This is a **12-month, owner-friendly plan** based on what’s already working in this filtered slice.")
        md.append("")

        # 1) What customers are telling you
        md.append("#### 1) What customers are telling you (data signals)")
        md.append(f"- Your core mix (top-selling): **{scope_fingerprint}**.")
        md.append(f"- Revenue is up **{fmt_pct_delta(rev_yoy)} YoY** and units are up **{fmt_pct_delta(units_yoy)} YoY** → growth is real demand, not only pricing.")
        md.append(f"- Margin is **{fmt_pts(gm_delta_pts)}** vs last year → watch promo / cost creep.")
        md.append(f"- Subscription is **{fmt_pts(sub_delta_pts)}** and new-customer rate is **{fmt_pts(new_delta_pts)}** → you can grow both if you make the right offers.")
        md.append("")

        # 2) Specific actions
        md.append("#### 2) Specific actions (simple, concrete, testable)")

        # Acquisition anchor: highest new customer rate with decent revenue
        new_list = rl.get("top_products_by_new_customer_rate", [])
        acq = None
        for r in new_list:
            if (r.get("revenue_2025", 0) or 0) > 0:
                acq = r
                break

        if acq:
            md.append("**A) Bring in new customers (acquisition offer that avoids cannibalization)**")
            md.append(
                f"- Use this as your “first-time favorite”: {format_item_line(acq, include_margin=True, include_new=True)}\n"
                "  - Offer: **free add-on** OR **5–10% off** for **first-time buyers only**.\n"
                "  - Why it grows topline: you’re converting new visitors, not discounting regulars.\n"
                "  - Pass criteria (monthly): new-customer rate improves **≥ +2 pts** AND revenue doesn’t drop."
            )
        else:
            md.append("**A) Bring in new customers:** —")

        md.append("")

        # Subscription anchor
        sub_list = rl.get("top_products_by_subscription_share", [])
        sub_anchor = sub_list[0] if sub_list else None
        if sub_anchor:
            md.append("**B) Grow subscriptions (repeat revenue without deep discounts)**")
            md.append(
                f"- Anchor item: {format_item_line(sub_anchor, include_margin=True, include_sub=True)}\n"
                "  - Program: **Monthly Classics** with perks (e.g., 1 free add-on/week, member-only seasonal, priority size upgrade).\n"
                "  - Why it grows bottom line: perks are cheaper than discounts, and they lock in repeat behavior.\n"
                "  - Target: subscription share **+1 to +2 pts** over the next 2 quarters."
            )
        else:
            md.append("**B) Grow subscriptions:** —")

        md.append("")

        # Upsell/add-ons based on flavors + sizes
        flavor_bd = bd.get("by_flavor", [])
        top_flavors = top_by_yoy(flavor_bd, "Flavor", n=3)
        fl_names = [r.get("flavor", "—") for r in top_flavors if r.get("flavor") is not None]

        md.append("**C) Grow revenue per customer (no discounts required)**")
        md.append(
            f"- Add-on direction: fastest-growing flavors by **Revenue YoY** are {_join(fl_names)} → sell them as paid add-ons.\n"
            "  - Why it grows topline: add-ons increase ticket size without needing new foot traffic.\n"
            "  - Test: track average ticket and margin; aim for +$0.25–$0.75 per order equivalent (directional)."
        )

        size_bd = bd.get("by_size", [])
        top_sizes = top_by_yoy(size_bd, "Size", n=2)
        sz_names = [r.get("size", "—") for r in top_sizes if r.get("size") is not None]
        md.append(
            f"- Size laddering: growth is strongest in sizes {_join(sz_names)} → add a consistent upsell script.\n"
            "  - Why it grows bottom line: size upgrades usually improve profit per order.\n"
            "  - Test: target +2–4 pts shift toward the higher size mix over 90 days."
        )

        md.append("")
        md.append("#### 3) Keep it grounded (decision rules)")
        md.append("- Don’t create 20 new menu items. Rotate **seasonal twists on winners** (3–6 per quarter max).")
        md.append("- Only discount for **new customers** (one redemption/customer).")
        md.append("- Fix low-margin items before you feature them heavily.")

        if custom_question.strip():
            md.append("")
            md.append(f"---\n**Custom question noted:** {custom_question.strip()}")

        return "\n".join(md)

    # --------------------------------
    # Custom question fallback
    # --------------------------------
    md.append("### Custom question")
    md.append("I can answer custom questions best when they map to one of these owner goals:")
    md.append("- bring in more new customers")
    md.append("- improve margin without losing units")
    md.append("- grow subscriptions / repeat behavior")
    md.append("")
    if custom_question.strip():
        md.append(f"**Your question:** {custom_question.strip()}")
        md.append("")
        md.append("Based on your current slice, I’d start with the Next-Year Growth Plan workflow and tailor the actions to your question.")
    else:
        md.append("Type a specific question and click Run / Analyze.")

    return "\n".join(md)

# ----------------------------
# Real LLM integration stub (Option A)
# ----------------------------
def call_llm_or_mock(workflow_name: str, payload: dict, custom_question: str = "") -> str:
    mode = os.getenv("COPILOT_MODE", "mock").lower()
    if mode != "real":
        return mock_copilot_response(workflow_name, payload, custom_question)

    # REAL MODE placeholder:
    # - send ONLY payload + workflow_name + custom_question to your hosted LLM endpoint
    # - return markdown
    return mock_copilot_response(workflow_name, payload, custom_question)

# ----------------------------
# Session state
# ----------------------------
if "copilot_messages" not in st.session_state:
    st.session_state.copilot_messages = []
if "copilot_running" not in st.session_state:
    st.session_state.copilot_running = False

def add_message(role: str, title: str, content: str):
    st.session_state.copilot_messages.append({
        "role": role,
        "title": title,
        "content": content,
        "ts": time.time(),
    })

def clear_conversation():
    st.session_state.copilot_messages = []
    st.session_state.copilot_running = False

# ----------------------------
# Load data
# ----------------------------
CSV_PATH = "coffee_shop_product_level_2025_vs_2024.csv"
df = pd.read_csv(CSV_PATH)

num_cols = [
    "revenue_2025","revenue_2024",
    "subscription_revenue_2025","subscription_revenue_2024",
    "nonsubscription_revenue_2025","nonsubscription_revenue_2024",
    "sold_units_2025","sold_units_2024",
    "new_customers_2025","new_customers_2024",
    "total_customers_2025","total_customers_2024",
    "cost_per_unit_2025","cost_per_unit_2024",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Enforce revenue reconciliation
df["nonsubscription_revenue_2025"] = (df["revenue_2025"] - df["subscription_revenue_2025"]).clip(lower=0)
df["nonsubscription_revenue_2024"] = (df["revenue_2024"] - df["subscription_revenue_2024"]).clip(lower=0)

# Cost / profit / margins
df["total_cost_2025"] = df["cost_per_unit_2025"] * df["sold_units_2025"]
df["total_cost_2024"] = df["cost_per_unit_2024"] * df["sold_units_2024"]

df["gross_profit_2025"] = df["revenue_2025"] - df["total_cost_2025"]
df["gross_profit_2024"] = df["revenue_2024"] - df["total_cost_2024"]

df["gross_margin_2025"] = np.where(df["revenue_2025"] > 0, df["gross_profit_2025"] / df["revenue_2025"], np.nan)
df["gross_margin_2024"] = np.where(df["revenue_2024"] > 0, df["gross_profit_2024"] / df["revenue_2024"], np.nan)

df["new_customer_rate_2025"] = np.where(df["total_customers_2025"] > 0, df["new_customers_2025"] / df["total_customers_2025"], np.nan)
df["new_customer_rate_2024"] = np.where(df["total_customers_2024"] > 0, df["new_customers_2024"] / df["total_customers_2024"], np.nan)

df["subscription_share_2025"] = np.where(df["revenue_2025"] > 0, df["subscription_revenue_2025"] / df["revenue_2025"], np.nan)
df["subscription_share_2024"] = np.where(df["revenue_2024"] > 0, df["subscription_revenue_2024"] / df["revenue_2024"], np.nan)

# ----------------------------
# Filters (top)
# ----------------------------
f1, f2, f3, f4, f5 = st.columns([1.25, 1.0, 1.25, 0.9, 1.4])

with f1:
    product_types = sorted(df["product_type"].dropna().unique().tolist())
    sel_product = multiselect_with_all("Product Type", product_types, default_all=True)

with f2:
    sizes = sorted(df["size"].dropna().unique().tolist())
    sel_size = multiselect_with_all("Size", sizes, default_all=True)

with f3:
    flavors = sorted(df["flavor"].dropna().unique().tolist())
    sel_flavor = multiselect_with_all("Flavor", flavors, default_all=True)

with f4:
    heats = sorted(df["heat"].dropna().unique().tolist())
    sel_heat = multiselect_with_all("Heat", heats, default_all=True)

with f5:
    item_ids = sorted(df["item_id"].dropna().unique().tolist())
    sel_item = multiselect_with_all("Item ID", item_ids, default_all=True)

filtered = df[
    df["product_type"].isin(sel_product)
    & df["size"].isin(sel_size)
    & df["flavor"].isin(sel_flavor)
    & df["heat"].isin(sel_heat)
    & df["item_id"].isin(sel_item)
].copy()

# ----------------------------
# Title
# ----------------------------
st.markdown("## AI-Powered Merchant Analytics Copilot (Product Prototype) for CoffeeShop SMB")

# ----------------------------
# KPI calculations (aggregated)
# ----------------------------
rev_25 = filtered["revenue_2025"].sum()
rev_24 = filtered["revenue_2024"].sum()
rev_yoy = safe_div(rev_25 - rev_24, rev_24) if rev_24 > 0 else np.nan

gp_25 = filtered["gross_profit_2025"].sum()
gp_24 = filtered["gross_profit_2024"].sum()
gp_yoy = safe_div(gp_25 - gp_24, gp_24) if gp_24 != 0 else np.nan

new_rate_25 = safe_div(filtered["new_customers_2025"].sum(), filtered["total_customers_2025"].sum())
new_rate_24 = safe_div(filtered["new_customers_2024"].sum(), filtered["total_customers_2024"].sum())
new_rate_delta_pts = (new_rate_25 - new_rate_24) * 100

sub_share_25 = safe_div(filtered["subscription_revenue_2025"].sum(), filtered["revenue_2025"].sum())
sub_share_24 = safe_div(filtered["subscription_revenue_2024"].sum(), filtered["revenue_2024"].sum())
sub_share_delta_pts = (sub_share_25 - sub_share_24) * 100

gm_25 = safe_div(gp_25, rev_25)
gm_24 = safe_div(gp_24, rev_24)
gm_delta_pts = (gm_25 - gm_24) * 100

units_25 = filtered["sold_units_2025"].sum()
units_24 = filtered["sold_units_2024"].sum()
units_yoy = safe_div(units_25 - units_24, units_24) if units_24 > 0 else np.nan

top_pt = top_attr(filtered, "product_type")
top_sz = top_attr(filtered, "size")
top_fl = top_attr(filtered, "flavor")
top_ht = top_attr(filtered, "heat")

# ----------------------------
# KPI Scorecards
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue (2025)", fmt_money(rev_25), fmt_pct_delta(rev_yoy))
k2.metric("Total Gross Profit (2025)", fmt_money(gp_25), fmt_pct_delta(gp_yoy))
k3.metric("New Customer % (2025)", fmt_pct(new_rate_25), fmt_pts(new_rate_delta_pts))
k4.metric("Subscription Revenue % (2025)", fmt_pct(sub_share_25), fmt_pts(sub_share_delta_pts))

k5, k6, k7, k8 = st.columns(4)
k5.metric("Gross Margin % (2025)", fmt_pct(gm_25), fmt_pts(gm_delta_pts))
k6.metric("Top Product Type", top_pt)
k7.metric("Top Size", top_sz)
k8.metric("Top Flavor", top_fl)

k9, k10, k11, k12 = st.columns(4)
k9.metric("Top Heat", top_ht)
k10.metric("Rows in Scope", f"{len(filtered):,}")
k11.metric("Units Sold (2025)", f"{int(units_25):,}", fmt_pct_delta(units_yoy))
k12.metric(" ", " ")

st.divider()

# ----------------------------
# Build Table View (numeric first; format at render time)
# ----------------------------
view = filtered.copy()

view["revenue_yoy_pct"] = np.where(
    view["revenue_2024"] > 0,
    (view["revenue_2025"] - view["revenue_2024"]) / view["revenue_2024"],
    np.nan
)
view["units_yoy_pct"] = np.where(
    view["sold_units_2024"] > 0,
    (view["sold_units_2025"] - view["sold_units_2024"]) / view["sold_units_2024"],
    np.nan
)

view["new_customer_rate_delta_pts"] = (view["new_customer_rate_2025"] - view["new_customer_rate_2024"]) * 100
view["subscription_share_delta_pts"] = (view["subscription_share_2025"] - view["subscription_share_2024"]) * 100
view["gross_margin_delta_pts"] = (view["gross_margin_2025"] - view["gross_margin_2024"]) * 100

# ----------------------------
# Two-column layout: Table (left) + AI Copilot (right)
# ----------------------------
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.subheader("Product Performance (Filtered)")

    sort_metric = st.selectbox(
        "Sort by",
        [
            "revenue_2025", "revenue_yoy_pct",
            "new_customer_rate_2025", "new_customer_rate_delta_pts",
            "subscription_share_2025", "subscription_share_delta_pts",
            "gross_margin_2025", "gross_margin_delta_pts",
            "sold_units_2025", "units_yoy_pct",
        ],
        index=0
    )
    view_sorted = view.sort_values(sort_metric, ascending=False)

    display_cols = [
        "item_id", "product_type", "size", "flavor", "heat",
        "revenue_2025", "revenue_yoy_pct",
        "new_customer_rate_2025", "new_customer_rate_delta_pts",
        "subscription_share_2025", "subscription_share_delta_pts",
        "gross_margin_2025", "gross_margin_delta_pts",
        "sold_units_2025", "units_yoy_pct",
    ]
    display_cols = [c for c in display_cols if c in view_sorted.columns]
    table_num = view_sorted[display_cols].copy()

    rename_map = {
        "item_id": "Item ID",
        "product_type": "Product Type",
        "size": "Size",
        "flavor": "Flavor",
        "heat": "Heat",
        "revenue_2025": "Revenue (2025)",
        "revenue_yoy_pct": "Revenue YoY %",
        "new_customer_rate_2025": "New Customer % (2025)",
        "new_customer_rate_delta_pts": "New Customer YoY (pts)",
        "subscription_share_2025": "Subscription Revenue % (2025)",
        "subscription_share_delta_pts": "Subscription YoY (pts)",
        "gross_margin_2025": "Gross Margin % (2025)",
        "gross_margin_delta_pts": "Gross Margin YoY (pts)",
        "sold_units_2025": "Units Sold (2025)",
        "units_yoy_pct": "Units YoY %",
    }
    table_num = table_num.rename(columns=rename_map)

    st.download_button(
        "Download filtered data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_scope.csv",
        mime="text/csv",
        use_container_width=True,
    )

    money_cols = ["Revenue (2025)"]
    pct_cols = [
        "Revenue YoY %",
        "New Customer % (2025)",
        "Subscription Revenue % (2025)",
        "Gross Margin % (2025)",
        "Units YoY %",
    ]
    pts_cols = [
        "New Customer YoY (pts)",
        "Subscription YoY (pts)",
        "Gross Margin YoY (pts)",
    ]
    int_cols = ["Units Sold (2025)"]

    fmt = {}
    for c in money_cols:
        if c in table_num.columns:
            fmt[c] = "${:,.0f}"
    for c in pct_cols:
        if c in table_num.columns:
            fmt[c] = "{:.1%}"
    for c in pts_cols:
        if c in table_num.columns:
            fmt[c] = "{:.1f}"
    for c in int_cols:
        if c in table_num.columns:
            fmt[c] = "{:,.0f}"

    styled_table = table_num.style.format(fmt, na_rep="—")
    st.dataframe(styled_table, use_container_width=True, height=560)

    # ----------------------------
# Full-width product context (below dashboard)
# ----------------------------
    st.divider()

    st.subheader("What this is and why it matters")

    st.markdown("""
    This application is a product analytics copilot prototype designed for small and medium sized businesses.
    It demonstrates how product level sales and customer data can be translated into clear, actionable business insights.

    ### The customer problem
    Small business owners are busy running day to day operations and often do not have time to analyze spreadsheets or interpret complex dashboards.
    Common questions go unanswered, such as:
    - Which products are actually driving revenue growth
    - Where margin is being lost
    - Which items should be promoted, bundled, or deprioritized
    - How subscriptions and repeat customers impact performance

    ### What this prototype delivers
    This prototype is designed to reduce the effort required to turn data into decisions.
    It does this by:
    - Highlighting the most impactful products and trends in the current filtered scope
    - Surfacing risks and opportunities related to growth and margin
    - Translating metrics into simple recommendations that an owner can act on

    ### Industry context
    Modern commerce platforms such as Square capture rich data across transactions, products, customers, and subscriptions.
    This prototype illustrates how that data can be operationalized into decision support experiences that help businesses succeed.

    ### Important note
    This is not an official Square product and does not integrate with Square systems.
    All data shown is sample data for demonstration and portfolio purposes.
    """)


with right:
    st.subheader("AI Copilot")
    st.caption("This copilot analyzes the **current filtered scope** of the dashboard. Pick a workflow or ask a custom question.")

    # Conversation controls + trace toggle
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear conversation", use_container_width=True):
            clear_conversation()
    with c2:
        show_trace = st.toggle("Show payload", value=False)

    # Build payload fresh every rerun (so it changes with filters)
    filters_applied = _filters_summary(sel_product, sel_size, sel_flavor, sel_heat, sel_item)
    kpis = {
        "rev_25": float(rev_25),
        "rev_24": float(rev_24),
        "rev_yoy": float(rev_yoy) if not pd.isna(rev_yoy) else np.nan,
        "gp_25": float(gp_25),
        "gp_24": float(gp_24),
        "gp_yoy": float(gp_yoy) if not pd.isna(gp_yoy) else np.nan,
        "gm_25": float(gm_25) if not pd.isna(gm_25) else np.nan,
        "gm_24": float(gm_24) if not pd.isna(gm_24) else np.nan,
        "gm_delta_pts": float(gm_delta_pts) if not pd.isna(gm_delta_pts) else np.nan,
        "new_rate_25": float(new_rate_25) if not pd.isna(new_rate_25) else np.nan,
        "new_rate_24": float(new_rate_24) if not pd.isna(new_rate_24) else np.nan,
        "new_rate_delta_pts": float(new_rate_delta_pts) if not pd.isna(new_rate_delta_pts) else np.nan,
        "sub_share_25": float(sub_share_25) if not pd.isna(sub_share_25) else np.nan,
        "sub_share_24": float(sub_share_24) if not pd.isna(sub_share_24) else np.nan,
        "sub_share_delta_pts": float(sub_share_delta_pts) if not pd.isna(sub_share_delta_pts) else np.nan,
        "units_25": float(units_25),
        "units_24": float(units_24),
        "units_yoy": float(units_yoy) if not pd.isna(units_yoy) else np.nan,
        "top_product_type": top_pt,
        "top_size": top_sz,
        "top_flavor": top_fl,
        "top_heat": top_ht,
    }

    payload = build_scope_payload(
        filtered=filtered,
        view=view,
        kpis=kpis,
        filters_applied=filters_applied,
        n_list=8,
    )

    st.markdown("**Workflows (yearly planning)**")
    b1, b2, b3 = st.columns(3)

    def run_workflow(workflow_name: str, custom_question: str = ""):
        if st.session_state.copilot_running:
            return
        st.session_state.copilot_running = True

        try:
            # Keep the UI clean: don't print huge JSON into the chat.
            add_message("user", "You", f"Workflow: **{workflow_name}**\n\nRows in scope: **{len(filtered):,}**")

            with st.spinner("Analyzing current scope..."):
                response_md = call_llm_or_mock(workflow_name, payload, custom_question)

            add_message("assistant", "Copilot response", response_md)

        except Exception as e:
            add_message("assistant", "Copilot error", f"Something went wrong while generating the response:\n\n`{e}`")

        finally:
            # Critical: never get stuck disabled after one run
            st.session_state.copilot_running = False

    with b1:
        if st.button("Next-Year Growth Plan", use_container_width=True, disabled=st.session_state.copilot_running):
            run_workflow("Next-Year Growth Plan")
    with b2:
        if st.button("What Changed This Year", use_container_width=True, disabled=st.session_state.copilot_running):
            run_workflow("What Changed This Year")
    with b3:
        if st.button("Profit Plan", use_container_width=True, disabled=st.session_state.copilot_running):
            run_workflow("Profit Plan")


    st.caption("Demo prototype using synthetic data. Recommendations are directional.")

    if show_trace:
        st.code(json.dumps(payload, indent=2), language="json")

    st.markdown("---")
    st.markdown("**Copilot output**")

    out = st.container(height=560, border=True)
    with out:
        if not st.session_state.copilot_messages:
            st.info("Run a workflow to see recommendations for the current filtered scope.")
        else:
            for m in st.session_state.copilot_messages:
                if m["role"] == "user":
                    st.markdown(m["content"])
                    st.markdown("---")
                else:
                    render_structured_response(m["title"], m["content"])
                    st.markdown("---")

    with st.expander("About this prototype"):
        st.markdown(
            "- Uses **synthetic data** to demonstrate product analytics workflows.\n"
            "- Pattern: **Filters → compact payload summary → AI recommendations** (no raw table ingestion).\n"
            "- Designed to show **AI-native UX**, structured outputs, and stateful interactions in Streamlit."
        )
