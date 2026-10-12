import sys
import time
import warnings
from pathlib import Path

if sys.version_info[:2] not in [(3, 10), (3, 11)]:
    warnings.warn(
        "This project is designed for Python 3.10 or 3.11. "
        "mlxtend may fail on Python 3.12+.",
        RuntimeWarning,
    )

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


GROCERIES_COLUMNS = {"Member_number", "Date", "itemDescription"}
INSTACART_MERGED_COLUMNS = {"order_id", "product_name"}
INSTACART_ORDER_COLUMNS = {"order_id", "product_id"}
INSTACART_PRODUCT_COLUMNS = {"product_id", "product_name"}
ONLINE_RETAIL_COLUMNS = {"InvoiceNo", "Description"}
ONLINE_RETAIL_II_COLUMNS = {"Invoice", "Description"}
SAMPLE_PATH = Path(__file__).with_name("sample_groceries.csv")

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
)

if sys.version_info[:2] not in [(3, 10), (3, 11)]:
    st.warning(
        "Python version warning: this app is designed for Python 3.10 or 3.11. "
        "mlxtend may break on Python 3.12+."
    )


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_PATH)
    is_valid, message = validate_dataset(df)
    if not is_valid:
        st.warning(message)
        return df

    transactions = preprocess_groceries_data(df)
    basket_df = encode_transactions(transactions)
    itemsets, _ = run_algorithm("fpgrowth", basket_df, 0.005, 3)
    rules = generate_association_rules(itemsets, 0.20, 1.0)
    if len(rules) < 3:
        st.warning(
            "The built-in sample produced fewer than 3 association rules with "
            "default settings. Try lowering support or confidence."
        )
    return df


@st.cache_data(show_spinner=False)
def load_uploaded_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def load_uploaded_products(file) -> pd.DataFrame:
    return pd.read_csv(file)


def detect_dataset_format(
    df: pd.DataFrame, products_df: pd.DataFrame | None = None
) -> tuple[str, str]:
    columns = set(df.columns)
    if GROCERIES_COLUMNS.issubset(columns):
        return "Kaggle Groceries", "TransactionID = Member_number + Date; item = itemDescription"
    if INSTACART_MERGED_COLUMNS.issubset(columns):
        return "Instacart merged", "TransactionID = order_id; item = product_name"
    if INSTACART_ORDER_COLUMNS.issubset(columns):
        if products_df is not None and INSTACART_PRODUCT_COLUMNS.issubset(products_df.columns):
            return "Instacart two-file", "TransactionID = order_id; item = product_name from products.csv"
        return "Instacart missing products", "Upload products.csv so product_id can be converted to product_name"
    if ONLINE_RETAIL_COLUMNS.issubset(columns):
        return "Online Retail", "TransactionID = InvoiceNo; item = Description"
    if ONLINE_RETAIL_II_COLUMNS.issubset(columns):
        return "Online Retail II", "TransactionID = Invoice; item = Description"
    return "Unknown", "Could not detect a supported transaction format"


def validate_dataset(
    df: pd.DataFrame, products_df: pd.DataFrame | None = None
) -> tuple[bool, str]:
    dataset_format, _ = detect_dataset_format(df, products_df)
    if dataset_format in {
        "Kaggle Groceries",
        "Instacart merged",
        "Instacart two-file",
        "Online Retail",
        "Online Retail II",
    }:
        return True, ""
    if dataset_format == "Instacart missing products":
        return (
            False,
            "This looks like Instacart order_products data. Please also upload products.csv "
            "with columns product_id and product_name. Found main columns: "
            f"{list(df.columns)}.",
        )
    return (
        False,
        "Your file is missing required columns. Supported formats are: "
        "Groceries_dataset.csv with Member_number, Date, itemDescription; "
        "Instacart merged data with order_id, product_name; "
        "Instacart order_products with order_id, product_id plus products.csv; "
        "or Online Retail with InvoiceNo, Description / Invoice, Description. "
        f"Found: {list(df.columns)}.",
    )


@st.cache_data(show_spinner=False)
def preprocess_groceries_data(
    df: pd.DataFrame, products_df: pd.DataFrame | None = None
) -> list[list[str]]:
    work_df = df.copy()
    dataset_format, _ = detect_dataset_format(work_df, products_df)

    if dataset_format == "Kaggle Groceries":
        work_df = work_df.dropna(subset=["Member_number", "Date", "itemDescription"])
        if work_df.empty:
            return []
        work_df["TransactionID"] = (
            work_df["Member_number"].astype(str) + "_" + work_df["Date"].astype(str)
        )
        grouped = work_df.groupby("TransactionID")["itemDescription"]
    elif dataset_format == "Instacart merged":
        work_df = work_df.dropna(subset=["order_id", "product_name"])
        if work_df.empty:
            return []
        grouped = work_df.groupby("order_id")["product_name"]
    elif dataset_format == "Instacart two-file":
        product_lookup = products_df[["product_id", "product_name"]].dropna().copy()
        product_lookup["product_id"] = product_lookup["product_id"].astype(str)
        work_df = work_df.dropna(subset=["order_id", "product_id"]).copy()
        work_df["product_id"] = work_df["product_id"].astype(str)
        work_df = work_df.merge(product_lookup, on="product_id", how="inner")
        if work_df.empty:
            return []
        grouped = work_df.groupby("order_id")["product_name"]
    elif dataset_format in {"Online Retail", "Online Retail II"}:
        invoice_col = "InvoiceNo" if "InvoiceNo" in work_df.columns else "Invoice"
        work_df = work_df.dropna(subset=[invoice_col, "Description"]).copy()
        work_df = work_df[~work_df[invoice_col].astype(str).str.upper().str.startswith("C")]
        if "Quantity" in work_df.columns:
            work_df = work_df[pd.to_numeric(work_df["Quantity"], errors="coerce") > 0]
        if work_df.empty:
            return []
        grouped = work_df.groupby(invoice_col)["Description"]
    else:
        work_df = work_df.dropna(how="all")
        if work_df.empty:
            return []
        grouped = work_df.apply(
            lambda row: [str(item).strip() for item in row.dropna().tolist()], axis=1
        )
        return [sorted(set(items)) for items in grouped if items]

    transactions = []
    for _, items in grouped:
        cleaned = sorted({str(item).strip() for item in items if str(item).strip()})
        if cleaned:
            transactions.append(cleaned)
    return transactions


@st.cache_data(show_spinner=False)
def encode_transactions(transactions: list[list[str]]) -> pd.DataFrame:
    encoder = TransactionEncoder()
    encoder.fit(transactions)
    array = encoder.transform(transactions, sparse=True)
    return pd.DataFrame.sparse.from_spmatrix(array, columns=encoder.columns_)


@st.cache_data(show_spinner=False)
def run_algorithm(
    algorithm_name: str, basket_df_hash: pd.DataFrame, min_support: float, max_len: int
) -> tuple[pd.DataFrame, float]:
    """Runs either 'apriori' or 'fpgrowth'. Returns frequent itemsets and runtime."""
    start = time.perf_counter()
    if algorithm_name == "apriori":
        itemsets = apriori(
            basket_df_hash,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
            low_memory=True,
        )
    elif algorithm_name == "fpgrowth":
        itemsets = fpgrowth(
            basket_df_hash,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
        )
    else:
        raise ValueError("algorithm_name must be 'apriori' or 'fpgrowth'")
    runtime = time.perf_counter() - start
    return itemsets.sort_values("support", ascending=False).reset_index(drop=True), runtime


def generate_association_rules(
    itemsets: pd.DataFrame, min_confidence: float, min_lift: float
) -> pd.DataFrame:
    if itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return rules
    rules = rules[rules["lift"] >= min_lift].copy()
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def _items_to_text(items: frozenset) -> str:
    return ", ".join(sorted(items))


def clean_rules_dataframe(rules_df: pd.DataFrame) -> pd.DataFrame:
    if rules_df.empty:
        return rules_df
    display_df = rules_df.copy()
    display_df["antecedents"] = display_df["antecedents"].apply(_items_to_text)
    display_df["consequents"] = display_df["consequents"].apply(_items_to_text)
    cols = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    ]
    return display_df[cols]


def _rule_key(row: pd.Series) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (tuple(sorted(row["antecedents"])), tuple(sorted(row["consequents"])))


def calculate_rule_overlap(
    rules_a: pd.DataFrame, rules_b: pd.DataFrame, top_n: int = 10
) -> float:
    if rules_a.empty or rules_b.empty:
        return 0.0
    top_a = {_rule_key(row) for _, row in rules_a.nlargest(top_n, "lift").iterrows()}
    top_b = {_rule_key(row) for _, row in rules_b.nlargest(top_n, "lift").iterrows()}
    denominator = max(1, min(len(top_a), len(top_b), top_n))
    return round(len(top_a.intersection(top_b)) / denominator * 100, 2)


def compare_algorithms(
    apriori_rules: pd.DataFrame,
    fpgrowth_rules: pd.DataFrame,
    apriori_time: float,
    fpgrowth_time: float,
    apriori_itemsets: pd.DataFrame | None = None,
    fpgrowth_itemsets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    overlap = calculate_rule_overlap(apriori_rules, fpgrowth_rules)

    def safe_mean(df: pd.DataFrame, col: str) -> float:
        return 0.0 if df.empty else float(df[col].mean())

    def safe_max(df: pd.DataFrame, col: str) -> float:
        return 0.0 if df.empty else float(df[col].max())

    return pd.DataFrame(
        [
            ["Runtime (seconds)", apriori_time, fpgrowth_time, "Efficiency comparison"],
            [
                "Frequent itemsets found",
                0 if apriori_itemsets is None else len(apriori_itemsets),
                0 if fpgrowth_itemsets is None else len(fpgrowth_itemsets),
                "Same parameters should produce the same itemsets",
            ],
            [
                "Association rules generated",
                len(apriori_rules),
                len(fpgrowth_rules),
                "Small differences can come from floating-point rounding",
            ],
            [
                "Average confidence",
                safe_mean(apriori_rules, "confidence"),
                safe_mean(fpgrowth_rules, "confidence"),
                "Recommendation strength average",
            ],
            [
                "Average lift",
                safe_mean(apriori_rules, "lift"),
                safe_mean(fpgrowth_rules, "lift"),
                "Positive association average",
            ],
            [
                "Best (max) confidence",
                safe_max(apriori_rules, "confidence"),
                safe_max(fpgrowth_rules, "confidence"),
                "Highest confidence rule",
            ],
            [
                "Best (max) lift",
                safe_max(apriori_rules, "lift"),
                safe_max(fpgrowth_rules, "lift"),
                "Highest lift rule",
            ],
            [
                "Top-10 rules overlap %",
                overlap,
                overlap,
                "Shared top rules by lift",
            ],
        ],
        columns=["Metric", "Apriori", "FP-Growth", "Comment"],
    )


def recommend_products(
    selected_items: list[str], apriori_rules: pd.DataFrame, fpgrowth_rules: pd.DataFrame
) -> pd.DataFrame:
    selected = set(selected_items)
    rows = []
    for algorithm, rules in [("Apriori", apriori_rules), ("FP-Growth", fpgrowth_rules)]:
        if rules.empty:
            continue
        for _, row in rules.iterrows():
            antecedents = set(row["antecedents"])
            if antecedents.issubset(selected):
                for item in row["consequents"]:
                    if item not in selected:
                        rows.append(
                            {
                                "Product": item,
                                "Source": algorithm,
                                "Confidence": row["confidence"],
                                "Lift": row["lift"],
                                "Support": row["support"],
                                "Antecedents": _items_to_text(row["antecedents"]),
                            }
                        )
    if not rows:
        return pd.DataFrame()

    recs = pd.DataFrame(rows)
    combined = []
    for product, group in recs.groupby("Product"):
        best = group.sort_values(["Confidence", "Lift"], ascending=False).iloc[0].copy()
        sources = set(group["Source"])
        best["Source"] = "Both" if sources == {"Apriori", "FP-Growth"} else next(iter(sources))
        combined.append(best)
    return pd.DataFrame(combined).sort_values(["Confidence", "Lift"], ascending=False)


def create_frequency_chart(transactions: list[list[str]]) -> go.Figure:
    counts = pd.Series([item for basket in transactions for item in basket]).value_counts().head(15)
    data = counts.sort_values(ascending=True).reset_index()
    data.columns = ["Item", "Frequency"]
    fig = px.bar(
        data,
        x="Frequency",
        y="Item",
        orientation="h",
        title="Top 15 Most Frequent Items",
        labels={"Frequency": "Number of transactions", "Item": "Product"},
    )
    fig.update_layout(font_size=14, title_font_size=20, yaxis_title="Product")
    return fig


def create_runtime_chart(apriori_time: float, fpgrowth_time: float) -> go.Figure:
    fig = px.bar(
        pd.DataFrame(
            {"Algorithm": ["Apriori", "FP-Growth"], "Runtime": [apriori_time, fpgrowth_time]}
        ),
        x="Algorithm",
        y="Runtime",
        title="Apriori vs FP-Growth Runtime",
        labels={"Runtime": "Runtime (seconds)"},
        color="Algorithm",
    )
    fig.update_layout(font_size=14, title_font_size=20, showlegend=False)
    return fig


def create_lift_scatter(apriori_rules: pd.DataFrame, fpgrowth_rules: pd.DataFrame) -> go.Figure:
    frames = []
    for name, rules in [("Apriori", apriori_rules), ("FP-Growth", fpgrowth_rules)]:
        if not rules.empty:
            frame = rules.nlargest(10, "lift").copy()
            frame["Algorithm"] = name
            frame["Rule"] = frame.apply(
                lambda row: f"{_items_to_text(row['antecedents'])} -> {_items_to_text(row['consequents'])}",
                axis=1,
            )
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return go.Figure()
    fig = px.scatter(
        combined,
        x="confidence",
        y="lift",
        size="support",
        color="Algorithm",
        hover_name="Rule",
        title="Top 10 Rules by Lift",
        labels={"confidence": "Confidence", "lift": "Lift", "support": "Support"},
    )
    fig.update_layout(font_size=14, title_font_size=20)
    return fig


def create_comparison_charts(comparison_df: pd.DataFrame) -> go.Figure:
    subset = comparison_df[
        comparison_df["Metric"].isin(["Average confidence", "Average lift"])
    ].melt(id_vars="Metric", value_vars=["Apriori", "FP-Growth"], var_name="Algorithm")
    fig = px.bar(
        subset,
        x="Metric",
        y="value",
        color="Algorithm",
        barmode="group",
        title="Average Confidence and Average Lift",
        labels={"value": "Score", "Metric": "Metric"},
    )
    fig.update_layout(font_size=14, title_font_size=20)
    return fig


def create_rules_count_chart(apriori_rules: pd.DataFrame, fpgrowth_rules: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        pd.DataFrame(
            {
                "Algorithm": ["Apriori", "FP-Growth"],
                "Rules": [len(apriori_rules), len(fpgrowth_rules)],
            }
        ),
        x="Algorithm",
        y="Rules",
        color="Algorithm",
        title="Number of Rules Generated",
        labels={"Rules": "Association rules"},
    )
    fig.update_layout(font_size=14, title_font_size=20, showlegend=False)
    return fig


def create_support_distribution(itemsets: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        itemsets,
        x="support",
        nbins=30,
        title="Support Distribution for Frequent Itemsets",
        labels={"support": "Support", "count": "Number of itemsets"},
    )
    fig.update_layout(font_size=14, title_font_size=20, yaxis_title="Number of itemsets")
    return fig


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    export_df = clean_rules_dataframe(df) if "antecedents" in df.columns else df
    return export_df.to_csv(index=False).encode("utf-8")


def show_rules_table(rules: pd.DataFrame, top_rules: int) -> None:
    if rules.empty:
        st.warning(
            "Frequent itemsets were found but no rules meet your confidence threshold. "
            "Try lowering minimum confidence to 0.10."
        )
        return
    st.dataframe(
        clean_rules_dataframe(rules).head(top_rules).style.format(
            {
                "support": "{:.3f}",
                "confidence": "{:.1%}",
                "lift": "{:.2f}",
                "leverage": "{:.3f}",
                "conviction": "{:.2f}",
            }
        ),
        use_container_width=True,
    )


def show_top_recommendations(rules: pd.DataFrame, top_rules: int) -> None:
    if rules.empty:
        return
    st.subheader("Top Recommendations")
    for _, row in rules.head(top_rules).iterrows():
        st.info(
            f"Customers who bought {_items_to_text(row['antecedents'])} also frequently "
            f"bought {_items_to_text(row['consequents'])} "
            f"(confidence: {row['confidence']:.0%}, lift: {row['lift']:.2f})."
        )


def init_state() -> None:
    defaults = {
        "analysis_ready": False,
        "sidebar_min_support": 0.005,
        "sidebar_min_confidence": 0.20,
        "sidebar_min_lift": 1.0,
        "sidebar_max_len": 3,
        "sidebar_top_rules": 10,
        "transactions": [],
        "basket_df": pd.DataFrame(),
        "apriori_itemsets": pd.DataFrame(),
        "fpgrowth_itemsets": pd.DataFrame(),
        "apriori_rules": pd.DataFrame(),
        "fpgrowth_rules": pd.DataFrame(),
        "apriori_time": 0.0,
        "fpgrowth_time": 0.0,
        "run_error": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()

st.title("Frequently Bought Together: Market Basket Analysis using Apriori and FP-Growth")
st.caption(
    "A beginner-friendly university project showing how stores recommend products that are often purchased together."
)

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader(
        "Upload main dataset CSV",
        type=["csv"],
        help=(
            "Groceries: Groceries_dataset.csv. Instacart: order_products__prior.csv "
            "or a merged CSV with order_id and product_name. Online Retail: CSV with InvoiceNo and Description."
        ),
    )
    products_file = None
    if uploaded_file is not None:
        products_file = st.file_uploader(
            "Optional Instacart products.csv",
            type=["csv"],
            help="Only needed when the main Instacart file has product_id instead of product_name.",
        )

    st.header("Settings")
    if uploaded_file is not None:
        st.info(
            "Real market-basket uploads can be sparse after grouping rows into transactions. "
            "If no rules appear, use the Kaggle-friendly preset below."
        )
        if st.button("Use large-dataset friendly settings", use_container_width=True):
            st.session_state.sidebar_min_support = 0.001
            st.session_state.sidebar_min_confidence = 0.05
            st.session_state.sidebar_min_lift = 1.0
            st.session_state.sidebar_max_len = 3

    min_support = st.number_input(
        "Minimum support",
        min_value=0.001,
        max_value=0.1,
        value=st.session_state.sidebar_min_support,
        step=0.001,
        format="%.3f",
        help="How often an itemset appears across all transactions.",
        key="sidebar_min_support",
    )
    min_confidence = st.number_input(
        "Minimum confidence",
        min_value=0.05,
        max_value=1.0,
        value=st.session_state.sidebar_min_confidence,
        step=0.05,
        format="%.2f",
        help="How often the rule is correct when the antecedent is present.",
        key="sidebar_min_confidence",
    )
    min_lift = st.number_input(
        "Minimum lift",
        min_value=1.0,
        max_value=5.0,
        value=st.session_state.sidebar_min_lift,
        step=0.1,
        format="%.1f",
        help="How much more likely the consequent is compared to random chance. Lift > 1 means a positive association.",
        key="sidebar_min_lift",
    )
    max_len = st.slider("Maximum itemset length", 2, 5, key="sidebar_max_len")
    top_rules = st.slider("Number of top rules", 5, 50, key="sidebar_top_rules")

if uploaded_file is not None:
    source_df = load_uploaded_data(uploaded_file)
    products_df = load_uploaded_products(products_file) if products_file is not None else None
    detected_format, transaction_note = detect_dataset_format(source_df, products_df)
    data_label = f"uploaded dataset ({detected_format})"
else:
    source_df = load_sample_data()
    products_df = None
    detected_format, transaction_note = detect_dataset_format(source_df)
    data_label = "built-in sample dataset"

is_valid, error_message = validate_dataset(source_df, products_df)
if not is_valid:
    st.error(error_message)
    st.stop()

transactions_all = preprocess_groceries_data(source_df, products_df)
if not transactions_all:
    st.error("Your dataset has no valid rows after removing missing values.")
    st.stop()

with st.sidebar:
    transaction_count = len(transactions_all)
    use_sampling = False
    sample_size = transaction_count
    if transaction_count > 5000:
        st.warning(
            f"This dataset has {transaction_count:,} transactions. The default is to sample "
            "5,000 transactions for speed. Disable sampling only if your computer has enough memory."
        )
        use_sampling = st.checkbox("Enable sampling", value=True)
        sample_size = st.slider(
            "Transaction sample size",
            min_value=1000,
            max_value=transaction_count,
            value=min(5000, transaction_count),
            step=500,
        )
        if not use_sampling and transaction_count > 20000:
            st.warning(
                f"Your dataset has {transaction_count:,} transactions. We recommend sampling. "
                "Enable sampling in the sidebar."
            )
    else:
        st.caption("Using all transactions because the dataset is small.")

    run_clicked = st.button("▶ Run Analysis", type="primary", use_container_width=True)

if use_sampling and sample_size < transaction_count:
    rng = np.random.default_rng(42)
    sample_idx = sorted(rng.choice(transaction_count, size=sample_size, replace=False))
    transactions = [transactions_all[i] for i in sample_idx]
else:
    transactions = transactions_all

unique_items = sorted({item for basket in transactions_all for item in basket})

if run_clicked:
    st.session_state.run_error = ""
    try:
        basket_df = encode_transactions(transactions)
        with st.spinner("Running Apriori algorithm, please wait..."):
            apriori_itemsets, apriori_time = run_algorithm("apriori", basket_df, min_support, max_len)
        if apriori_itemsets.empty:
            st.session_state.run_error = (
                f"No frequent itemsets found. Support {min_support:.3f} may be too high "
                "for this dataset. Try lowering it to 0.005 or 0.001."
            )
        with st.spinner("Running FP-Growth algorithm, please wait..."):
            fpgrowth_itemsets, fpgrowth_time = run_algorithm("fpgrowth", basket_df, min_support, max_len)
        if fpgrowth_itemsets.empty:
            st.session_state.run_error = (
                f"No frequent itemsets found. Support {min_support:.3f} may be too high "
                "for this dataset. Try lowering it to 0.005 or 0.001."
            )

        apriori_rules = generate_association_rules(apriori_itemsets, min_confidence, min_lift)
        fpgrowth_rules = generate_association_rules(fpgrowth_itemsets, min_confidence, min_lift)

        if (
            st.session_state.run_error == ""
            and (not apriori_itemsets.empty or not fpgrowth_itemsets.empty)
            and apriori_rules.empty
            and fpgrowth_rules.empty
        ):
            st.session_state.run_error = (
                "Frequent itemsets were found, but no association rules meet your current "
                f"confidence/lift thresholds. For larger real datasets, try support 0.001, "
                "confidence 0.05, and lift 1.0, then press ▶ Run Analysis again."
            )

        st.session_state.update(
            {
                "analysis_ready": True,
                "transactions": transactions,
                "basket_df": basket_df,
                "apriori_itemsets": apriori_itemsets,
                "fpgrowth_itemsets": fpgrowth_itemsets,
                "apriori_rules": apriori_rules,
                "fpgrowth_rules": fpgrowth_rules,
                "apriori_time": apriori_time,
                "fpgrowth_time": fpgrowth_time,
            }
        )
    except MemoryError:
        st.session_state.run_error = (
            f"Your dataset has {len(transactions):,} transactions. We recommend sampling. "
            "Enable sampling in the sidebar."
        )
    except Exception as exc:
        st.session_state.run_error = f"Something went wrong while running the analysis: {exc}"

if st.session_state.run_error:
    st.error(st.session_state.run_error)

tabs = st.tabs(
    [
        "🛒 Introduction",
        "📂 Dataset",
        "📊 Apriori",
        "🌲 FP-Growth",
        "⚖️ Comparison",
        "🔍 Recommendation Demo",
        "📥 Downloads",
    ]
)

with tabs[0]:
    st.header("How Stores Find Frequently Bought Together Products")
    st.write(
        "Market Basket Analysis looks at shopping baskets and finds products that appear together. "
        "Stores use these patterns for product bundles, checkout recommendations, shelf placement, "
        "and promotions."
    )
    cards = [
        (
            "What is Market Basket Analysis?",
            "It studies many shopping baskets to find useful product combinations. "
            "Example: if many shoppers buy whole milk and other vegetables together, the store can recommend one when the other is in the cart.",
        ),
        (
            "What is Support?",
            "Support measures how common an itemset is. Formula: support(A) = freq(A) / total transactions. "
            "Example: if whole milk appears in 30 out of 100 baskets, support is 0.30.",
        ),
        (
            "What is Confidence?",
            "Confidence measures how often a rule is correct. Formula: confidence(A→B) = freq(A∪B) / freq(A). "
            "Example: if 20 shoppers bought whole milk and 12 of them also bought yogurt, confidence is 60%.",
        ),
        (
            "What is Lift?",
            "Lift compares a recommendation with random chance. Formula: lift(A→B) = confidence(A→B) / support(B). "
            "Example: lift 1.50 means B is 1.5 times more likely when A is present.",
        ),
        (
            "How does Apriori work?",
            "Apriori creates candidate itemsets level by level and removes combinations that are too rare. "
            "Example: if coffee and sugar are frequent, it may test coffee, sugar, and cookies next.",
        ),
        (
            "How does FP-Growth work?",
            "FP-Growth compresses transactions into an FP-tree and mines patterns from that tree. "
            "Example: repeated grocery paths like whole milk → yogurt can be counted efficiently.",
        ),
        (
            "Why compare them?",
            "Both algorithms produce the same results with the same settings. The meaningful difference is efficiency, especially on larger datasets.",
        ),
    ]
    for title, body in cards:
        with st.expander(title):
            st.write(body)

with tabs[1]:
    st.header("Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(source_df):,}")
    col2.metric("Transactions in use", f"{len(transactions):,}")
    col3.metric("Unique items", f"{len(unique_items):,}")
    st.write(
        f"Current source: **{data_label}**. Transactions are created as "
        f"**{transaction_note}**, then all items in the same transaction are grouped into one basket."
    )
    st.dataframe(source_df.head(20), use_container_width=True)
    st.plotly_chart(create_frequency_chart(transactions), use_container_width=True)

analysis_ready = st.session_state.analysis_ready

with tabs[2]:
    st.header("Apriori Algorithm")
    if not analysis_ready:
        st.info("Choose settings in the sidebar, then press ▶ Run Analysis.")
    else:
        st.metric("Runtime", f"{st.session_state.apriori_time:.3f} seconds")
        st.metric("Frequent itemsets found", f"{len(st.session_state.apriori_itemsets):,}")
        if st.session_state.apriori_itemsets.empty:
            st.warning(
                f"No frequent itemsets found. Support {min_support:.3f} may be too high for this dataset. "
                "Try lowering it to 0.005 or 0.001."
            )
        else:
            st.plotly_chart(create_support_distribution(st.session_state.apriori_itemsets), use_container_width=True, key="apriori_support_dist")
            show_rules_table(st.session_state.apriori_rules, top_rules)
            show_top_recommendations(st.session_state.apriori_rules, min(top_rules, 5))

with tabs[3]:
    st.header("FP-Growth Algorithm")
    if not analysis_ready:
        st.info("Choose settings in the sidebar, then press ▶ Run Analysis.")
    else:
        st.metric("Runtime", f"{st.session_state.fpgrowth_time:.3f} seconds")
        st.metric("Frequent itemsets found", f"{len(st.session_state.fpgrowth_itemsets):,}")
        if st.session_state.fpgrowth_itemsets.empty:
            st.warning(
                f"No frequent itemsets found. Support {min_support:.3f} may be too high for this dataset. "
                "Try lowering it to 0.005 or 0.001."
            )
        else:
            st.plotly_chart(create_support_distribution(st.session_state.fpgrowth_itemsets), use_container_width=True, key="fpgrowth_support_dist")
            show_rules_table(st.session_state.fpgrowth_rules, top_rules)
            show_top_recommendations(st.session_state.fpgrowth_rules, min(top_rules, 5))

with tabs[4]:
    st.header("Comparison")
    st.write(
        "Both algorithms produce the same results. The only meaningful difference is efficiency. "
        "If counts differ slightly, it is due to floating-point rounding in support calculations, not algorithmic quality."
    )
    if not analysis_ready:
        st.info("Run the analysis to compare Apriori and FP-Growth.")
    else:
        comparison = compare_algorithms(
            st.session_state.apriori_rules,
            st.session_state.fpgrowth_rules,
            st.session_state.apriori_time,
            st.session_state.fpgrowth_time,
            st.session_state.apriori_itemsets,
            st.session_state.fpgrowth_itemsets,
        )
        st.dataframe(comparison, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(create_runtime_chart(st.session_state.apriori_time, st.session_state.fpgrowth_time), use_container_width=True)
        c2.plotly_chart(create_rules_count_chart(st.session_state.apriori_rules, st.session_state.fpgrowth_rules), use_container_width=True)
        st.plotly_chart(create_lift_scatter(st.session_state.apriori_rules, st.session_state.fpgrowth_rules), use_container_width=True)
        st.plotly_chart(create_comparison_charts(comparison), use_container_width=True)

        overlap = calculate_rule_overlap(st.session_state.apriori_rules, st.session_state.fpgrowth_rules)
        if overlap >= 95:
            if st.session_state.fpgrowth_time < st.session_state.apriori_time:
                speedup = st.session_state.apriori_time / max(st.session_state.fpgrowth_time, 1e-9)
                timing_text = (
                    f"FP-Growth completed in {st.session_state.fpgrowth_time:.3f}s vs "
                    f"Apriori's {st.session_state.apriori_time:.3f}s, a {speedup:.2f}x speedup."
                )
            elif st.session_state.apriori_time < st.session_state.fpgrowth_time:
                speedup = st.session_state.fpgrowth_time / max(st.session_state.apriori_time, 1e-9)
                timing_text = (
                    f"Apriori completed in {st.session_state.apriori_time:.3f}s vs "
                    f"FP-Growth's {st.session_state.fpgrowth_time:.3f}s, a {speedup:.2f}x speedup. "
                    "This can happen on small, sparse baskets or when maximum itemset length is low."
                )
            else:
                timing_text = "Both algorithms completed in almost the same time."
            st.success(f"Both algorithms found the same recommendation patterns. {timing_text}")
        else:
            st.info(
                "Minor differences exist in the top rules, likely due to floating-point precision "
                "in support calculations. Both algorithms use the same mathematical foundation."
            )

with tabs[5]:
    st.header("Recommendation Demo")
    selected_items = st.multiselect("Select one or more items", unique_items)
    if not analysis_ready:
        st.info("Run the analysis first, then select basket items to get recommendations.")
    elif selected_items:
        recommendations = recommend_products(
            selected_items,
            st.session_state.apriori_rules,
            st.session_state.fpgrowth_rules,
        )
        if recommendations.empty:
            st.warning(
                "No recommendations found for this combination. Try selecting fewer items, "
                "or lower minimum support and confidence in the sidebar."
            )
        else:
            for _, rec in recommendations.iterrows():
                st.container(border=True).markdown(
                    f"**Product:** {rec['Product']}  \n"
                    f"**Source:** {rec['Source']}  \n"
                    f"**Confidence:** {rec['Confidence']:.0%}  \n"
                    f"**Lift:** {rec['Lift']:.2f}  \n"
                    f"**Support:** {rec['Support']:.3f}  \n"
                    f"**Explanation:** Customers who bought {rec['Antecedents']} also frequently "
                    f"bought {rec['Product']} (found in {rec['Confidence'] * 100:.0f} out of 100 "
                    f"transactions containing {rec['Antecedents']})."
                )
    else:
        st.info("Select basket items to see frequently bought together recommendations.")

with tabs[6]:
    st.header("Downloads")
    if not analysis_ready:
        st.info("Run the analysis to enable downloads.")
    else:
        st.download_button(
            "Download Apriori rules CSV",
            data=convert_df_to_csv(st.session_state.apriori_rules),
            file_name="apriori_rules.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download FP-Growth rules CSV",
            data=convert_df_to_csv(st.session_state.fpgrowth_rules),
            file_name="fpgrowth_rules.csv",
            mime="text/csv",
        )
        comparison = compare_algorithms(
            st.session_state.apriori_rules,
            st.session_state.fpgrowth_rules,
            st.session_state.apriori_time,
            st.session_state.fpgrowth_time,
            st.session_state.apriori_itemsets,
            st.session_state.fpgrowth_itemsets,
        )
        st.download_button(
            "Download comparison CSV",
            data=convert_df_to_csv(comparison),
            file_name="algorithm_comparison.csv",
            mime="text/csv",
        )
