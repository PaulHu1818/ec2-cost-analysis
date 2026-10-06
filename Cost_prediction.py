# ============================================================
# Amazon EC2 Instance Cost Analysis and Prediction
# INFO49971 - Cloud Economics
# Streamlit Version
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="EC2 Cost Analysis",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Amazon EC2 Instance Cost Analysis")

st.write(
    "Analyze Amazon EC2 pricing, compare instance families, "
    "identify cost outliers, and predict On-Demand costs."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    file_path = "ec2dataset.csv"

    df = pd.read_csv(file_path)

    return df


try:

    data = load_data()

except FileNotFoundError:

    st.error(
        "ec2dataset.csv was not found. "
        "Make sure the CSV file is in the same folder "
        "as Cost_prediction.py."
    )

    st.stop()


# Keep original data for preview
original_data = data.copy()


# ============================================================
# CLEAN COST COLUMNS
# ============================================================

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]


for column in cost_columns:

    data[column] = (
        data[column]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("hourly", "", regex=False)
        .str.strip()
    )

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("EC2 Analysis")

st.sidebar.write(
    "INFO49971 - Cloud Economics"
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Dataset:",
    "ec2dataset.csv"
)

st.sidebar.write(
    "Rows:",
    len(data)
)

st.sidebar.write(
    "Columns:",
    len(data.columns)
)


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dataset Overview",
        "💰 Cost Analysis",
        "🖥️ Instance Families",
        "🤖 Cost Prediction"
    ]
)


# ============================================================
# TAB 1 - DATASET OVERVIEW
# ============================================================

with tab1:

    st.header("Dataset Overview")

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "EC2 Instances",
        f"{len(data):,}"
    )

    col2.metric(
        "Dataset Features",
        len(data.columns)
    )

    col3.metric(
        "Average On-Demand Cost",
        f"${data['On Demand'].mean():.4f}/hr"
    )


    # --------------------------------------------------------
    # Data Preview
    # --------------------------------------------------------

    st.subheader("First 10 Rows")

    st.dataframe(
        original_data.head(10),
        use_container_width=True
    )


    # --------------------------------------------------------
    # Dataset Columns
    # --------------------------------------------------------

    st.subheader("Dataset Columns")

    column_description = pd.DataFrame(
        {
            "Column": [
                "Name",
                "API Name",
                "Instance Memory",
                "vCPUs",
                "Instance Storage",
                "Network Performance",
                "On Demand",
                "Linux Reserved cost",
                "Linux Spot Minimum cost",
                "Windows On Demand cost",
                "Windows Reserved cost"
            ],

            "Description": [
                "Name of the EC2 instance type",
                "AWS API identifier",
                "Amount of memory",
                "Number of virtual CPUs",
                "Instance storage type",
                "Network performance",
                "Linux On-Demand hourly cost",
                "Linux Reserved hourly cost",
                "Linux Spot minimum cost",
                "Windows On-Demand hourly cost",
                "Windows Reserved hourly cost"
            ]
        }
    )

    st.dataframe(
        column_description,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Missing Values
    # --------------------------------------------------------

    st.subheader("Missing Cost Values")

    missing_values = (
        data[cost_columns]
        .isnull()
        .sum()
        .reset_index()
    )

    missing_values.columns = [
        "Pricing Option",
        "Missing Values"
    ]

    st.dataframe(
        missing_values,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 - COST ANALYSIS
# ============================================================

with tab2:

    st.header("EC2 Cost Analysis")


    # --------------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------------

    st.subheader("Cost Summary Statistics")

    cost_summary = (
        data[cost_columns]
        .describe()
        .round(4)
    )

    st.dataframe(
        cost_summary,
        use_container_width=True
    )


    # --------------------------------------------------------
    # BOX PLOT
    # --------------------------------------------------------

    st.subheader("Cost Distribution")

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    sns.boxplot(
        data=data[cost_columns],
        ax=ax
    )

    ax.set_title(
        "Cost Comparison of Amazon EC2 Instances (Hourly)"
    )

    ax.set_xlabel(
        "Pricing Option"
    )

    ax.set_ylabel(
        "Cost (USD per hour)"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    st.info(
        "The circles outside the boxes represent potential "
        "cost outliers. Some high-performance EC2 instances "
        "have much higher hourly prices than most instances."
    )


    # ========================================================
    # OUTLIER ANALYSIS
    # ========================================================

    st.subheader("On-Demand Cost Outliers")


    def detect_outliers(df, column):

        Q1 = df[column].quantile(0.25)

        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = (
            Q1 - 1.5 * IQR
        )

        upper_bound = (
            Q3 + 1.5 * IQR
        )

        outliers = df[
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ]

        return (
            outliers,
            lower_bound,
            upper_bound
        )


    (
        outliers_on_demand,
        lower_bound,
        upper_bound

    ) = detect_outliers(
        data,
        "On Demand"
    )


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Number of Outliers",
        len(outliers_on_demand)
    )

    col2.metric(
        "Lower IQR Boundary",
        f"${lower_bound:.4f}"
    )

    col3.metric(
        "Upper IQR Boundary",
        f"${upper_bound:.4f}"
    )


    st.dataframe(
        outliers_on_demand[
            [
                "Name",
                "API Name",
                "Instance Memory",
                "vCPUs",
                "On Demand"
            ]
        ].sort_values(
            "On Demand",
            ascending=False
        ),

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # CHEAPEST INSTANCES
    # ========================================================

    st.subheader(
        "10 Lowest-Cost EC2 Instances"
    )

    cheapest_instances = (
        data[
            [
                "Name",
                "API Name",
                "Instance Memory",
                "vCPUs",
                "On Demand",
                "Linux Reserved cost"
            ]
        ]
        .dropna(
            subset=[
                "On Demand",
                "Linux Reserved cost"
            ]
        )
        .sort_values(
            "On Demand"
        )
        .head(10)
    )

    st.dataframe(
        cheapest_instances,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # RESERVED VS ON-DEMAND
    # ========================================================

    st.subheader(
        "Linux On-Demand vs Reserved Cost"
    )

    comparison_data = (
        data[
            [
                "Name",
                "On Demand",
                "Linux Reserved cost"
            ]
        ]
        .dropna()
        .copy()
    )

    comparison_data[
        "Savings ($/hr)"
    ] = (

        comparison_data[
            "On Demand"
        ]

        -

        comparison_data[
            "Linux Reserved cost"
        ]
    )


    comparison_data[
        "Savings (%)"
    ] = (

        comparison_data[
            "Savings ($/hr)"
        ]

        /

        comparison_data[
            "On Demand"
        ]

        * 100
    )


    comparison_data[
        "Savings (%)"
    ] = (

        comparison_data[
            "Savings (%)"
        ]
        .round(2)
    )


    st.dataframe(
        comparison_data
        .sort_values(
            "Savings ($/hr)",
            ascending=False
        )
        .head(20),

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# TAB 3 - INSTANCE FAMILY COMPARISON
# ============================================================

with tab3:

    st.header(
        "EC2 Instance Family Comparison"
    )


    # --------------------------------------------------------
    # Corrected family filter
    # --------------------------------------------------------

    def filter_instance_family(
        df,
        family
    ):

        return df[
            df["Name"]
            .astype(str)
            .str.upper()
            .str.startswith(
                family.upper(),
                na=False
            )
        ]


    t2_instances = (
        filter_instance_family(
            data,
            "T2"
        )
    )

    t3_instances = (
        filter_instance_family(
            data,
            "T3"
        )
    )


    col1, col2 = st.columns(2)

    col1.metric(
        "T2 Instances",
        len(t2_instances)
    )

    col2.metric(
        "T3 Instances",
        len(t3_instances)
    )


    # ========================================================
    # T2 SUMMARY
    # ========================================================

    st.subheader(
        "T2 Instance Cost Summary"
    )

    if len(t2_instances) > 0:

        st.dataframe(
            t2_instances[
                cost_columns
            ]
            .describe()
            .round(4),

            use_container_width=True
        )

    else:

        st.warning(
            "No T2 instances found."
        )


    # ========================================================
    # T2 PLOT
    # ========================================================

    if len(t2_instances) > 0:

        fig_t2, ax_t2 = plt.subplots(
            figsize=(12, 5)
        )

        sns.boxplot(
            data=t2_instances[
                cost_columns
            ],
            showmeans=True,
            ax=ax_t2
        )

        ax_t2.set_title(
            "Cost Distribution for T2 Instances"
        )

        ax_t2.set_ylabel(
            "Cost (USD per hour)"
        )

        ax_t2.tick_params(
            axis="x",
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(fig_t2)

        plt.close(fig_t2)


    # ========================================================
    # T3 SUMMARY
    # ========================================================

    st.subheader(
        "T3 Instance Cost Summary"
    )

    if len(t3_instances) > 0:

        st.dataframe(
            t3_instances[
                cost_columns
            ]
            .describe()
            .round(4),

            use_container_width=True
        )

    else:

        st.warning(
            "No T3 instances found."
        )


    # ========================================================
    # T3 PLOT
    # ========================================================

    if len(t3_instances) > 0:

        fig_t3, ax_t3 = plt.subplots(
            figsize=(12, 5)
        )

        sns.boxplot(
            data=t3_instances[
                cost_columns
            ],
            showmeans=True,
            ax=ax_t3
        )

        ax_t3.set_title(
            "Cost Distribution for T3 Instances"
        )

        ax_t3.set_ylabel(
            "Cost (USD per hour)"
        )

        ax_t3.tick_params(
            axis="x",
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(fig_t3)

        plt.close(fig_t3)


    # ========================================================
    # T2 / T3 CHEAPEST
    # ========================================================

    st.subheader(
        "Lowest-Cost T2 and T3 Instances"
    )


    t2_t3_comparison = pd.concat(
        [
            t2_instances[
                [
                    "Name",
                    "On Demand",
                    "Linux Reserved cost"
                ]
            ],

            t3_instances[
                [
                    "Name",
                    "On Demand",
                    "Linux Reserved cost"
                ]
            ]
        ]
    )


    t2_t3_comparison = (
        t2_t3_comparison
        .dropna()
        .sort_values(
            "On Demand"
        )
    )


    st.dataframe(
        t2_t3_comparison.head(10),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 4 - COST PREDICTION
# ============================================================

with tab4:

    st.header(
        "EC2 On-Demand Cost Prediction"
    )

    st.write(
        "The Linear Regression model uses "
        "**Instance Memory** and **vCPUs** "
        "to predict the Linux On-Demand hourly cost."
    )


    # ========================================================
    # CLEAN MEMORY
    # ========================================================

    regression_data = data.copy()


    regression_data[
        "Instance Memory"
    ] = (

        regression_data[
            "Instance Memory"
        ]
        .astype(str)
        .str.extract(
            r"([\d.]+)",
            expand=False
        )
    )


    regression_data[
        "Instance Memory"
    ] = pd.to_numeric(

        regression_data[
            "Instance Memory"
        ],

        errors="coerce"
    )


    # ========================================================
    # CLEAN VCPU
    # ========================================================

    regression_data[
        "vCPUs"
    ] = (

        regression_data[
            "vCPUs"
        ]
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False
        )
    )


    regression_data[
        "vCPUs"
    ] = pd.to_numeric(

        regression_data[
            "vCPUs"
        ],

        errors="coerce"
    )


    # ========================================================
    # DROP MISSING VALUES
    # ========================================================

    data_cleaned = (

        regression_data
        .dropna(

            subset=[
                "On Demand",
                "Instance Memory",
                "vCPUs"
            ]

        )
        .copy()
    )


    # ========================================================
    # MODEL FEATURES
    # ========================================================

    X = data_cleaned[
        [
            "Instance Memory",
            "vCPUs"
        ]
    ]

    y = data_cleaned[
        "On Demand"
    ]


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test

    ) = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================

    y_pred = model.predict(
        X_test
    )


    # ========================================================
    # MODEL METRICS
    # ========================================================

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = (
        mse ** 0.5
    )


    st.subheader(
        "Model Performance"
    )


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"${mae:.4f}"
    )

    col2.metric(
        "MSE",
        f"{mse:.4f}"
    )

    col3.metric(
        "RMSE",
        f"${rmse:.4f}"
    )


    # ========================================================
    # MODEL DETAILS
    # ========================================================

    with st.expander(
        "View Regression Model Details"
    ):

        st.write(
            f"Training Samples: "
            f"{len(X_train)}"
        )

        st.write(
            f"Testing Samples: "
            f"{len(X_test)}"
        )

        st.write(
            f"Intercept: "
            f"{model.intercept_:.6f}"
        )

        st.write(
            f"Memory Coefficient: "
            f"{model.coef_[0]:.6f}"
        )

        st.write(
            f"vCPU Coefficient: "
            f"{model.coef_[1]:.6f}"
        )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.subheader(
        "Actual vs Predicted Costs"
    )


    fig_pred, ax_pred = plt.subplots(
        figsize=(8, 6)
    )

    ax_pred.scatter(
        y_test,
        y_pred,
        alpha=0.7
    )


    minimum = min(
        y_test.min(),
        y_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_pred.max()
    )


    ax_pred.plot(
        [
            minimum,
            maximum
        ],

        [
            minimum,
            maximum
        ],

        linestyle="--"
    )


    ax_pred.set_title(
        "Actual vs Predicted On-Demand Costs"
    )

    ax_pred.set_xlabel(
        "Actual On-Demand Cost ($/hour)"
    )

    ax_pred.set_ylabel(
        "Predicted On-Demand Cost ($/hour)"
    )

    plt.tight_layout()

    st.pyplot(fig_pred)

    plt.close(fig_pred)


    # ========================================================
    # USER PREDICTION
    # ========================================================

    st.subheader(
        "Predict a New EC2 Instance"
    )


    col1, col2 = st.columns(2)


    with col1:

        memory_input = st.number_input(

            "Instance Memory (GiB)",

            min_value=0.5,

            max_value=100000.0,

            value=4.0,

            step=0.5
        )


    with col2:

        vcpu_input = st.number_input(

            "Number of vCPUs",

            min_value=1,

            max_value=1000,

            value=2,

            step=1
        )


    if st.button(
        "Predict On-Demand Cost",
        type="primary"
    ):

        new_instance = pd.DataFrame(
            {
                "Instance Memory": [
                    memory_input
                ],

                "vCPUs": [
                    vcpu_input
                ]
            }
        )


        predicted_cost = model.predict(
            new_instance
        )[0]


        st.success(
            f"Predicted On-Demand Cost: "
            f"${predicted_cost:.4f} per hour"
        )


        if predicted_cost > 0:

            estimated_monthly = (
                predicted_cost
                * 24
                * 30
            )


            st.write(
                "Approximate 30-day cost "
                "if running continuously:"
            )


            st.metric(
                "Estimated Monthly Cost",
                f"${estimated_monthly:,.2f}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "INFO49971 - Cloud Economics | "
    "Amazon EC2 Cost Analysis"
)