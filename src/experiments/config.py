from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str
    file_path: str
    target_column: str
    minority_label: int | str
    numerical_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    n_runs: int = 25
    patience: int = 10
    max_iterations: int = 500
    test_size: float = 0.1
    random_state: int = 42
    results_dir: str = "experiments/results"


DATASETS: dict[str, DatasetConfig] = {
    "credit_card_fraud": DatasetConfig(
        name="credit_card_fraud",
        file_path="data/raw/credit_card_fraud/index.csv",
        target_column="Class",
        minority_label=1,
        numerical_features=[
            "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
            "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
            "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
            "V28", "Amount",
        ],
        categorical_features=[],
    ),
    "give_me_some_credit": DatasetConfig(
        name="give_me_some_credit",
        file_path="data/raw/give_me_some_credit/index.csv",
        target_column="SeriousDlqin2yrs",
        minority_label=1,
        numerical_features=[
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ],
        categorical_features=[],
        drop_columns=["Unnamed: 0"],
    ),
    "adult_income": DatasetConfig(
        name="adult_income",
        file_path="data/raw/adult_income/index.csv",
        target_column="income",
        minority_label=">50K",
        numerical_features=[
            "age", "fnlwgt", "education-num",
            "capital-gain", "capital-loss", "hours-per-week",
        ],
        categorical_features=[
            "workclass", "education", "marital-status",
            "occupation", "relationship", "race", "sex", "native-country",
        ],
    ),
}
