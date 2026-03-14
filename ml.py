!pip install aiogram>=3.0.0 torch transformers datasets scikit-learn pandas nest-asyncio cachetools

import json
import logging
import os
import random
import unittest # ДОБАВЛЕНО ДЛЯ ТЕСТОВ
from unittest.mock import patch # ДОБАВЛЕНО ДЛЯ ТЕСТОВ
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =========================
# 1. Global Configuration
# =========================
RANDOM_SEED = 42

# Source datasets (can be overridden by env vars)
MULTICLASS_DATA_PATH = "hackaton_test.csv"
BINARY_DATA_PATH = "/content/бинарный классификатор.csv" # Убрал /content/ чтобы работало универсально

# Expected canonical columns
TEXT_COLUMN = "question"
INTENT_LABEL_COLUMN = "Intent"

# Model / training settings
MODEL_NAME = "cointegrated/rubert-tiny2"
PIPELINE_OUTPUT_DIR = "rubert_two_stage_pipeline"
STAGE1_OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "stage1_binary")
STAGE2_OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "stage2_intents")
EVAL_ARTIFACTS_DIR = "evaluation_artifacts"
PIPELINE_E2E_ARTIFACTS_DIR = os.path.join(PIPELINE_OUTPUT_DIR, EVAL_ARTIFACTS_DIR, "two_stage_e2e")

MAX_LENGTH = 128
TEST_SIZE = 0.2
NUM_EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.0

# Stage-1 threshold tuning settings
STAGE1_THRESHOLD_DEFAULT = 0.50
STAGE1_THRESHOLD_MIN = 0.05
STAGE1_THRESHOLD_MAX = 0.95
STAGE1_THRESHOLD_STEP = 0.01
STAGE1_THRESHOLD_FILE_NAME = "stage1_threshold.json"
STAGE1_THRESHOLD_CURVE_CSV = "stage1_threshold_tuning_curve.csv"
STAGE1_THRESHOLD_CURVE_PNG = "stage1_threshold_tuning_curve.png"

# Business labels
TRASH_LABEL_NAME = "Мусор"
NOT_TRASH_LABEL_NAME = "Не мусор"

# Runtime / Colab env vars
ENV_MULTICLASS_DATA_PATH = "MULTICLASS_DATA_PATH"
ENV_BINARY_DATA_PATH = "BINARY_DATA_PATH"
ENV_PIPELINE_OUTPUT_DIR = "PIPELINE_OUTPUT_DIR"
ENV_COLAB_MOUNT_DRIVE = "COLAB_MOUNT_DRIVE"

# Colab paths
COLAB_CONTENT_DIR = "/content"
COLAB_DRIVE_MOUNT_POINT = "/content/drive"
COLAB_DRIVE_ROOT = "/content/drive/MyDrive"
COLAB_NOTEBOOKS_DIR = "/content/drive/MyDrive/Colab Notebooks"

# Binary label synonyms (normalized)
BINARY_TRASH_VALUES = {
    "1",
    "true",
    "yes",
    "мусор",
    "спам",
    "spam",
    "trash",
    "garbage",
}
BINARY_NOT_TRASH_VALUES = {
    "0",
    "false",
    "no",
    "не мусор",
    "не_мусор",
    "не-мусор",
    "чисто",
    "ham",
    "not_trash",
    "not trash",
    "не спам",
}


# =========================
# 2. Logging Setup
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# 3. Utility Structures
# =========================
@dataclass
class PreparedData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    label2id: Dict[str, int]
    id2label: Dict[int, str]


@dataclass
class StageTrainingResult:
    trainer: Trainer
    tokenizer: AutoTokenizer
    tokenized_test: Dataset
    prepared_data: PreparedData


# Lazy inference cache (loaded only on first predict call).
_STAGE1_MODEL: Optional[AutoModelForSequenceClassification] = None
_STAGE1_TOKENIZER: Optional[AutoTokenizer] = None
_STAGE2_MODEL: Optional[AutoModelForSequenceClassification] = None
_STAGE2_TOKENIZER: Optional[AutoTokenizer] = None
_STAGE2_ID2LABEL: Optional[Dict[int, str]] = None
_STAGE1_THRESHOLD: Optional[float] = None
_RUNTIME_INITIALIZED = False


# =========================
# 4. Runtime / Colab Setup
# =========================
def is_running_in_colab() -> bool:
    """Detect whether code runs inside Google Colab."""
    try:
        import google.colab  # type: ignore # noqa: F401
        return True
    except ImportError:
        return False


def parse_env_bool(env_name: str, default: bool) -> bool:
    """Parse boolean environment variable safely."""
    raw = os.getenv(env_name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def try_mount_colab_drive() -> None:
    """Try to mount Google Drive in Colab when enabled by env var."""
    if not is_running_in_colab():
        return

    should_mount = parse_env_bool(ENV_COLAB_MOUNT_DRIVE, default=False)
    if not should_mount:
        logger.info(
            "Colab detected. Drive mounting is skipped (set %s=1 to enable).",
            ENV_COLAB_MOUNT_DRIVE,
        )
        return

    try:
        from google.colab import drive  # type: ignore
        logger.info("Mounting Google Drive to: %s", COLAB_DRIVE_MOUNT_POINT)
        drive.mount(COLAB_DRIVE_MOUNT_POINT, force_remount=False)
    except Exception as exc:  # pragma: no cover - runtime-dependent
        logger.warning("Failed to mount Google Drive: %s", exc)


def build_dataset_search_roots() -> List[str]:
    """Build ordered list of roots where datasets may be located."""
    roots = [os.getcwd()]

    if is_running_in_colab():
        roots.extend([COLAB_CONTENT_DIR, COLAB_DRIVE_ROOT, COLAB_NOTEBOOKS_DIR])

    unique_roots = []
    seen = set()
    for root in roots:
        normalized = os.path.abspath(root)
        if normalized not in seen:
            seen.add(normalized)
            unique_roots.append(normalized)

    return unique_roots


def find_file_in_roots(filename: str, roots: List[str], max_depth: int = 4) -> Optional[str]:
    """Find file by name in roots with bounded recursive search depth."""
    basename = os.path.basename(filename)

    for root in roots:
        if not os.path.isdir(root):
            continue

        for current_root, dirs, files in os.walk(root):
            rel_path = os.path.relpath(current_root, root)
            depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
            if depth > max_depth:
                dirs[:] = []
                continue

            if basename in files:
                return os.path.join(current_root, basename)

    return None


def resolve_dataset_path(original_path: str, env_var_name: str) -> str:
    """Resolve dataset path with env override and Colab-aware fallbacks."""
    env_value = os.getenv(env_var_name)
    candidate_path = env_value.strip() if env_value else original_path

    # 1) Direct path as-is
    if os.path.exists(candidate_path):
        return os.path.abspath(candidate_path)

    # 2) Relative to CWD
    cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), candidate_path))
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    # 3) Colab-aware root checks
    roots = build_dataset_search_roots()
    for root in roots:
        for merged_candidate in (
            os.path.join(root, candidate_path),
            os.path.join(root, os.path.basename(candidate_path)),
        ):
            merged_abs = os.path.abspath(merged_candidate)
            if os.path.exists(merged_abs):
                return merged_abs

    # 4) Bounded recursive search by filename (useful after upload to Colab)
    recursive_match = find_file_in_roots(candidate_path, roots=roots, max_depth=4)
    if recursive_match is not None:
        return os.path.abspath(recursive_match)

    search_roots_message = ", ".join(roots)
    raise FileNotFoundError(
        "Dataset file not found. "
        f"Requested path='{candidate_path}', env var='{env_var_name}'. "
        f"Checked roots: {search_roots_message}. "
        "If you run in Colab, upload file to /content or set explicit env path."
    )


def initialize_runtime() -> None:
    """Initialize runtime paths and Colab-specific defaults once."""
    global _RUNTIME_INITIALIZED
    global MULTICLASS_DATA_PATH, BINARY_DATA_PATH
    global PIPELINE_OUTPUT_DIR, STAGE1_OUTPUT_DIR, STAGE2_OUTPUT_DIR, PIPELINE_E2E_ARTIFACTS_DIR

    if _RUNTIME_INITIALIZED:
        return

    in_colab = is_running_in_colab()
    if in_colab:
        logger.info("Google Colab runtime detected.")
        try_mount_colab_drive()

    output_dir_override = os.getenv(ENV_PIPELINE_OUTPUT_DIR)
    if output_dir_override:
        resolved_output_dir = os.path.abspath(output_dir_override)
    elif in_colab:
        resolved_output_dir = os.path.join(COLAB_CONTENT_DIR, PIPELINE_OUTPUT_DIR)
    else:
        resolved_output_dir = os.path.abspath(PIPELINE_OUTPUT_DIR)

    PIPELINE_OUTPUT_DIR = resolved_output_dir
    STAGE1_OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "stage1_binary")
    STAGE2_OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "stage2_intents")
    PIPELINE_E2E_ARTIFACTS_DIR = os.path.join(PIPELINE_OUTPUT_DIR, EVAL_ARTIFACTS_DIR, "two_stage_e2e")

    MULTICLASS_DATA_PATH = resolve_dataset_path(MULTICLASS_DATA_PATH, ENV_MULTICLASS_DATA_PATH)
    BINARY_DATA_PATH = resolve_dataset_path(BINARY_DATA_PATH, ENV_BINARY_DATA_PATH)

    os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)

    logger.info("Runtime initialized:")
    logger.info("  MULTICLASS_DATA_PATH=%s", MULTICLASS_DATA_PATH)
    logger.info("  BINARY_DATA_PATH=%s", BINARY_DATA_PATH)
    logger.info("  PIPELINE_OUTPUT_DIR=%s", PIPELINE_OUTPUT_DIR)

    _RUNTIME_INITIALIZED = True


# =========================
# 5. Reproducibility
# =========================
def set_global_seed(seed: int) -> None:
    """Set random seed across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 6. Generic CSV + Cleaning
# =========================
def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV with explicit validation and auto-delimiter detection."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    logger.info("Loading dataset from: %s", csv_path)
    return pd.read_csv(csv_path, sep=None, engine='python', on_bad_lines='skip')


def normalize_text(value: object) -> str:
    """Convert any value to a normalized text string."""
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() == "nan":
        return ""

    return text


def normalize_label_name(value: object) -> str:
    """Normalize label for robust comparisons."""
    return normalize_text(value).lower().replace("ё", "е")


def detect_column(df: pd.DataFrame, candidates: List[str], kind: str) -> str:
    """Detect column by candidate names, then fallback to a safe heuristic."""
    columns_map = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in columns_map:
            return columns_map[candidate_lower]

    if kind == "binary label":
        num_cols = df.select_dtypes(include=['number', 'int']).columns
        if len(num_cols) > 0:
            return num_cols[0]

    object_cols = [col for col in df.columns if df[col].dtype == "object"]
    if object_cols:
        logger.warning(
            "Column for %s was not found by name. Using fallback column: %s",
            kind,
            object_cols[0],
        )
        return object_cols[0]

    raise ValueError(
        f"Unable to detect {kind} column. Available columns: {list(df.columns)}"
    )


def split_stratified(df: pd.DataFrame, stratify_column: str = "label") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform stratified split with safety checks."""
    class_counts = df[stratify_column].value_counts()
    rare_classes = class_counts[class_counts < 2]
    if not rare_classes.empty:
        raise ValueError(
            "Stratified split failed: some classes have fewer than 2 samples. "
            f"Insufficient classes: {rare_classes.index.tolist()}"
        )

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[stratify_column],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# =========================
# 7. Stage-1 (Binary) Data
# =========================
def parse_binary_label(raw_label: object) -> int:
    """
    Parse binary label into {0: not trash, 1: trash}.
    Supports numbers and common text aliases.
    """
    normalized = normalize_label_name(raw_label)

    if normalized in BINARY_TRASH_VALUES:
        return 1
    if normalized in BINARY_NOT_TRASH_VALUES:
        return 0

    try:
        numeric = float(normalized)
        if numeric == 1.0:
            return 1
        if numeric == 0.0:
            return 0
    except ValueError:
        pass

    raise ValueError(
        "Unsupported binary label value: "
        f"'{raw_label}'. Supported trash aliases: {sorted(BINARY_TRASH_VALUES)}; "
        f"supported non-trash aliases: {sorted(BINARY_NOT_TRASH_VALUES)}"
    )


def load_and_prepare_binary_data(csv_path: str) -> PreparedData:
    """Load and prepare binary dataset for stage-1 trash filter."""
    df = load_csv(csv_path)

    text_col = detect_column(
        df,
        candidates=["text", TEXT_COLUMN, "query", "message", "utterance", "question_text"],
        kind="text",
    )
    label_col = detect_column(
        df,
        candidates=["is_question", "label", "target", "class", "is_trash", "binary_label", INTENT_LABEL_COLUMN],
        kind="binary label",
    )

    logger.info("Stage-1: Выбрана колонка '%s' для меток", label_col)

    local_df = df[[text_col, label_col]].copy()
    local_df = local_df.dropna(subset=[text_col, label_col])

    local_df["text"] = local_df[text_col].apply(normalize_text)
    local_df = local_df[local_df["text"] != ""].reset_index(drop=True)

    if label_col.lower() == "is_question":
        logger.info("Обнаружена колонка 'is_question'. Применяем инверсию (1 -> Не мусор/0, 0 -> Мусор/1)")
        def invert_logic(val):
            try:
                parsed = parse_binary_label(val)
                return 1 - parsed
            except ValueError:
                return -1
        local_df["label"] = local_df[label_col].apply(invert_logic)
    else:
        def safe_parse(val):
             try: return parse_binary_label(val)
             except ValueError: return -1
        local_df["label"] = local_df[label_col].apply(safe_parse)

    local_df = local_df[local_df["label"] != -1].reset_index(drop=True)

    id2label = {0: NOT_TRASH_LABEL_NAME, 1: TRASH_LABEL_NAME}
    label2id = {v: k for k, v in id2label.items()}

    unique_labels = sorted(local_df["label"].unique().tolist())
    if unique_labels != [0, 1]:
        raise ValueError(
            "Binary dataset must contain both classes 0 and 1. "
            f"Detected classes: {unique_labels}"
        )

    train_df, test_df = split_stratified(local_df[["text", "label"]])
    logger.info("Stage-1 dataset prepared | Train: %d | Test: %d", len(train_df), len(test_df))

    return PreparedData(
        train_df=train_df,
        test_df=test_df,
        label2id=label2id,
        id2label=id2label,
    )


# =========================
# 8. Multiclass Data Helpers
# =========================
def load_and_split_multiclass_full_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load full multiclass dataset including trash and create a common stratified split.
    This split is reused for stage-2 validation and end-to-end pipeline evaluation.
    """
    df = load_csv(csv_path)

    required_columns = {TEXT_COLUMN, INTENT_LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    local_df = df[[TEXT_COLUMN, INTENT_LABEL_COLUMN]].copy()
    local_df = local_df.dropna(subset=[TEXT_COLUMN, INTENT_LABEL_COLUMN])

    local_df["text"] = local_df[TEXT_COLUMN].apply(normalize_text)
    local_df["intent"] = local_df[INTENT_LABEL_COLUMN].apply(normalize_text)
    local_df = local_df[(local_df["text"] != "") & (local_df["intent"] != "")].reset_index(drop=True)

    if local_df.empty:
        raise ValueError("Multiclass dataset is empty after cleaning.")

    train_df, test_df = split_stratified(local_df[["text", "intent"]], stratify_column="intent")
    logger.info("Common multiclass split prepared | Train: %d | Test: %d", len(train_df), len(test_df))
    return train_df, test_df


def prepare_stage2_data_from_common_split(
    train_full_df: pd.DataFrame,
    test_full_df: pd.DataFrame,
    excluded_label: str,
) -> PreparedData:
    """Prepare stage-2 train/test from common split while excluding trash class."""
    excluded_label_normalized = normalize_label_name(excluded_label)

    train_df = train_full_df[
        train_full_df["intent"].apply(normalize_label_name) != excluded_label_normalized
    ].reset_index(drop=True)
    test_df = test_full_df[
        test_full_df["intent"].apply(normalize_label_name) != excluded_label_normalized
    ].reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Stage-2 train/test is empty after excluding trash label from common split."
        )

    all_non_trash_intents = pd.concat([train_df["intent"], test_df["intent"]], axis=0)
    label_names = sorted(all_non_trash_intents.unique().tolist())
    if len(label_names) < 2:
        raise ValueError(
            "Stage-2 requires at least 2 intent classes after removing trash. "
            f"Detected: {label_names}"
        )

    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}

    prepared_train = train_df[["text", "intent"]].copy()
    prepared_test = test_df[["text", "intent"]].copy()
    prepared_train["label"] = prepared_train["intent"].map(label2id).astype(int)
    prepared_test["label"] = prepared_test["intent"].map(label2id).astype(int)

    return PreparedData(
        train_df=prepared_train[["text", "label"]],
        test_df=prepared_test[["text", "label"]],
        label2id=label2id,
        id2label=id2label,
    )


# =========================
# 9. Dataset + Tokenization
# =========================
def to_hf_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
    """Convert pandas DataFrames to HuggingFace Datasets."""
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False)
    return train_dataset, test_dataset


def tokenize_datasets(
    train_dataset: Dataset,
    test_dataset: Dataset,
    model_name: str,
) -> Tuple[Dataset, Dataset, AutoTokenizer]:
    """Load tokenizer and tokenize train/test datasets."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_train = train_dataset.map(tokenize_batch, batched=True)
    tokenized_test = test_dataset.map(tokenize_batch, batched=True)

    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_test = tokenized_test.remove_columns(["text"])

    return tokenized_train, tokenized_test, tokenizer


# =========================
# 10. Metrics
# =========================
def build_metrics_fn(num_labels: int) -> Callable:
    """Build metrics function adapted for binary/multiclass classification."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        }

        if num_labels == 2:
            metrics["binary_f1"] = f1_score(
                labels,
                predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )

        return metrics

    return compute_metrics


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for numpy arrays."""
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted_logits)
    return exps / np.sum(exps, axis=1, keepdims=True)


# =========================
# 11. Validation Artifacts
# =========================
def save_validation_artifacts(
    trainer: Trainer,
    eval_dataset: Dataset,
    id2label: Dict[int, str],
    output_dir: str,
) -> None:
    """Generate and save validation reports and confusion matrices."""
    artifacts_dir = os.path.join(output_dir, EVAL_ARTIFACTS_DIR)
    os.makedirs(artifacts_dir, exist_ok=True)

    logger.info("Generating validation predictions for artifacts...")
    predictions_output = trainer.predict(eval_dataset)
    y_true = predictions_output.label_ids
    y_pred = np.argmax(predictions_output.predictions, axis=-1)

    label_ids = sorted(id2label.keys())
    target_names = [id2label[idx] for idx in label_ids]

    report_dict = classification_report(
        y_true, y_pred, labels=label_ids, target_names=target_names, output_dict=True, zero_division=0
    )

    with open(os.path.join(artifacts_dir, "classification_report.json"), "w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, ensure_ascii=False, indent=2)

    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(artifacts_dir, "classification_report.csv"), index=True)

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(os.path.join(artifacts_dir, "confusion_matrix.csv"), index=True)

    fig_width = max(8, len(target_names) * 0.9)
    fig_height = max(6, len(target_names) * 0.8)
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45, ha="right")
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(artifacts_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    logger.info("Validation artifacts saved to: %s", artifacts_dir)


def tune_stage1_threshold(
    trainer: Trainer,
    eval_dataset: Dataset,
    output_dir: str,
) -> float:
    """Find optimal stage-1 threshold by maximizing trash-class F1."""
    artifacts_dir = os.path.join(output_dir, EVAL_ARTIFACTS_DIR)
    os.makedirs(artifacts_dir, exist_ok=True)

    predictions_output = trainer.predict(eval_dataset)
    y_true = predictions_output.label_ids.astype(int)
    probabilities = softmax_numpy(predictions_output.predictions)
    trash_probabilities = probabilities[:, 1]

    thresholds = np.arange(STAGE1_THRESHOLD_MIN, STAGE1_THRESHOLD_MAX + STAGE1_THRESHOLD_STEP / 2, STAGE1_THRESHOLD_STEP)

    rows: List[Dict[str, float]] = []
    for threshold_value in thresholds:
        y_pred = (trash_probabilities >= threshold_value).astype(int)
        row = {
            "threshold": float(round(threshold_value, 4)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "trash_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        }
        rows.append(row)

    best_threshold = float(max(rows, key=lambda x: x["trash_f1"])["threshold"])

    curve_df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 6))
    plt.plot(curve_df["threshold"], curve_df["trash_f1"], label="Trash F1", linewidth=2)
    plt.axvline(best_threshold, color="red", linestyle="-.", label=f"Best threshold={best_threshold:.2f}")
    plt.title("Stage-1 threshold tuning")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(artifacts_dir, STAGE1_THRESHOLD_CURVE_PNG), dpi=300)
    plt.close()

    with open(os.path.join(output_dir, STAGE1_THRESHOLD_FILE_NAME), "w", encoding="utf-8") as fp:
        json.dump({"selected_threshold": best_threshold}, fp, ensure_ascii=False, indent=2)

    logger.info("Stage-1 threshold selected: %.4f", best_threshold)
    return best_threshold


# =========================
# 12. Generic Stage Trainer
# =========================
def train_stage(
    stage_name: str,
    prepared_data: PreparedData,
    output_dir: str,
) -> StageTrainingResult:
    """Train one stage (binary or multiclass), evaluate, and save artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    train_dataset, test_dataset = to_hf_dataset(prepared_data.train_df, prepared_data.test_df)
    tokenized_train, tokenized_test, tokenizer = tokenize_datasets(train_dataset, test_dataset, MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(prepared_data.label2id),
        id2label=prepared_data.id2label,
        label2id=prepared_data.label2id,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        seed=RANDOM_SEED,
        report_to="none",
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=build_metrics_fn(num_labels=len(prepared_data.label2id)),
        callbacks=[early_stopping_callback],
    )

    logger.info("[%s] Starting training...", stage_name)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    save_validation_artifacts(trainer, tokenized_test, prepared_data.id2label, output_dir)

    metadata_payload = {
        "label2id": prepared_data.label2id,
        "id2label": {str(k): v for k, v in prepared_data.id2label.items()},
    }
    with open(os.path.join(output_dir, "labels_metadata.json"), "w", encoding="utf-8") as fp:
        json.dump(metadata_payload, fp, ensure_ascii=False, indent=2)

    return StageTrainingResult(trainer, tokenizer, tokenized_test, prepared_data)


# =========================
# 13. Inference (Lazy Loaded)
# =========================
def _load_stage1_threshold() -> float:
    """Load tuned stage-1 threshold from file."""
    threshold_path = os.path.join(STAGE1_OUTPUT_DIR, STAGE1_THRESHOLD_FILE_NAME)
    if not os.path.exists(threshold_path):
        return STAGE1_THRESHOLD_DEFAULT
    with open(threshold_path, "r", encoding="utf-8") as fp:
        return float(json.load(fp).get("selected_threshold", STAGE1_THRESHOLD_DEFAULT))


def _ensure_stage1_loaded() -> None:
    global _STAGE1_MODEL, _STAGE1_TOKENIZER, _STAGE1_THRESHOLD
    if _STAGE1_MODEL is not None: return
    _STAGE1_TOKENIZER = AutoTokenizer.from_pretrained(STAGE1_OUTPUT_DIR)
    _STAGE1_MODEL = AutoModelForSequenceClassification.from_pretrained(STAGE1_OUTPUT_DIR).eval()
    _STAGE1_THRESHOLD = _load_stage1_threshold()


def _ensure_stage2_loaded() -> None:
    global _STAGE2_MODEL, _STAGE2_TOKENIZER, _STAGE2_ID2LABEL
    if _STAGE2_MODEL is not None: return
    _STAGE2_TOKENIZER = AutoTokenizer.from_pretrained(STAGE2_OUTPUT_DIR)
    _STAGE2_MODEL = AutoModelForSequenceClassification.from_pretrained(STAGE2_OUTPUT_DIR).eval()
    with open(os.path.join(STAGE2_OUTPUT_DIR, "labels_metadata.json"), "r") as f:
        _STAGE2_ID2LABEL = {int(k): v for k, v in json.load(f)["id2label"].items()}


def _predict_stage1_trash_probability_batch(texts: List[str]) -> np.ndarray:
    """Predict stage-1 P(trash) for a batch of texts."""
    _ensure_stage1_loaded()

    if not texts:
        return np.array([], dtype=float)

    assert _STAGE1_MODEL is not None
    assert _STAGE1_TOKENIZER is not None

    encoded = _STAGE1_TOKENIZER(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = _STAGE1_MODEL(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

    return probabilities[:, 1].astype(float)


def _predict_stage2_batch(texts: List[str]) -> Tuple[List[str], List[float]]:
    """Predict stage-2 intent labels and confidences for a batch of non-trash texts."""
    _ensure_stage2_loaded()

    if not texts:
        return [], []

    assert _STAGE2_MODEL is not None
    assert _STAGE2_TOKENIZER is not None
    assert _STAGE2_ID2LABEL is not None

    encoded = _STAGE2_TOKENIZER(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = _STAGE2_MODEL(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

    pred_ids = np.argmax(probabilities, axis=-1)
    pred_conf = np.max(probabilities, axis=-1)

    labels = [_STAGE2_ID2LABEL[int(label_id)] for label_id in pred_ids]
    confidences = [float(conf_value) for conf_value in pred_conf]
    return labels, confidences


def predict_two_stage(text: str) -> Dict[str, object]:
    """Inference logic: Stage 1 (Trash) -> Stage 2 (Intent)."""
    normalized_text = normalize_text(text)
    if normalized_text == "":
        raise ValueError("Input text must be non-empty.")

    trash_probability = float(_predict_stage1_trash_probability_batch([normalized_text])[0])

    _ensure_stage1_loaded()
    assert _STAGE1_THRESHOLD is not None

    is_trash = trash_probability >= _STAGE1_THRESHOLD
    if is_trash:
        return {
            "text": normalized_text,
            "route": "stage1_trash",
            "label": TRASH_LABEL_NAME,
            "confidence": trash_probability,
            "stage1": {
                "trash_probability": trash_probability,
                "threshold": _STAGE1_THRESHOLD,
                "is_trash": True,
            },
        }

    stage2_labels, stage2_confidences = _predict_stage2_batch([normalized_text])
    return {
        "text": normalized_text,
        "route": "stage2_intent",
        "label": stage2_labels[0],
        "confidence": stage2_confidences[0],
        "stage1": {
            "trash_probability": trash_probability,
            "threshold": _STAGE1_THRESHOLD,
            "is_trash": False,
        },
    }


# =========================
# 14. End-to-End Evaluation
# =========================
def evaluate_two_stage_pipeline(common_test_df: pd.DataFrame, output_dir: str):
    """Full chain test."""
    os.makedirs(output_dir, exist_ok=True)
    y_true = common_test_df["intent"].tolist()
    y_pred = [predict_two_stage(t)["label"] for t in common_test_df["text"]]

    acc = accuracy_score(y_true, y_pred)
    logger.info("E2E Pipeline Accuracy: %.4f", acc)
    with open(os.path.join(output_dir, "e2e_metrics.json"), "w") as f:
        json.dump({"accuracy": acc}, f)


# =========================
# 15. Two-Stage Pipeline
# =========================
def train_two_stage_pipeline() -> None:
    """Full execution logic."""
    initialize_runtime()
    set_global_seed(RANDOM_SEED)

    # Stage 1
    binary_data = load_and_prepare_binary_data(BINARY_DATA_PATH)
    stage1_res = train_stage("stage1_binary", binary_data, STAGE1_OUTPUT_DIR)
    tune_stage1_threshold(stage1_res.trainer, stage1_res.tokenized_test, STAGE1_OUTPUT_DIR)

    # Stage 2
    full_train, full_test = load_and_split_multiclass_full_data(MULTICLASS_DATA_PATH)
    stage2_data = prepare_stage2_data_from_common_split(full_train, full_test, TRASH_LABEL_NAME)
    train_stage("stage2_intents", stage2_data, STAGE2_OUTPUT_DIR)

    # E2E Test
    evaluate_two_stage_pipeline(full_test, PIPELINE_E2E_ARTIFACTS_DIR)
    logger.info("Пайплайн обучен и проверен!")

# =========================
# 16. Tests (Smoke, Regression, E2E)
# =========================
class PipelineTests(unittest.TestCase):

    # --- SMOKE TESTS ---
    def test_normalize_text_handles_empty_and_nan_values(self):
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("   "), "")
        self.assertEqual(normalize_text("NaN"), "")
        self.assertEqual(normalize_text("  Привет  "), "Привет")

    def test_parse_binary_label_supports_common_aliases(self):
        self.assertEqual(parse_binary_label("Мусор"), 1)
        self.assertEqual(parse_binary_label("spam"), 1)
        self.assertEqual(parse_binary_label("1"), 1)
        self.assertEqual(parse_binary_label("Не мусор"), 0)
        self.assertEqual(parse_binary_label("ham"), 0)
        self.assertEqual(parse_binary_label("0"), 0)

    def test_parse_binary_label_raises_on_unknown_values(self):
        for bad_value in ["unknown", "2", "", "нет данных"]:
            with self.assertRaises(ValueError):
                parse_binary_label(bad_value)

    def test_split_stratified_raises_for_rare_classes(self):
        data = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 0, 1]})
        with self.assertRaisesRegex(ValueError, "fewer than 2 samples"):
            split_stratified(data, stratify_column="label")

    # --- REGRESSION TESTS ---
    def test_prepare_stage2_data_from_common_split_excludes_trash(self):
        train_full_df = pd.DataFrame({"text": ["t1", "t2", "t3", "t4"], "intent": ["Погода", "Мусор", "Музыка", "Погода"]})
        test_full_df = pd.DataFrame({"text": ["q1", "q2", "q3"], "intent": ["Музыка", "Мусор", "Погода"]})

        prepared = prepare_stage2_data_from_common_split(train_full_df, test_full_df, TRASH_LABEL_NAME)

        self.assertEqual(set(prepared.label2id.keys()), {"Музыка", "Погода"})
        self.assertTrue(prepared.train_df["label"].isin(prepared.id2label.keys()).all())
        self.assertTrue(prepared.test_df["label"].isin(prepared.id2label.keys()).all())

    # --- E2E TESTS (MOCKED) ---
    @patch("__main__._predict_stage1_trash_probability_batch", return_value=[0.85])
    @patch("__main__._ensure_stage1_loaded")
    def test_predict_two_stage_routes_to_trash(self, mock_ensure, mock_predict1):
        global _STAGE1_THRESHOLD
        _STAGE1_THRESHOLD = 0.3

        result = predict_two_stage("спам сообщение")

        self.assertEqual(result["route"], "stage1_trash")
        self.assertEqual(result["label"], TRASH_LABEL_NAME)
        self.assertTrue(result["stage1"]["is_trash"])
        self.assertEqual(result["stage1"]["threshold"], 0.3)
        self.assertAlmostEqual(result["confidence"], 0.85)

    @patch("__main__._predict_stage2_batch", return_value=(["Погода"], [0.91]))
    @patch("__main__._predict_stage1_trash_probability_batch", return_value=[0.2])
    @patch("__main__._ensure_stage1_loaded")
    def test_predict_two_stage_routes_to_stage2(self, mock_ensure, mock_predict1, mock_predict2):
        global _STAGE1_THRESHOLD
        _STAGE1_THRESHOLD = 0.5

        result = predict_two_stage("какая сегодня погода")

        self.assertEqual(result["route"], "stage2_intent")
        self.assertEqual(result["label"], "Погода")
        self.assertFalse(result["stage1"]["is_trash"])
        self.assertEqual(result["stage1"]["threshold"], 0.5)
        self.assertAlmostEqual(result["confidence"], 0.91)

def run_all_tests():
    """Runs all embedded tests before pipeline execution."""
    logger.info("Running embedded unit and E2E tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(PipelineTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Tests failed! Aborting pipeline execution.")
    logger.info("All tests passed successfully! Proceeding to training...")


# =========================
# 17. Entrypoint
# =========================
if __name__ == "__main__":
    # Сначала прогоняем тесты
    run_all_tests()
    # Если тесты пройдены, запускаем обучение
    train_two_stage_pipeline()