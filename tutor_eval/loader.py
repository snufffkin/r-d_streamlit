"""Load xlsx dialog files into structured data."""

import re
from dataclasses import dataclass
from pathlib import Path

import openpyxl


@dataclass
class Dialog:
    """A single tutor-student dialog with metadata."""

    dialog_id: str
    text: str
    task: str
    task_id: str
    file_name: str
    student_type: str  # weak / medium / otlichnik
    student_model: str  # gemini3flash / glm45 / deepseekV31Terminus / gemini25falsh
    grade_group: str
    theme: str
    subtheme: str
    skill: str


_STUDENT_TYPES = ["weak", "medium", "otlichnik"]
_STUDENT_MODELS = [
    "gemini3flash",
    "gemini25falsh",
    "glm45",
    "deepseekV31Terminus",
    "aliceai235b",
]


def _parse_filename(filename: str) -> tuple[str, str]:
    """Extract student_type and student_model from xlsx filename."""
    clean = re.sub(r"^\d+\)\s*", "", filename)

    student_type = "unknown"
    student_model = "unknown"

    for st in _STUDENT_TYPES:
        if f"_{st}_" in clean:
            student_type = st
            break

    # Try known models first
    for sm in _STUDENT_MODELS:
        if f"_{sm}" in clean:
            student_model = sm
            break

    # Fallback: extract everything between _{type}_ and .xlsx
    if student_model == "unknown" and student_type != "unknown":
        suffix = clean.split(f"_{student_type}_", 1)[-1]
        suffix = re.sub(r"(_rendered)?\.xlsx$", "", suffix)
        if suffix:
            student_model = suffix

    return student_type, student_model


def load_xlsx(filepath: str | Path) -> list[Dialog]:
    """Load dialogs from a single xlsx file."""
    filepath = Path(filepath)
    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb.active

    rows = list(ws.rows)
    if len(rows) < 2:
        wb.close()
        return []

    headers = [cell.value for cell in rows[0]]
    col = {name: idx for idx, name in enumerate(headers) if name}

    # Detect column format: old (dialog, task_id, grade_group, theme_0_name)
    # vs new (dialog_render, request_id, class, theme-0)
    _COL_ALIASES = {
        "dialog": ["dialog", "dialog_render"],
        "task": ["task"],
        "task_id": ["task_id", "request_id"],
        "grade_group": ["grade_group", "class"],
        "theme_0_name": ["theme_0_name", "theme-0"],
        "theme_1_name": ["theme_1_name"],
        "theme_2_name": ["theme_2_name"],
    }

    def _resolve(alias_key: str) -> str | None:
        for candidate in _COL_ALIASES[alias_key]:
            if candidate in col:
                return candidate
        return None

    dialog_col = _resolve("dialog")
    task_col = _resolve("task")
    task_id_col = _resolve("task_id")

    if not dialog_col or not task_col:
        wb.close()
        if len(rows) == 1:
            return []
        raise ValueError(
            f"Missing required columns in {filepath.name}: "
            f"need dialog/dialog_render and task, got {list(col.keys())}"
        )

    _type_keywords = {"string", "int64", "float64", "any", "bool", "object"}
    row2_values = {str(cell.value).strip().lower() for cell in rows[1] if cell.value}
    has_type_row = row2_values.issubset(_type_keywords) and len(row2_values) > 0

    data_start = 2 if has_type_row else 1

    student_type, student_model = _parse_filename(filepath.name)
    grade_col = _resolve("grade_group")
    theme0_col = _resolve("theme_0_name")
    theme1_col = _resolve("theme_1_name")
    theme2_col = _resolve("theme_2_name")
    dialogs = []

    for row_idx, row in enumerate(rows[data_start:], start=1):
        def val(column_name: str | None, default: str = "") -> str:
            if column_name is None:
                return default
            idx = col.get(column_name)
            if idx is None or idx >= len(row):
                return default
            v = row[idx].value
            return str(v).strip() if v is not None else default

        dialog_text = val(dialog_col)
        task_text = val(task_col)
        task_id = val(task_id_col, default=str(row_idx))

        if not dialog_text or not task_text:
            continue

        dialog_id = f"{student_type}_{student_model}_{task_id}"

        dialogs.append(Dialog(
            dialog_id=dialog_id,
            text=dialog_text,
            task=task_text,
            task_id=task_id,
            file_name=filepath.name,
            student_type=student_type,
            student_model=student_model,
            grade_group=val(grade_col),
            theme=val(theme0_col),
            subtheme=val(theme1_col),
            skill=val(theme2_col),
        ))

    wb.close()
    return dialogs


def load_all(data_dir: str | Path) -> list[Dialog]:
    """Load dialogs from all xlsx files in a directory."""
    data_dir = Path(data_dir)
    all_dialogs = []
    files = sorted(data_dir.glob("*.xlsx"))

    for f in files:
        try:
            dialogs = load_xlsx(f)
            print(f"  {f.name}: {len(dialogs)} dialogs")
            all_dialogs.extend(dialogs)
        except Exception as e:
            print(f"  ERROR loading {f.name}: {e}")

    print(f"Total: {len(all_dialogs)} dialogs from {len(files)} files")
    return all_dialogs
