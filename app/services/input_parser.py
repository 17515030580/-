# -*- coding: utf-8 -*-
from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from app.core.exceptions import InputValidationError
from app.schemas.domain import ParsedOmics, RawOmicsSample

_ID_NAMES = {
    "sample_id", "patient_id", "modelid", "model_id", "sample", "patient",
    "id", "case_id", "subject_id",
}
_FEATURE_NAMES = {"feature", "feature_name", "gene", "gene_name", "probe", "marker"}
_VALUE_NAMES = {"value", "expression", "mutation", "methylation", "score", "beta"}


def _decode_csv(content: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = content.decode(encoding)
            return pd.read_csv(io.StringIO(text))
        except Exception as exc:
            last_error = exc
    raise InputValidationError(
        "CSV文件无法解析，请使用UTF-8编码并检查分隔符。",
        {"parser_error": str(last_error)},
    )


def _norm(value) -> str:
    return str(value).strip()


def _numeric_series(series: pd.Series, filename: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    invalid_mask = converted.isna() & ~series.isna()
    if invalid_mask.any():
        bad = series[invalid_mask].astype(str).head(10).tolist()
        raise InputValidationError(
            "组学文件包含无法转换为数值的特征值。",
            {"filename": filename, "examples": bad},
        )
    if converted.isna().any():
        missing_count = int(converted.isna().sum())
        raise InputValidationError(
            "组学文件包含缺失值。",
            {"filename": filename, "missing_value_count": missing_count},
        )
    return converted.astype(float)


def _parse_long(df: pd.DataFrame, filename: str) -> Optional[Tuple[pd.Series, str]]:
    if df.shape[1] != 2:
        return None
    first, second = df.columns[:2]
    first_name = _norm(first).lower()
    second_name = _norm(second).lower()
    if first_name in _ID_NAMES:
        return None
    likely_long = (
        first_name in _FEATURE_NAMES
        or second_name in _VALUE_NAMES
        or not pd.api.types.is_numeric_dtype(df[first])
    )
    if not likely_long:
        return None
    features = df[first].astype(str).str.strip()
    if features.duplicated().any():
        duplicates = features[features.duplicated()].head(10).tolist()
        raise InputValidationError(
            "组学文件包含重复特征。",
            {"filename": filename, "duplicates": duplicates},
        )
    values = _numeric_series(df[second], filename)
    values.index = features
    return values, "long_feature_value"


def parse_omics_csv(
    content: bytes,
    filename: str,
    modality: str,
    requested_patient_id: Optional[str] = None,
) -> ParsedOmics:
    df = _decode_csv(content)
    if df.empty:
        raise InputValidationError("组学CSV没有数据行。", {"filename": filename})
    df.columns = [_norm(column) for column in df.columns]

    long_result = _parse_long(df, filename)
    if long_result is not None:
        values, orientation = long_result
        return ParsedOmics(
            modality=modality,
            sample_id=requested_patient_id,
            values=values,
            source_filename=filename,
            orientation=orientation,
        )

    id_columns = [column for column in df.columns if column.lower() in _ID_NAMES]
    id_column = id_columns[0] if id_columns else None

    # Handle pandas index exported as "Unnamed: 0".
    unnamed = [column for column in df.columns if column.lower().startswith("unnamed:")]
    if id_column is None and unnamed:
        candidate = unnamed[0]
        if not pd.api.types.is_numeric_dtype(df[candidate]):
            id_column = candidate
        elif df[candidate].tolist() == list(range(len(df))):
            df = df.drop(columns=[candidate])

    sample_id = requested_patient_id
    if id_column is not None:
        ids = df[id_column].astype(str).str.strip()
        if requested_patient_id:
            matched = df[ids == requested_patient_id]
            if matched.empty:
                raise InputValidationError(
                    "指定patient_id未在组学文件中找到。",
                    {"filename": filename, "patient_id": requested_patient_id},
                )
            if len(matched) > 1:
                raise InputValidationError(
                    "同一patient_id在组学文件中出现多次。",
                    {"filename": filename, "patient_id": requested_patient_id},
                )
            row = matched.iloc[0]
            sample_id = requested_patient_id
        elif len(df) == 1:
            row = df.iloc[0]
            sample_id = ids.iloc[0]
        else:
            raise InputValidationError(
                "组学文件包含多个样本，请在表单中提供patient_id。",
                {"filename": filename, "sample_count": int(len(df))},
            )
        feature_columns = [column for column in df.columns if column != id_column]
        values = _numeric_series(row[feature_columns], filename)
        values.index = feature_columns
        orientation = "wide_sample_row_with_id"
    elif len(df) == 1:
        row = df.iloc[0]
        values = _numeric_series(row, filename)
        values.index = df.columns
        orientation = "wide_single_sample_row"
    elif df.shape[1] == 2:
        # This path is normally handled by _parse_long, retained for clarity.
        features = df.iloc[:, 0].astype(str).str.strip()
        values = _numeric_series(df.iloc[:, 1], filename)
        values.index = features
        orientation = "transposed_single_sample"
    else:
        raise InputValidationError(
            "无法自动识别组学CSV方向。建议使用一行一个样本、列名为特征名的宽表。",
            {"filename": filename, "rows": int(df.shape[0]), "columns": int(df.shape[1])},
        )

    values.index = pd.Index([_norm(item) for item in values.index])
    if values.index.duplicated().any():
        duplicates = values.index[values.index.duplicated()].unique().tolist()[:10]
        raise InputValidationError(
            "组学文件包含重复特征名。",
            {"filename": filename, "duplicates": duplicates},
        )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise InputValidationError(
            "组学文件包含NaN或无穷值。", {"filename": filename}
        )

    return ParsedOmics(
        modality=modality,
        sample_id=sample_id,
        values=values,
        source_filename=filename,
        orientation=orientation,
    )


def build_raw_sample(
    expression: ParsedOmics,
    mutation: ParsedOmics,
    methylation: ParsedOmics,
    requested_patient_id: Optional[str],
) -> RawOmicsSample:
    ids = [item.sample_id for item in (expression, mutation, methylation) if item.sample_id]
    unique_ids = sorted(set(ids))
    if requested_patient_id:
        patient_id = requested_patient_id
        mismatched = [item for item in unique_ids if item != patient_id]
        if mismatched:
            raise InputValidationError(
                "三个组学文件中的样本ID与patient_id不一致。",
                {"patient_id": patient_id, "file_sample_ids": unique_ids},
            )
    elif len(unique_ids) == 1:
        patient_id = unique_ids[0]
    elif len(unique_ids) > 1:
        raise InputValidationError(
            "三个组学文件中的样本ID不一致。", {"sample_ids": unique_ids}
        )
    else:
        raise InputValidationError(
            "文件中未提供样本ID，请在multipart表单中提供patient_id。"
        )

    return RawOmicsSample(
        patient_id=patient_id,
        expression=expression,
        mutation=mutation,
        methylation=methylation,
    )
