# -*- coding: utf-8 -*-
from app.services.input_parser import build_raw_sample, parse_omics_csv


def test_wide_single_row_and_same_patient():
    content = b"patient_id,G1,G2\nP001,1.2,3.4\n"
    ge = parse_omics_csv(content, "ge.csv", "expression")
    mut = parse_omics_csv(content, "mut.csv", "mutation")
    meth = parse_omics_csv(content, "meth.csv", "methylation")
    sample = build_raw_sample(ge, mut, meth, None)
    assert sample.patient_id == "P001"
    assert list(sample.expression.values.index) == ["G1", "G2"]


def test_long_format():
    content = "feature,value\nG1,1\nG2,2\n".encode("utf-8")
    parsed = parse_omics_csv(content, "ge.csv", "expression", "P002")
    assert parsed.sample_id == "P002"
    assert parsed.orientation == "long_feature_value"


def test_two_column_wide_with_patient_id_column():
    content = b"patient_id,G1\nP003,4.2\n"
    parsed = parse_omics_csv(content, "ge.csv", "expression")
    assert parsed.sample_id == "P003"
    assert parsed.values["G1"] == 4.2
