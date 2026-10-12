"""Bug 6: split must be temporal AND uploader-disjoint, and test must be strictly newer than train."""
import pandas as pd

from cleaning_and_split import CleaningReport, temporal_group_split


def make_df():
    rows = []
    heavy_days = {0, 1, 2, 97, 98, 99}                 # one uploader active at both ends of the timeline
    for i in range(100):
        uploader = "HEAVY" if i in heavy_days else f"u{i // 5}"
        rows.append({"timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i),
                     "uploader": uploader, "categories": [["A"]][0], "id": i})
    return pd.DataFrame(rows)


def test_uploaders_are_disjoint_across_splits():
    train, val, test = temporal_group_split(make_df(), ratios=(0.7, 0.15, 0.15))
    assert set(train.uploader).isdisjoint(val.uploader)
    assert set(train.uploader).isdisjoint(test.uploader)
    assert set(val.uploader).isdisjoint(test.uploader)


def test_split_is_strictly_temporal():
    train, val, test = temporal_group_split(make_df(), ratios=(0.7, 0.15, 0.15))
    assert len(train) and len(val) and len(test)
    assert train.timestamp.max() < val.timestamp.min()
    assert val.timestamp.max() < test.timestamp.min()
    # explicit statement of the requirement: no test image older than the latest train image
    assert (test.timestamp > train.timestamp.max()).all()


def test_dropped_rows_are_counted_and_nothing_is_duplicated():
    report = CleaningReport()
    df = make_df()
    train, val, test = temporal_group_split(df, ratios=(0.7, 0.15, 0.15), report=report)
    kept = pd.concat([train, val, test])
    assert kept.id.is_unique
    assert report.temporal_dropped == len(df) - len(kept)
    assert report.temporal_dropped > 0                 # the heavy uploader's out-of-window images
