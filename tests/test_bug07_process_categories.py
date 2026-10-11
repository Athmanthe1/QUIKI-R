"""Bug 7: the maintenance regex dropped real categories and missed real process categories."""
from cleaning_and_split import CleaningReport, _clean_category_list


def clean(cats, hidden=None):
    report = CleaningReport()
    kwargs = {} if hidden is None else {"hidden_categories": hidden}
    return _clean_category_list(cats, report, **kwargs), report


def test_real_content_categories_are_kept():
    cats = ["Aircraft maintenance", "Beach cleanup", "Category:Aircraft_maintenance_hangars", "Beach cleanups"]
    out, report = clean(cats)
    assert out == cats
    assert report.maintenance_categories_removed == 0


def test_real_process_categories_are_dropped_by_fallback():
    process = [
        "Uploaded with UploadWizard",
        "Category:Uploaded_via_Campaign:wlm-in",
        "Media needing categories as of 12 May 2020",
        "Self-published work",
        "CC-BY-SA-4.0",
        "Files with no machine-readable author",
        "Media missing infobox template",
    ]
    out, report = clean(process + ["Mountains"])
    assert out == ["Mountains"]
    assert report.maintenance_categories_removed == len(process)


def test_hidden_flag_is_preferred_over_regex():
    # Name looks like content, but the dump says it is hidden => drop.
    out, _ = clean(["Odd internal bucket", "Aircraft maintenance"], hidden={"Odd internal bucket"})
    assert out == ["Aircraft maintenance"]


def test_hidden_flag_given_means_regex_is_not_used():
    # With dump flags present the regex must not second-guess them.
    out, _ = clean(["Uploaded with UploadWizard"], hidden=set())
    assert out == ["Uploaded with UploadWizard"]


def test_hidden_flag_matching_ignores_prefix_underscores_and_case():
    out, _ = clean(["Category:Odd_internal_bucket"], hidden={"odd internal bucket"})
    assert out == []
