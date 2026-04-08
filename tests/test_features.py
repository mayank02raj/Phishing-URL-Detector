"""tests/test_features.py"""

import pytest

from ml.features import extract_features, feature_names, shannon_entropy


def test_feature_count_stable():
    assert len(feature_names()) == len(extract_features("http://example.com/"))


def test_features_are_numeric():
    f = extract_features("http://example.com/path?q=1")
    for name, val in f.items():
        assert isinstance(val, (int, float)), f"{name} is not numeric"


def test_handles_empty_url():
    f = extract_features("")
    assert f["url_length"] == len("http://")  # gets prepended


def test_handles_malformed_url():
    f = extract_features("http://[not-a-valid-host")
    assert isinstance(f["url_length"], int)


def test_ip_literal_detection():
    assert extract_features("http://192.168.1.1/admin")["is_ip"] == 1
    assert extract_features("http://example.com/")["is_ip"] == 0


def test_suspicious_tld():
    assert extract_features("http://login.tk/x")["is_suspicious_tld"] == 1
    assert extract_features("http://login.com/x")["is_suspicious_tld"] == 0


def test_brand_outside_domain():
    """paypal in subdomain of an unrelated domain should fire."""
    f = extract_features("http://paypal.evil-tk.tk/login")
    assert f["brand_in_subdomain"] == 1
    assert f["brand_outside_domain"] == 1


def test_brand_in_legit_domain_does_not_fire_outside():
    f = extract_features("http://www.paypal.com/login")
    assert f["brand_in_subdomain"] == 0


def test_https_flag():
    assert extract_features("https://example.com/")["uses_https"] == 1
    assert extract_features("http://example.com/")["uses_https"] == 0


def test_entropy_higher_for_random():
    low = shannon_entropy("aaaaaaaaaa")
    high = shannon_entropy("a3f9k2xq8z")
    assert high > low


def test_shortener_detection():
    assert extract_features("http://bit.ly/abc")["is_shortener"] == 1
    assert extract_features("http://google.com/")["is_shortener"] == 0


@pytest.mark.parametrize("url", [
    "http://paypa1-secure.tk/login",
    "http://login.microsoft.update.cf/auth",
    "http://192.168.1.1:8080/admin/wp-login.php",
    "https://www.google.com/",
    "https://docs.python.org/3/library/json.html",
])
def test_extraction_does_not_throw(url):
    f = extract_features(url)
    assert len(f) == len(feature_names())
