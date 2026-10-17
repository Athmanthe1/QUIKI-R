"""A tiny stand-in for the parts of Streamlit that app.py uses, so main() can be executed (and its output
inspected) in tests without Streamlit installed. It is NOT a UI test: layout and widgets are not rendered."""
import contextlib
import sys
import types


class _SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class _Ctx:
    calls = None                                  # set by install(): shared recorder for message-like calls

    def metric(self, label, value, **k):
        if _Ctx.calls is not None:
            _Ctx.calls.append(("metric", f"{label}={value}"))

    def caption(self, text, **k):
        if _Ctx.calls is not None:
            _Ctx.calls.append(("caption", str(text)))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):                 # any widget method on a column / container object
        return lambda *a, **k: None


def install(upload=None):
    st = types.ModuleType("streamlit")
    st.session_state = _SessionState()
    st.calls = []                                 # (kind, text) for every message-like call
    _Ctx.calls = st.calls
    st.state = {"upload": upload, "clicks": set()}

    def rec(kind):
        def f(*a, **k):
            st.calls.append((kind, str(a[0]) if a else ""))
        return f

    for kind in ("title", "caption", "success", "warning", "error", "info", "markdown", "subheader", "toast", "code", "text"):
        setattr(st, kind, rec(kind))
    for name in ("set_page_config", "divider", "write", "image", "map", "dataframe", "line_chart", "metric", "progress",
                 "download_button", "rerun"):
        setattr(st, name, lambda *a, **k: None)
    st.metric = lambda label, value, **k: st.calls.append(("metric", f"{label}={value}"))
    st.container = lambda **k: _Ctx()
    st.expander = lambda *a, **k: (st.calls.append(("expander", str(a[0])))) or _Ctx()
    st.spinner = lambda *a, **k: _Ctx()
    st.columns = lambda spec, **k: [_Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.file_uploader = lambda *a, **k: st.state["upload"]
    st.text_input = lambda *a, **k: ""
    st.text_area = lambda *a, **k: ""
    st.slider = lambda label, min_value=0, max_value=0, value=0, **k: value
    st.number_input = lambda label, value=0, **k: value
    st.radio = lambda label, options, key=None, **k: st.session_state.setdefault(key, options[0])
    st.button = lambda label, key=None, on_click=None, args=(), **k: (key in st.state["clicks"])

    class _Sidebar(_Ctx):
        def expander(self, *a, **k):
            return _Ctx()

        def subheader(self, *a, **k):
            return None

        def divider(self):
            return None

        def metric(self, label, value, **k):
            st.calls.append(("metric", f"{label}={value}"))

    st.sidebar = _Sidebar()

    def cache_resource(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f
    st.cache_resource = cache_resource
    sys.modules["streamlit"] = st
    return st


class FakeUpload:
    type = "image/png"

    def __init__(self, data):
        self._d = data

    def getvalue(self):
        return self._d
