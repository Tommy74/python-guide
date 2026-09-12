"""Part 0 – Structuring a Python Project."""

PART = "Structuring a Python Project"

CHAPTERS = [
    {
        "title": "Project Layout, Virtual Environments and pytest",
        "blocks": [
            # ---- overview ----
            {"t": "p", "text":
                "Before writing a single line of Python it pays to give your "
                "project a predictable layout. A consistent structure makes "
                "imports reliable, keeps test discovery automatic, and means "
                "anyone who clones the repository already knows where "
                "everything lives."},

            # ---- folder layout ----
            {"t": "h", "text": "Recommended folder layout"},
            {"t": "p", "text":
                "The convention used throughout this guide separates "
                "production code from tests right from the start:"},
            {"t": "bullets", "items": [
                "`src/` — all importable Python modules and packages live "
                "here. Keeping code under `src/` prevents accidental "
                "imports of the raw source tree before the package is "
                "installed.",
                "`tests/` — every test file starts with `test_` so pytest "
                "discovers it automatically. Test files import from `src/` "
                "the same way a real consumer would.",
                "`pyproject.toml` — single configuration file for the build "
                "system, pytest, linters and type-checkers.",
                "`.venv/` — the virtual environment directory (never "
                "committed to git; add it to `.gitignore`).",
            ]},
            {"t": "p", "text":
                "A minimal project therefore looks like this:"},
            {"t": "code", "run": False, "caption": "Typical project tree",
             "code":
                "my_project/\n"
                "├── src/\n"
                "│   ├── calculator.py\n"
                "│   └── utils.py\n"
                "├── tests/\n"
                "│   ├── test_calculator.py\n"
                "│   └── test_utils.py\n"
                "├── pyproject.toml\n"
                "├── requirements.txt   # optional – pinned deps\n"
                "└── .gitignore"},

            # ---- virtual env ----
            {"t": "h", "text": "Creating and activating a virtual environment"},
            {"t": "p", "text":
                "A **virtual environment** is an isolated Python installation "
                "that keeps the project's dependencies separate from every "
                "other project and from the system Python. Always create one "
                "before installing any packages."},
            {"t": "code", "run": False, "caption": "Create and activate .venv",
             "code":
                "# Create the environment (run once, from the project root)\n"
                "python3 -m venv .venv\n\n"
                "# Activate on Linux / macOS\n"
                "source .venv/bin/activate\n\n"
                "# Activate on Windows (PowerShell)\n"
                ".venv\\Scripts\\Activate.ps1"},
            {"t": "p", "text":
                "After activation your shell prompt shows `(.venv)` and "
                "`python` / `pip` point into the environment. To leave it, "
                "run `deactivate`."},
            {"t": "note", "text":
                "Add `.venv/` to your `.gitignore` so the environment is "
                "never committed. Each developer (and each CI run) recreates "
                "it from `requirements.txt` or `pyproject.toml`."},

            # ---- pyproject.toml ----
            {"t": "h", "text": "Configuring pytest in pyproject.toml"},
            {"t": "p", "text":
                "A single `[tool.pytest.ini_options]` section is all that is "
                "needed to tell pytest where the source code and tests live. "
                "The `pythonpath` key adds `src/` to `sys.path` so that "
                "test files can `import calculator` without any path hacks."},
            {"t": "code", "run": False, "caption": "pyproject.toml – pytest section",
             "code":
                "[build-system]\n"
                "requires = [\"setuptools\", \"wheel\"]\n"
                "build-backend = \"setuptools.build_meta\"\n\n"
                "[tool.pytest.ini_options]\n"
                "pythonpath = [\"src\"]\n"
                "testpaths  = [\"tests\"]"},

            # ---- installing pytest ----
            {"t": "h", "text": "Installing and running pytest"},
            {"t": "p", "text":
                "With the virtual environment active, install pytest with "
                "`pip`. It is good practice to upgrade `pip` first so you "
                "get the latest dependency resolver."},
            {"t": "code", "run": False, "caption": "Install pytest",
             "code":
                "pip install --upgrade pip\n"
                "pip install pytest"},
            {"t": "p", "text":
                "Running `pytest` from the project root discovers and "
                "executes every file that matches `test_*.py` or `*_test.py` "
                "inside the configured `testpaths`."},
            {"t": "code", "run": False, "caption": "Run the test suite",
             "code":
                "pytest\n\n"
                "# Verbose output – shows each test name\n"
                "pytest -v\n\n"
                "# Run only tests whose name contains a keyword\n"
                "pytest -k divide"},

            # ---- example module and test ----
            {"t": "h", "text": "A minimal module and its tests"},
            {"t": "p", "text":
                "The `src/calculator.py` module below defines two functions. "
                "Notice the type annotations: they serve as documentation and "
                "are checked by static type-checkers such as mypy."},
            {"t": "code", "run": False, "caption": "src/calculator.py",
             "code":
                "def add(a: float, b: float) -> float:\n"
                "    return a + b\n\n"
                "def divide(a: float, b: float) -> float:\n"
                "    if b == 0:\n"
                "        raise ValueError(\"Cannot divide by zero.\")\n"
                "    return a / b"},
            {"t": "p", "text":
                "The matching test file uses `pytest.raises` to assert that "
                "the function raises the expected exception — a pattern you "
                "will use everywhere:"},
            {"t": "code", "run": False, "caption": "tests/test_calculator.py",
             "code":
                "import pytest\n"
                "from calculator import add, divide\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
                "    assert add(-1, 1) == 0\n\n"
                "def test_divide():\n"
                "    assert divide(6, 3) == 2\n\n"
                "def test_divide_by_zero():\n"
                "    with pytest.raises(ValueError):\n"
                "        divide(10, 0)"},
            {"t": "note", "text":
                "Keep each test function focused on a single behaviour. "
                "When a test fails, a narrow scope makes the root cause "
                "immediately obvious."},
        ],
    },
]
