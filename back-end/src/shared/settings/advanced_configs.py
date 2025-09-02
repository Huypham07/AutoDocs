from __future__ import annotations

import google.generativeai as genai
from shared.logging import get_logger
from shared.utils import get_settings

logger = get_logger(__name__)
settings = get_settings()

DEFAULT_EXCLUDED_DIRS = [
    './.venv/', './venv/', './env/', './virtualenv/',
    './node_modules/', './bower_components/', './jspm_packages/',
    './.git/', './.svn/', './.hg/', './.bzr/',
    './__pycache__/', './.pytest_cache/', './.mypy_cache/', './.ruff_cache/', './.coverage/',
    './dist/', './build/', './out/', './target/', './bin/', './obj/',
    './docs/', './_docs/', './site-docs/', './_site/',
    './.idea/', './.vscode/', './.vs/', './.eclipse/', './.settings/',
    './logs/', './log/', './tmp/', './temp/',
]
DEFAULT_EXCLUDED_FILES = [
    'yarn.lock', 'pnpm-lock.yaml', 'npm-shrinkwrap.json', 'poetry.lock',
    'Pipfile.lock', 'requirements.txt.lock', 'Cargo.lock', 'composer.lock',
    '.lock', '.DS_Store', 'Thumbs.db', 'desktop.ini', '*.lnk', '.env',
    '.env.*', '*.env', '*.cfg', '*.ini', '.flaskenv', '.gitignore',
    '.gitattributes', '.gitmodules', '.github', '.gitlab-ci.yml',
    '.prettierrc', '.eslintrc', '.eslintignore', '.stylelintrc',
    '.editorconfig', '.jshintrc', '.pylintrc', '.flake8', 'mypy.ini',
    'pyproject.toml', 'tsconfig.json', 'webpack.config.js', 'babel.config.js',
    'rollup.config.js', 'jest.config.js', 'karma.conf.js', 'vite.config.js',
    'next.config.js', '*.min.js', '*.min.css', '*.bundle.js', '*.bundle.css',
    '*.map', '*.gz', '*.zip', '*.tar', '*.tgz', '*.rar', '*.7z', '*.iso',
    '*.dmg', '*.img', '*.msix', '*.appx', '*.appxbundle', '*.xap', '*.ipa',
    '*.deb', '*.rpm', '*.msi', '*.exe', '*.dll', '*.so', '*.dylib', '*.o',
    '*.obj', '*.jar', '*.war', '*.ear', '*.jsm', '*.class', '*.pyc', '*.pyd',
    '*.pyo', '__pycache__', '*.a', '*.lib', '*.lo', '*.la', '*.slo', '*.dSYM',
    '*.egg', '*.egg-info', '*.dist-info', '*.eggs', 'node_modules',
    'bower_components', 'jspm_packages', 'lib-cov', 'coverage', 'htmlcov',
    '.nyc_output', '.tox', 'dist', 'build', 'bld', 'out', 'bin', 'target',
    'packages/*/dist', 'packages/*/build', '.output',
]

# Configuration models with API Keys
if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)
