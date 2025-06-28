import pytest
from unittest.mock import patch, MagicMock
from app.db.repo_db_manager import DBManager
import subprocess

def test_extract_repo_name():
    db_manager = DBManager()
    
    # Test with a standard GitHub URL
    test_cases = [
        ("https://github.com/user/repo.git", "user_repo"),
        ("https://github.com/user/repo", "user_repo"),
        ("git@github.com:user/repo.git", "repo"),
    ]
    for url, expected in test_cases:
        assert db_manager._extract_repo_name(url) == expected
    
@patch("app.core.db_manager.subprocess.run")
def test_download_repo_success(mock_run):
    dm = DBManager()
    mock_run.return_value = MagicMock()
    repo_url = "https://github.com/user/repo.git"
    local_path = "/tmp/repo"
    token = "token123"
    result = dm._download_repo(repo_url, local_path, token)
    assert result == local_path

@patch("app.core.db_manager.subprocess.run")
def test_download_repo_fail(mock_run):
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd="git",
        stderr=b"fatal: could not read Username for 'https://token123@github.com': No such device"
    )
    dm = DBManager()
    with pytest.raises(ValueError) as e:
        dm._download_repo("https://github.com/user/repo.git", "/tmp/repo", "token123")
    assert "***TOKEN***" in str(e.value)