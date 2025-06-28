def extract_full_repo_name(repo_url: str) -> str:
        """
        Extracts the repository name from the given URL.
        """
        url_parts = repo_url.rstrip('/').split('/')

        # GitHub URL format: https://github.com/owner/repo
        if  len(url_parts) >= 5:
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        return repo_name
    
def extract_repo_info(repo_url: str) -> dict:
    """
    Extracts the owner and repository name from the given URL.
    """
    url_parts = repo_url.rstrip('/').split('/')
    
    if len(url_parts) >= 5:
        owner = url_parts[-2]
        repo_name = url_parts[-1].replace(".git", "")
    else:
        owner = url_parts[-1].replace(".git", "")
        repo_name = owner
    
    return {
        "owner": owner,
        "repo_name": repo_name
    }