import os


def clone_repository(repo_dir, repo_url):
    try:
        os.system(f"git clone {repo_url} {repo_dir}")
    except Exception as e:
        print(f"Error {e}: Unable to clone repository")
