"""Trigger a Read the Docs build; never push, merge, or expose credentials."""
import argparse
import os
import re

import requests


def deploy(project, version, token):
    if not token:
        raise ValueError("Set READTHEDOCS_TOKEN in your local environment, then retry. "
                         "Create/connect the project at https://app.readthedocs.org/ first.")
    if not all(re.fullmatch(r"[a-zA-Z0-9_.-]+", slug) for slug in (project, version)):
        raise ValueError("Project/version must be Read the Docs slugs")
    url = f"https://app.readthedocs.org/api/v3/projects/{project}/versions/{version}/builds/"
    response = requests.post(url, headers={"Authorization": f"Token {token}"}, timeout=30)
    if response.status_code != 202:
        raise ValueError(f"Read the Docs returned HTTP {response.status_code}. Check project/version and token access.")
    return f"https://app.readthedocs.org/projects/{project}/builds/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="polyclustx")
    parser.add_argument("--version", default="latest")
    args = parser.parse_args()
    try:
        url = deploy(args.project, args.version, os.environ.get("READTHEDOCS_TOKEN"))
    except (ValueError, requests.RequestException) as exc:
        # Request headers (including the token) are never printed.
        parser.exit(1, f"Deployment not triggered: {exc}\n")
    print(f"Build requested. Check completion at {url}")
    print(f"Documentation after a successful build: https://{args.project}.readthedocs.io/en/{args.version}/")


if __name__ == "__main__":
    main()
