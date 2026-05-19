# scripts/llm_pr_review.py
import os
import requests
from openai import OpenAI

repo = os.environ["REPO"]
pr_number = os.environ["PR_NUMBER"]
github_token = os.environ["GITHUB_TOKEN"]

headers = {
    "Authorization": f"Bearer {github_token}",
    "Accept": "application/vnd.github+json",
}

files_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
files = requests.get(files_url, headers=headers, timeout=30).json()

diff_parts = []
for f in files:
    filename = f.get("filename")
    patch = f.get("patch", "")
    if not patch:
        continue
    diff_parts.append(f"### {filename}\n```diff\n{patch[:12000]}\n```")

diff = "\n\n".join(diff_parts)[:60000]

client = OpenAI()

prompt = f"""
Review this pull request diff.

Focus on:
- correctness bugs
- security issues
- API misuse
- edge cases
- maintainability
- tests that should be added

Do not comment on formatting unless it affects correctness.
Be concise and specific. If there are no serious issues, say so.

PR diff:

{diff}
"""

response = client.responses.create(
    model="gpt-5.2",
    instructions="You are a senior software engineer performing a careful GitHub PR review.",
    input=prompt,
)

body = "## ChatGPT PR Review\n\n" + response.output_text

comments_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
requests.post(
    comments_url,
    headers=headers,
    json={"body": body},
    timeout=30,
).raise_for_status()