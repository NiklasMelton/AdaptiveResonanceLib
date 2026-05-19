# scripts/ChatGPT_failure_review.py
import io
import os
import zipfile
import requests
from openai import OpenAI

repo = os.environ["REPO"]
run_id = os.environ["RUN_ID"]
workflow_name = os.environ["WORKFLOW_NAME"]
token = os.environ["GITHUB_TOKEN"]

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
}

run = requests.get(
    f"https://api.github.com/repos/{repo}/actions/runs/{run_id}",
    headers=headers,
    timeout=30,
).json()

prs = run.get("pull_requests", [])
if not prs:
    print("No PR associated with run.")
    raise SystemExit(0)

pr_number = prs[0]["number"]

labels = requests.get(
    f"https://api.github.com/repos/{repo}/issues/{pr_number}/labels",
    headers=headers,
    timeout=30,
).json()

if "chatgpt-review" not in [label["name"] for label in labels]:
    print("PR does not have chatgpt-review label.")
    raise SystemExit(0)

artifacts = requests.get(
    f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts",
    headers=headers,
    timeout=30,
).json()["artifacts"]

logs = []

for artifact in artifacts:
    url = artifact["archive_download_url"]
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            if name.endswith((".log", ".xml", ".txt")):
                content = z.read(name).decode("utf-8", errors="replace")
                logs.append(f"## {artifact['name']} / {name}\n\n{content}")

failure_text = "\n\n".join(logs)[-100000:]

client = OpenAI()

response = client.responses.create(
    model="gpt-5.2",
    instructions=(
        "You are reviewing CI failure logs for a Python open-source repository. "
        "Explain the likely root cause and suggest concrete fixes. "
        "Do not invent files or APIs not present in the logs."
    ),
    input=f"""
Workflow: {workflow_name}
PR: #{pr_number}

Failure logs:

{failure_text}
""",
)

comment = f"""## ChatGPT CI Failure Review

Workflow failed: `{workflow_name}`

{response.output_text}
"""

requests.post(
    f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
    headers=headers,
    json={"body": comment},
    timeout=30,
).raise_for_status()