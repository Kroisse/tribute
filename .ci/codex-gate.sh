#!/usr/bin/env bash
# Set the `codex-review` commit status of a pull request's head from the
# review state Codex keeps in its summary comment.
#
# Usage: .ci/codex-gate.sh <owner/repo> <pr-number>
# Needs GH_TOKEN with statuses:write. With DRY_RUN=1 it prints the status
# instead of posting it.

set -euo pipefail

repo=$1
pr=$2
bot='chatgpt-codex-connector[bot]'
context='codex-review'
ack_label='codex-ack'

pr_json=$(gh pr view "$pr" -R "$repo" --json headRefOid,labels,url)
head=$(jq -r .headRefOid <<<"$pr_json")
url=$(jq -r .url <<<"$pr_json")
acked=$(jq --arg l "$ack_label" '[.labels[].name] | index($l) != null' <<<"$pr_json")

# Codex edits one summary comment in place; take the most recently updated
# state it wrote.
review=$(gh api --paginate --slurp "repos/$repo/issues/$pr/comments" | jq -c --arg bot "$bot" '
  [ .[][]
    | select(.user.login == $bot)
    | {updated_at, m: (.body | capture("codex-review-state:v1 (?<j>\\{[^\\n]*\\}) -->"; "g"))}
    | {updated_at, state: (.m.j | try fromjson catch {})} ]
  | sort_by(.updated_at) | last | .state // empty')

short=${head:0:7}
if [[ -z $review ]]; then
  state=pending description="No Codex review yet"
else
  sha=$(jq -r '.code_review.commit_sha // ""' <<<"$review")
  status=$(jq -r '.code_review.status // ""' <<<"$review")
  # A review counts as clean only when it says so; a missing field is unknown.
  unresolved=$(jq -r '
    [.code_review.has_unresolved_findings,
     (.security_review | select(.status != null) | .has_unresolved_findings)]
    | if any(. == true) then "true" elif all(. == false) then "false" else "unknown" end
  ' <<<"$review")
  if [[ $sha != "$head" ]]; then
    state=pending description="Codex reviewed ${sha:0:7}, not head $short"
  elif [[ $status != completed ]]; then
    state=pending description="Codex review of $short is ${status:-unknown}"
  elif [[ $unresolved == unknown ]]; then
    state=pending description="Codex review of $short reports no finding state"
  elif [[ $unresolved == true && $acked == true ]]; then
    state=success description="Codex findings on $short acknowledged ($ack_label)"
  elif [[ $unresolved == true ]]; then
    state=failure description="Codex has unresolved findings on $short"
  else
    state=success description="Codex reviewed $short"
  fi
fi

if [[ ${DRY_RUN:-} == 1 ]]; then
  echo "$head $context $state: $description"
  exit 0
fi

gh api -X POST "repos/$repo/statuses/$head" \
  -f state="$state" -f context="$context" \
  -f description="$description" -f target_url="$url" >/dev/null
echo "$context $state: $description"
