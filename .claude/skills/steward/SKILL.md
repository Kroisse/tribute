---
name: steward
description: |
  How a session reacts to GitHub activity on a Tribute pull request it opened
  or drives (`subscribe_pr_activity` wake events and check-ins). Use when:
  (1) A `<wake reason="external-event">` GitHub event arrives for a PR
  (2) Deciding whether a review, CI result, or bot comment needs action
---

# PR Steward

These rules adjust the generic PR subscription guidance for this repository.
Each wake replays the whole session context, so a wake that needs nothing
should end at once.

## Events That Need No Action

End the turn without tool calls, apart from at most one line to the user,
for:

- `issue_comment.edited` and `pull_request_review_comment.edited`. Bots
  rewrite their summary and inline comments after every push; the edit
  carries no new request.
- New comments from `codecov` or `codspeed`. They are reports. A coverage
  or performance problem that matters also turns a check red, and that
  check's own event covers it.
- A green CI result or check-suite rollup while no other item is open.

Do not inspect the PR on these events to look for other work.

## Do Not Schedule Check-ins

Do not schedule `send_later` safety-net check-ins. GitHub events reach the
session on their own; the merge or close does too.

## CodeRabbit Reviews

Treat each CodeRabbit finding as a bug report: verify it against the code,
fix the valid ones, and leave the rest. Do not reply to or resolve
CodeRabbit threads. CodeRabbit re-reviews after the next push and resolves
what was addressed itself. Report to the user which findings were fixed and
which were left, and why.

## Human Reviews

Present a human reviewer's requests to the user with a proposed change
before pushing, unless the user already asked the session to address them.

## CI Failures

A red check on the PR is work, as the generic guidance says. Investigate it
with the commands in the `tribute-testing` skill before pushing a fix.

## Commits and Pushes

- Add new commits for fixes; do not amend or force-push.
- Batch related fixes into one push.
- In plan mode, the session cannot edit files: summarize the needed change
  and wait for the user instead of pushing.

## When the PR Is Merged or Closed

Report it to the user in one line and stop acting on the PR.
