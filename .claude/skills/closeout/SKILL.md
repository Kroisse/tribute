---
name: closeout
description: Close out a merged PR and its issue
---

# Close Out a Merged PR

Reply in Korean.

1. `gh pr view $ARGUMENTS --json state,mergeCommit,closingIssuesReferences`
   and confirm the PR is merged into main.
2. For each linked issue, read the body and check off every completed
   checklist item (`gh issue edit`).
3. List known limitations or deferred items. Draft follow-up issues and show
   them to me before filing.
4. Draft a closing comment summarizing the PR(s). Post it and close the issue
   after I approve.
5. Suggest the next task, starting with continuations of this issue or
   milestone.
