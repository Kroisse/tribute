// Set the `codex-review` commit status of a pull request's head.
//
// Usage: node .ci/codex-gate/main.mjs <owner/repo> <pr-number>
// Needs `gh` authenticated with statuses:write. With DRY_RUN=1 it prints the
// status instead of posting it.

import { execFileSync } from 'node:child_process';
import { ACK_LABEL, decide } from './decide.mjs';

const CONTEXT = 'codex-review';

const [repo, pr] = process.argv.slice(2);
if (!repo || !pr) {
  console.error('usage: main.mjs <owner/repo> <pr-number>');
  process.exit(2);
}

const gh = (...args) =>
  execFileSync('gh', args, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });

const info = JSON.parse(gh('pr', 'view', pr, '-R', repo, '--json', 'headRefOid,labels,url'));
const list = (path) => JSON.parse(gh('api', '--paginate', '--slurp', path)).flat();
const comments = list(`repos/${repo}/issues/${pr}/comments`);

let ackedAt = null;
if (info.labels.some((label) => label.name === ACK_LABEL)) {
  for (const event of list(`repos/${repo}/issues/${pr}/events`)) {
    if (event.event === 'labeled' && event.label?.name === ACK_LABEL) {
      if (!ackedAt || event.created_at > ackedAt) ackedAt = event.created_at;
    }
  }
}

const head = info.headRefOid;
const { state, description } = decide({ head, ackedAt, comments });

if (process.env.DRY_RUN === '1') {
  console.log(`${head} ${CONTEXT} ${state}: ${description}`);
  process.exit(0);
}

gh(
  'api', '-X', 'POST', `repos/${repo}/statuses/${head}`,
  '-f', `state=${state}`,
  '-f', `context=${CONTEXT}`,
  '-f', `description=${description}`,
  '-f', `target_url=${info.url}`,
);
console.log(`${CONTEXT} ${state}: ${description}`);
