import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BOT, decide } from './decide.mjs';

const HEAD = 'a'.repeat(40);
const OLD = 'b'.repeat(40);

const clean = { commit_sha: HEAD, has_unresolved_findings: false, status: 'completed' };
const inactive = { commit_sha: null, has_unresolved_findings: null, status: null };

function summary(state, { user = BOT, updatedAt = '2026-01-01T00:00:00Z' } = {}) {
  const json = typeof state === 'string' ? state : JSON.stringify(state);
  return {
    user: { login: user },
    updated_at: updatedAt,
    body: `<!-- codex-pull-request-review-summary -->\n<!-- codex-review-state:v1 ${json} -->\n`,
  };
}

const run = (comments, ackedAt = null, author = 'someone') =>
  decide({ head: HEAD, author, ackedAt, comments }).state;

test('no Codex comment is pending', () => {
  assert.equal(run([]), 'pending');
});

test('a Dependabot pull request passes without a Codex review', () => {
  assert.equal(run([], null, 'app/dependabot'), 'success');
  assert.equal(run([], null, 'dependabot[bot]'), 'success');
});

test('a clean review of the head passes', () => {
  assert.equal(run([summary({ code_review: clean, security_review: inactive })]), 'success');
});

test('a review of an older head is pending', () => {
  assert.equal(run([summary({ code_review: { ...clean, commit_sha: OLD } })]), 'pending');
});

test('a running review is pending', () => {
  assert.equal(run([summary({ code_review: { ...clean, status: 'in_progress' } })]), 'pending');
});

test('unresolved findings fail', () => {
  const review = { code_review: { ...clean, has_unresolved_findings: true } };
  assert.equal(run([summary(review)]), 'failure');
});

const findings = (sha, updatedAt) =>
  summary({ code_review: { ...clean, commit_sha: sha, has_unresolved_findings: true } }, { updatedAt });

test('an ack added after the review waives its findings', () => {
  assert.equal(run([findings(HEAD, '2026-01-01T00:00:00Z')], '2026-01-01T01:00:00Z'), 'success');
});

test('an ack added before a later review does not carry over', () => {
  // The ack was given for an earlier head; Codex then reviewed this one.
  assert.equal(run([findings(HEAD, '2026-01-02T00:00:00Z')], '2026-01-01T01:00:00Z'), 'failure');
});

test('an ack does not pass a review of an older head', () => {
  assert.equal(run([findings(OLD, '2026-01-01T00:00:00Z')], '2026-01-01T01:00:00Z'), 'pending');
});

test('a missing finding state is pending, not clean', () => {
  const { has_unresolved_findings: _, ...partial } = clean;
  assert.equal(run([summary({ code_review: partial })]), 'pending');
});

test('an active security review with unresolved findings fails', () => {
  const security = { commit_sha: HEAD, has_unresolved_findings: true, status: 'completed' };
  assert.equal(run([summary({ code_review: clean, security_review: security })]), 'failure');
});

test('malformed state JSON is pending', () => {
  assert.equal(run([summary('{"code_review": oops}')]), 'pending');
});

test('state written by anyone but Codex is ignored', () => {
  const forged = summary({ code_review: clean }, { user: 'someone' });
  assert.equal(run([forged]), 'pending');
});

test('the most recently updated state wins', () => {
  const older = summary({ code_review: clean }, { updatedAt: '2026-01-01T00:00:00Z' });
  const newer = summary(
    { code_review: { ...clean, has_unresolved_findings: true } },
    { updatedAt: '2026-01-02T00:00:00Z' },
  );
  assert.equal(run([newer, older]), 'failure');
  assert.equal(run([older, newer]), 'failure');
});
