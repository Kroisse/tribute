// Decide the `codex-review` commit status of a pull request from the review
// state Codex keeps in its summary comment:
//
//   <!-- codex-review-state:v1 {"code_review":{"commit_sha":..., ...}} -->
//
// Codex edits that comment after each review, so the most recently updated
// state is the current one.
//
// The `codex-ack` label waives the unresolved findings of the review it was
// given for: it counts only when it was last added after that state was
// written. A later review, of a new head or of the same one, needs a new
// acknowledgement, so no event the gate misses can carry a waiver forward.

export const BOT = 'chatgpt-codex-connector[bot]';
export const ACK_LABEL = 'codex-ack';

const STATE = /codex-review-state:v1 (\{[^\n]*\}) -->/g;

export function latestReview(comments) {
  let latest = null;
  for (const comment of comments) {
    if (comment.user?.login !== BOT) continue;
    for (const [, json] of (comment.body ?? '').matchAll(STATE)) {
      let state;
      try {
        state = JSON.parse(json);
      } catch {
        state = {};
      }
      if (!latest || comment.updated_at >= latest.updatedAt) {
        latest = { updatedAt: comment.updated_at, state };
      }
    }
  }
  return latest;
}

// true or false when every active review says so; undefined when a review
// leaves it out, which the gate treats as unknown rather than clean.
function hasUnresolvedFindings(review) {
  const flags = [review.code_review?.has_unresolved_findings];
  if (review.security_review?.status != null) {
    flags.push(review.security_review.has_unresolved_findings);
  }
  if (flags.includes(true)) return true;
  if (flags.every((flag) => flag === false)) return false;
  return undefined;
}

const status = (state, description) => ({ state, description });

// `ackedAt` is when the `codex-ack` label was last added, or null when the
// pull request does not carry it.
export function decide({ head, ackedAt, comments }) {
  const short = head.slice(0, 7);
  const latest = latestReview(comments);
  if (!latest) return status('pending', 'No Codex review yet');
  const review = latest.state;

  const { commit_sha: sha, status: run } = review.code_review ?? {};
  if (sha !== head) {
    const reviewed = sha ? sha.slice(0, 7) : 'no commit';
    return status('pending', `Codex reviewed ${reviewed}, not head ${short}`);
  }
  if (run !== 'completed') {
    return status('pending', `Codex review of ${short} is ${run ?? 'unknown'}`);
  }
  switch (hasUnresolvedFindings(review)) {
    case false:
      return status('success', `Codex reviewed ${short}`);
    case true:
      if (ackedAt && ackedAt >= latest.updatedAt) {
        return status('success', `Codex findings on ${short} acknowledged (${ACK_LABEL})`);
      }
      return status(
        'failure',
        ackedAt
          ? `${ACK_LABEL} predates the latest Codex review of ${short}`
          : `Codex has unresolved findings on ${short}`,
      );
    default:
      return status('pending', `Codex review of ${short} reports no finding state`);
  }
}
