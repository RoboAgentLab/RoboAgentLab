# What this fork changes in pi

RoboAgentLab is [pi](https://github.com/earendil-works/pi) plus robotics extensions. The goal is to stay a thin
layer over upstream so that GitHub's **Sync fork** button merges cleanly: fork code lives in fork-only files, and
the few edits to upstream files are listed here so a sync can re-check them.

## Fork-only content (never conflicts)

| Path | What |
|---|---|
| `packages/embodied/` | `@lv-robotics/pi-embodied`: robots as pi extensions (Apache-2.0, versioned on its own) |
| `services/` | Python robot services and the RPC protocol (`services/test.sh`) |
| `docs/audit/` | Audit notes |
| `.github/workflows/fork-ci.yml`, `.github/actions/setup-node` | The fork's CI (the only workflow meant to run in this repository) |
| `FORK.md`, `RELEASES.md`, `LICENSE-MIT`, `NOTICE`, `.claude-plugin/`, `.agents/` | Fork documentation, licensing notices, agent integrations |

Upstream's workflows under `.github/workflows/` and `.github/APPROVED_CONTRIBUTORS` are kept **verbatim** and
disabled in this repository's Actions settings (Settings → Actions, or
`PUT /repos/RoboAgentLab/RoboAgentLab/actions/workflows/<file>/disable`). Do not edit or delete them: every upstream
change to a deleted or edited workflow is a merge conflict. After a sync adds a new upstream workflow, disable it too.

## Edits to upstream files (re-check after every sync)

Kept because they add functionality the fork needs; each is small and self-contained.

- **Queued-message withdrawal** — `packages/agent/src/agent.ts`: `Agent.removeQueuedMessage(match)` removes the
  first queued steering (else follow-up) message that matches before the loop takes it, and returns `false` when none
  matches. `packages/coding-agent/src/core/agent-session.ts`: `AgentSession.withdrawQueuedMessage(text)`.
  Exposed to extensions as `ctx.withdrawQueuedMessage(text)` (`core/extensions/types.ts`, `core/extensions/runner.ts`,
  `modes/interactive/interactive-mode.ts`). Tests: `packages/agent/test/agent.test.ts`,
  `packages/coding-agent/test/trigger-compact-extension.test.ts`.
- **Session export for extensions** — `ctx.exportSession("jsonl" | "html", outputPath)` exports the current session
  branch as `/export` does and resolves to the written path (same files as above).
- **Retry pattern** — `packages/ai/src/utils/retry.ts`: `expired while waiting for admission` (NInfer drops a
  request queued too long for KV-cache admission; mid-stream the SSE error carries only this text) is retried instead
  of ending the turn. Test: `packages/ai/test/retry.test.ts`.
- **Release tooling** — `scripts/release-packages.mjs` adds `isPiReleasePackage` / `getPiReleasePackages`
  (packages in `@lv-robotics/` are versioned and published on their own, not in pi's lockstep release);
  `scripts/{sync-versions.js,release.mjs,publish.mjs,package-artifacts.mjs,publish-release-announcement.mjs}` use
  them. Test: `scripts/sync-versions.test.mjs`.
- **Machine-local `.local/`** — ignored in `.gitignore`, `biome.json` and the `ignoredDirectories` of
  `scripts/check-pinned-deps.mjs` and `scripts/check-ts-relative-imports.mjs`.
- `AGENTS.md`: one line on `services/test.sh`. `LICENSE`: Apache-2.0 for the fork's additions; pi's MIT license is
  `LICENSE-MIT` (see `NOTICE`). The `license` fields of upstream packages stay `MIT`.

These API additions are candidates for upstream pull requests; once merged upstream they can be dropped here.

## Syncing with upstream

```bash
git remote add upstream https://github.com/earendil-works/pi.git   # once
git fetch upstream main
git checkout -b sync-upstream-$(date +%F) main
git merge upstream/main
```

- Lock files (`package-lock.json`, `packages/coding-agent/install-lock/package-lock.json`): take upstream's, then
  `npm install --package-lock-only --ignore-scripts` and `npm run install-lock:coding-agent`.
- Generated model data is not committed; `npm run build` generates it from the catalog sources (the pi.dev catalog
  pin in `nix/model-catalog.json` can lag upstream's source, so prefer `build` over `build:offline` after a sync).
- Then `npm run check`, the suites in `fork-ci.yml`, and `services/test.sh`.
- Upstream breaking changes that affected the fork so far: pi 1.5.0 moved tool schemas to `@earendil-works/pi-schema`
  (`packages/embodied/src/string-enum.ts` keeps a TypeBox `StringEnum`) and replaced `Harness.taskGraph()` with
  `watchTaskGraph()`.
