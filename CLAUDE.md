## Approach
- Think before acting. Read existing files before writing code.
- Be concise in output but thorough in reasoning.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read unless the file may have changed.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- Keep solutions simple and direct.
- User instructions always override this file.

## Commit messages
- All commits MUST follow the Conventional Commits specification: https://www.conventionalcommits.org/
- Format: `<type>(<optional scope>): <subject>`
  - `type` is one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
  - `scope` is optional and identifies the affected area (e.g. `rust`, `python`, `docs`, `mixer`, `loss`)
  - `subject` is lowercase, imperative mood, no trailing period, ≤ 72 chars
- Body is optional but when present, separated from the subject by a blank line and wrapped at ~72 chars
- Breaking changes use `!` after type/scope (e.g. `refactor!: drop BSRNN model`) and/or a `BREAKING CHANGE:` footer
- Footers (including `Co-Authored-By:`) are separated from the body by a blank line
- Examples:
  - `feat(python): add SpeakerBeam-SS separator skeleton`
  - `fix(mixer): prevent silence augmentation from zeroing full target`
  - `docs(onnx): specify state tensor metadata schema`
  - `refactor!: split rust and python into sibling crates`

## .context 项目上下文

> 项目使用 `.context/` 管理开发决策上下文。

- 编码规范：`.context/prefs/coding-style.md`
- 工作流规则：`.context/prefs/workflow.md`
- 决策历史：`.context/history/commits.md`

**规则**：修改代码前必读 prefs/，做决策时按 workflow.md 规则记录日志。
