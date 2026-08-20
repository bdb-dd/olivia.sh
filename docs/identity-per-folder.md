# Per-folder git + `gh` identity

Recipe for making commit authorship, push credentials and `gh` all agree, chosen
by which folder you are in. Written after a session where commits were authored
as one GitHub account while being pushed to a repo owned by another.

**The core asymmetry:** git has native per-folder configuration; `gh` does not.
`gh` keeps ONE active account in `~/.config/gh/hosts.yml`. `gh auth switch` is
global mutable state and is not reliable — observed switching to an account,
successfully creating a PR, and reverting on its own by the next command.
`GH_TOKEN` overrides account selection entirely, so that is the lever to use.

Assumes [mise](https://mise.jdx.dev) is installed and shell-activated. `direnv`
works identically; use its `.envrc` in place of `mise.toml`.

---

## 1. One SSH alias per identity

`~/.ssh/config`:

```
Host github-ACCOUNT
    HostName github.com
    User git
    IdentityFile ~/.ssh/KEY_FOR_ACCOUNT
    IdentitiesOnly yes
```

`IdentitiesOnly yes` matters — without it ssh offers every key it has and you
authenticate as whoever answers first.

## 2. An identity file per account

`~/.gitconfig-ACCOUNT`:

```ini
[user]
    name  = Your Name
    email = ID+ACCOUNT@users.noreply.github.com

# Force any plain github.com remote through this identity's alias, so a remote
# added the ordinary way cannot silently authenticate as the wrong key.
[url "git@github-ACCOUNT:"]
    insteadOf = git@github.com:
    insteadOf = https://github.com/
```

Find the no-reply address at GitHub → Settings → Emails, or read it out of an
existing repo: `git log --format='%ce' | sort -u`.

## 3. Select it automatically

Append to `~/.gitconfig`:

```ini
[includeIf "gitdir:~/dev/FOLDER/"]
    path = ~/.gitconfig-ACCOUNT
[includeIf "hasconfig:remote.*.url:git@github-ACCOUNT:**"]
    path = ~/.gitconfig-ACCOUNT
```

Both, because each covers a gap in the other:

- `gitdir:` is folder-based and works **before a remote exists** (fresh `git init`).
  Trailing slash required. For a bare+worktree layout, `GIT_DIR` is
  `<repo>/.bare/worktrees/<name>`, so point the pattern at the tree root, not the
  checkout directory.
- `hasconfig:` (git ≥ 2.36) keys off **where the repo actually pushes**, so the
  identity follows the repo if it is cloned somewhere unexpected. This is the one
  that catches an authored-as-A / pushed-to-B mismatch.

Later includes win, so list `hasconfig` last.

## 4. `gh`, via the folder's environment

Put this **outside the repo** (or in a gitignored `mise.local.toml`) — it is a
personal mapping, and a collaborator's `gh` will not have your accounts:

```toml
# ~/dev/FOLDER/mise.toml
[env]
GH_TOKEN = "{{ exec(command='gh auth token --user ACCOUNT') }}"
```

Then `mise trust ~/dev/FOLDER`.

The token is read from gh's own keyring at runtime, so **no secret is written to
disk**. Requires `gh auth login --user ACCOUNT` to have been done once.

## 5. Verify

```ini
# ~/.gitconfig  [alias]
whoami = "!f() { \
    printf 'author   : %s <%s>\\n' \"$(git config user.name)\" \"$(git config user.email)\"; \
    printf 'remote   : %s\\n' \"$(git config remote.origin.url)\"; \
    src=$([ -n \"$GH_TOKEN\" ] && echo 'GH_TOKEN env' || echo 'keyring active account'); \
    printf 'gh       : %s (via %s)\\n' \"$(gh api user --jq .login 2>/dev/null || echo '<gh unavailable>')\" \"$src\"; \
}; f"
```

Use `gh api user` rather than parsing `gh auth status` — the latter's account
line ends in the storage backend, so naive parsing reports `keyring` as the
username.

## Gotchas

- **Never echo `$GH_TOKEN`.** `${GH_TOKEN:-no}` prints the *value* when set. A
  token was leaked into a terminal transcript exactly this way; it had to be
  revoked via GitHub → Settings → Applications → Authorized OAuth Apps.
- An existing shell will not pick up a new `mise.toml` until the directory hook
  re-runs — `cd ..` and back.
- Quote nothing in `[user] name`. A config containing `name = \"Your Name\"`
  yields a literal-quoted author string.
- `gh` reads `GH_TOKEN` **before** the keyring, so this wins over whatever
  `gh auth switch` last did. That is the point.
