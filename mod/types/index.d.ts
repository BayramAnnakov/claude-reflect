/** One learning the check found: a rule, where it belongs, and what happened to it. */
export type Learning = {
  id: string
  /** The rule, one imperative line in the language the user said it in. */
  rule: string
  scope: 'global' | 'project'
  /** The session root it was said in. */
  project: string
  /** How many times it was said (the check merges repeats). */
  count: number
  firstAt: number
  lastAt: number
  status: 'pending' | 'saved' | 'skipped'
  confidence: number
  /** Already compared once with the user's instruction files. */
  coveredChecked?: boolean
}

/** A rule saved in this session, with where it applies. */
export type SavedRule = { rule: string; scope: 'global' | 'project'; project: string }

declare module 'claude-code' {
  interface PluginState {
    'reflect-mod': {
      /** Pending learnings for this project and global ones, as the band draws them. */
      pending: Learning[]
      /** Rules saved in this session: the system prompt carries them until CLAUDE.md is read again. */
      savedThisSession: SavedRule[]
    }
  }
}
