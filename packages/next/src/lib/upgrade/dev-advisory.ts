import type { UpgradeContext } from './nudge'
import type { UpgradeAdvisory } from '../../next-devtools/shared/upgrade-advisory'
import { assessUpgrade } from './nudge'
import { isCI } from '../../server/ci-info'

export function createUpgradeAdvisory(
  directory: string,
  config: UpgradeContext,
  installedVersion: string
) {
  const createSnapshot = (policy: string): UpgradeAdvisory => {
    return {
      installedVersion,
      prompt: `Run \`next upgrade --ai=${policy}\` from this app's directory (${directory}) and follow its instructions.`,
    }
  }
  // Fall back to a simulated insight for preview and tests when no security
  // advisory was found. Keep the actual assessment for upgrade nudges.
  const simulateSecurityInsight =
    process.env.__NEXT_AGENT_UPGRADE_FORCE_DEVTOOLS_FOR_TESTING === '1'
  let snapshot: UpgradeAdvisory | null = null
  const policy = config.experimental.agenticAutoUpgrade
  const forced = process.env.__NEXT_AGENTIC_AUTO_UPGRADE === policy
  const assessment = (
    forced && isCI
      ? Promise.resolve(null)
      : assessUpgrade(directory, config, installedVersion, null, forced)
  ).then((advisory) => {
    if (advisory?.kind === 'security') {
      snapshot = createSnapshot(advisory.policy)
    } else if (simulateSecurityInsight) {
      snapshot = createSnapshot('security')
    }
    return advisory
  })

  return { assessment, getSnapshot: () => snapshot }
}
