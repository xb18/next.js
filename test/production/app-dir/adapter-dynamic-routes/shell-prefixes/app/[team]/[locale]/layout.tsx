import { locale, team } from 'next/root-params'

// This layout takes two root params. `generateStaticParams` returns three
// combinations of them, and not the full product of four. The build therefore
// produces no output for the `sparse` team with the `de` locale.
//
// One team value contains `.`, `-` and `,`. A regex treats those characters as
// special.
export function generateStaticParams() {
  return [
    { team: 'acme.one-two,three', locale: 'en' },
    { team: 'acme.one-two,three', locale: 'de' },
    { team: 'sparse', locale: 'en' },
  ]
}

export default async function Root({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang={await locale()} data-team={await team()}>
      <body>{children}</body>
    </html>
  )
}
