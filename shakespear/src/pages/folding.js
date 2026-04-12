import Head from 'next/head'
import Layout from '@/components/Layout'
import TransitionEffect from '@/components/TransitionEffect'
import dynamic from 'next/dynamic'

const FoldingSimulation = dynamic(
  () => import('@/components/proteus/FoldingSimulation'),
  { ssr: false }
)

export default function FoldingPage() {
  return (
    <>
      <Head>
        <title>shakespear | Protein Folding</title>
        <meta name="description"
          content="Real-time protein folding simulation via Kuramoto oscillator dynamics. Watch phase-locking drive secondary structure formation." />
      </Head>
      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="!pt-8">
          <div className="mb-6">
            <h1 className="text-4xl font-bold text-dark dark:text-light tracking-tight md:text-3xl">
              Protein Folding
            </h1>
            <p className="text-sm text-dark/50 dark:text-light/40 mt-1 tracking-wider">
              Kuramoto oscillator synchronization in S-entropy space
            </p>
          </div>
          <FoldingSimulation />
        </Layout>
      </main>
    </>
  )
}
