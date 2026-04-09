import dynamic from 'next/dynamic';
import Head from 'next/head';

const ProteusInstrument = dynamic(
  () => import('../src/components/proteus/ProteusInstrument'),
  { ssr: false }
);

export default function ProteusPage() {
  return (
    <>
      <Head>
        <title>PROTEUS | Protein Observation via Universal Shaders</title>
        <meta name="description" content="Real-time protein analysis through GPU fragment shader observation. No backend. The shader IS the instrument." />
      </Head>
      <ProteusInstrument />
    </>
  );
}
