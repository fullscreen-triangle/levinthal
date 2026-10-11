import Head from "next/head";
import dynamic from "next/dynamic";

// The deck draws with d3 timers and the Fullscreen API, so it renders on the
// client only. Keys: arrows / space, F full screen, N notes, O overview.
const Deck = dynamic(() => import("@/components/talk/Deck"), { ssr: false });

export default function Talk() {
  return (
    <>
      <Head>
        <title>Cytochrome P450 from counted commitment · talk</title>
        <meta name="description" content="A derivation of cytochrome P450 and its applications, as a full-screen presentation." />
      </Head>
      <Deck />
    </>
  );
}
