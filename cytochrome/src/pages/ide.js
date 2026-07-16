import Head from "next/head";
import dynamic from "next/dynamic";

// Client-only: the IDE uses D3, refs, and an in-memory receiver.
const ShakespeareIDE = dynamic(
  () => import("@/components/shakespeare/ShakespeareIDE"),
  { ssr: false, loading: () => <div className="flex h-screen items-center justify-center bg-[#1b1b1b] font-mono text-neutral-500">loading Shakespeare…</div> }
);

export default function IdePage() {
  return (
    <>
      <Head>
        <title>Shakespeare · P450 receiver sandbox</title>
      </Head>
      {/* fixed full-screen overlay escapes the site Navbar/Footer chrome */}
      <div className="fixed inset-0 z-50">
        <ShakespeareIDE />
      </div>
    </>
  );
}
