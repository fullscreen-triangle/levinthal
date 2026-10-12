import { Html, Head, Main, NextScript } from "next/document";
import Script from "next/script";

// The site is dark only. The `dark` class is rendered on the server so every
// `dark:` variant applies from the first paint, whatever the visitor's OS
// preference or an older stored "light" setting says.
export default function Document() {
  return (
    <Html lang="en" className="dark" style={{ colorScheme: "dark" }}>
      <Head>
        <meta name="theme-color" content="#1b1b1b" />
        <meta name="color-scheme" content="dark" />
      </Head>
      <body className="bg-dark text-light">
        <Script id="theme-dark" strategy="beforeInteractive">
          {`document.documentElement.classList.add('dark');
try { localStorage.setItem('theme', 'dark'); } catch (e) {}`}
        </Script>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
