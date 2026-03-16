import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Mule Account Detection | NFPC - dmj.one",
  description:
    "ML system detecting money mule accounts across 160K accounts and 400M+ transactions. 208 features, 3-model ensemble, 0.968 AUC-ROC. RBIH x IIT Delhi.",
  openGraph: {
    title: "Mule Account Detection | NFPC",
    description:
      "0.968 AUC-ROC detecting money mule accounts in Indian banking. 208 features, 3-model ensemble (LightGBM + XGBoost + CatBoost), 160K accounts.",
    url: "https://nfpc.dmj.one",
    siteName: "dmj.one",
    type: "website",
    locale: "en_IN",
  },
  twitter: {
    card: "summary_large_image",
    title: "Mule Account Detection | NFPC",
    description:
      "0.968 AUC-ROC detecting money mule accounts across 160K accounts. 208 features, 3-model ensemble.",
  },
  robots: { index: true, follow: true },
  alternates: { canonical: "https://nfpc.dmj.one" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <meta name="theme-color" content="#0a0a0a" />
        <link rel="icon" href="/favicon.ico" sizes="any" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
