import type { Metadata } from "next";
import PitchDeck from "./pitch-deck";

export const metadata: Metadata = {
  title: "Mule Account Detection - Pitch | dmj.one",
  description: "0.985 AUC-ROC detecting money mule accounts in Indian banking.",
  openGraph: {
    title: "Mule Account Detection - Pitch",
    description: "0.985 AUC-ROC detecting money mule accounts in Indian banking.",
    url: "https://nfpc.dmj.one/pitch",
    type: "website",
  },
};

export default function PitchPage() {
  return <PitchDeck />;
}
