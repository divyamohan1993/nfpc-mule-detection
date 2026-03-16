import type { Metadata } from "next";
import PitchDeck from "./pitch-deck";

export const metadata: Metadata = {
  title: "Mule Account Detection - Pitch | dmj.one",
  description: "0.968 AUC-ROC detecting money mule accounts across 160K accounts. 208 features, 3-model ensemble.",
  openGraph: {
    title: "Mule Account Detection - Pitch",
    description: "0.968 AUC-ROC detecting money mule accounts across 160K accounts. 208 features, 3-model ensemble.",
    url: "https://nfpc.dmj.one/pitch",
    type: "website",
  },
};

export default function PitchPage() {
  return <PitchDeck />;
}
