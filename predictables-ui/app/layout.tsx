import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@components/Navbar";

const navbarWidth = "100px";

export const metadata: Metadata = {
  title: "PredicTables",
  description: "PredicTables Web Client",
};

// bg-slate-950 text-yellow-50

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`
        bg-slate-50 text-black
        flex flex-col w-[100vw] h-[100vh] justify-center items-center `}
      >
        <Navbar />
        <section className={`mt-[120px] w-[80%] h-[90%]`}>{children}</section>
      </body>
    </html>
  );
}
