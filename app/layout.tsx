import "./globals.css";
import { GeistSans } from "geist/font/sans";
import { Toaster } from "sonner";
import { cn } from "@/lib/utils";
import { Navbar } from "@/components/navbar";

export const metadata = {
  title: "DWAO - Analysis Agent",
  description:
    "Intelligent analytics assistant for Adobe Analytics data insights",
  openGraph: {
    images: [
      {
        url: "/og?title=DWAO - Analysis Agent",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    images: [
      {
        url: "/og?title=DWAO - Analysis Agent",
      },
    ],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
      </head>
      <body className={cn(GeistSans.className, "antialiased dark")}>
        <div className="h-screen w-screen flex flex-col overflow-hidden">
          <Toaster position="top-center" richColors />
          <Navbar />
          <div className="flex-1 overflow-hidden">
            {children}
          </div>
        </div>
      </body>
    </html>
  );
}