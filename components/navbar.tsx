"use client";

import Link from "next/link";

export const Navbar = () => {
  return (
    <div className="flex-shrink-0 pt-6 pl-8 flex items-center justify-start">
      <div
        className="text-3xl uppercase"
        style={{ fontFamily: "Aptos, sans-serif" }}
      >
        DWAO
      </div>
      {/* Right side is intentionally left empty or for future items */}
    </div>
  );
};