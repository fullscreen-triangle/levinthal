import Link from "next/link";
import React from "react";

const Footer = () => {
  return (
    <footer
      className="w-full border-t border-solid border-dark/10 dark:border-light/10
        font-medium text-sm dark:text-light sm:text-xs"
    >
      <div className="px-32 py-8 flex items-center justify-between
        xl:px-16 lg:px-12 lg:flex-col lg:gap-3 lg:py-6 md:px-8 sm:px-6">
        <span className="text-dark/60 dark:text-light/50">
          {new Date().getFullYear()} &copy; Sachikonye, K.~F. ·
          Categorical Mechanics of Cytochrome P450
        </span>

        <div className="flex items-center gap-3 text-dark/60 dark:text-light/50">
          <Link
            href="https://github.com/fullscreen-triangle/levinthal"
            target="_blank"
            className="underline underline-offset-2 hover:text-dark dark:hover:text-light"
          >
            source
          </Link>
          <span>·</span>
          <Link
            href="mailto:kundai.sachikonye@bitspark.com"
            className="underline underline-offset-2 hover:text-dark dark:hover:text-light"
          >
            contact
          </Link>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
