import Link from "next/link";
import React from "react";
import Layout from "./Layout";

const Footer = () => {
  return (
    <footer className="w-full border-t border-solid border-dark/10
      font-medium text-sm dark:text-light/50 dark:border-light/10">
      <Layout className="!py-6 flex items-center justify-between lg:flex-col lg:!py-4">
        <span className="text-dark/40 dark:text-light/30">
          {new Date().getFullYear()} &copy; shakespear
        </span>
        <span className="text-dark/30 dark:text-light/20 text-xs tracking-wider lg:mt-2">
          Observation = Computation = Processing
        </span>
        <Link
          href="https://github.com/fullscreen-triangle/shakespear"
          target="_blank"
          className="text-dark/40 dark:text-light/30 hover:text-primaryDark transition-colors"
        >
          GitHub
        </Link>
      </Layout>
    </footer>
  );
};

export default Footer;
