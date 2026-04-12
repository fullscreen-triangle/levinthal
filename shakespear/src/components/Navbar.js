import Link from "next/link";
import React, { useState } from "react";
import { useRouter } from "next/router";
import {
  GithubIcon,
  MoonIcon,
  SunIcon,
} from "./Icons";
import { motion } from "framer-motion";
import { useThemeSwitch } from "./Hooks/useThemeSwitch";

const NavLink = ({ href, title, className = "" }) => {
  const router = useRouter();
  const isActive = router.asPath === href;

  return (
    <Link href={href}
      className={`${className} relative text-sm tracking-wider uppercase
        ${isActive
          ? "text-primaryDark dark:text-primaryDark"
          : "text-dark/60 dark:text-light/50 hover:text-dark dark:hover:text-light"}
        transition-colors duration-200`}>
      {title}
      {isActive && (
        <span className="absolute left-0 -bottom-1 w-full h-[2px] bg-primaryDark" />
      )}
    </Link>
  );
};

const MobileNavLink = ({ href, title, toggle }) => {
  const router = useRouter();
  const isActive = router.asPath === href;

  return (
    <button
      className={`my-2 text-sm tracking-wider uppercase
        ${isActive ? "text-primaryDark" : "text-light/70 dark:text-dark/70"}`}
      onClick={() => { toggle(); router.push(href); }}>
      {title}
    </button>
  );
};

const Navbar = () => {
  const [mode, setMode] = useThemeSwitch();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="w-full flex items-center justify-between px-32 py-6 font-medium z-10
      dark:text-light lg:px-16 relative md:px-12 sm:px-8">

      {/* Mobile hamburger */}
      <button type="button"
        className="flex-col items-center justify-center hidden lg:flex"
        onClick={() => setIsOpen(!isOpen)}>
        <span className="sr-only">Menu</span>
        <span className={`bg-dark dark:bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ${isOpen ? 'rotate-45 translate-y-1' : '-translate-y-0.5'}`} />
        <span className={`bg-dark dark:bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ${isOpen ? 'opacity-0' : 'opacity-100'} my-0.5`} />
        <span className={`bg-dark dark:bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ${isOpen ? '-rotate-45 -translate-y-1' : 'translate-y-0.5'}`} />
      </button>

      {/* Desktop nav */}
      <div className="w-full flex justify-between items-center lg:hidden">

        {/* Brand */}
        <Link href="/" className="flex items-center gap-2">
          <span className="text-xl font-bold tracking-tight text-dark dark:text-light">
            shakespear
          </span>
          <span className="text-[10px] text-dark/30 dark:text-light/30 tracking-widest uppercase hidden xl:inline">
            observation instrument
          </span>
        </Link>

        {/* Nav links */}
        <nav className="flex items-center gap-6">
          <NavLink href="/" title="Home" />
          <NavLink href="/instrument" title="Instrument" />
          <NavLink href="/folding" title="Folding" />
          <NavLink href="/trajectory" title="Trajectory" />
          <NavLink href="/catalysis" title="Catalysis" />
          <NavLink href="/dynamics" title="Dynamics" />
        </nav>

        {/* Right: GitHub + Theme */}
        <nav className="flex items-center gap-3">
          <motion.a target="_blank" className="w-5"
            href="https://github.com/fullscreen-triangle/levinthal"
            whileHover={{ y: -2 }} whileTap={{ scale: 0.9 }}>
            <GithubIcon />
          </motion.a>
          <button
            onClick={() => setMode(mode === "light" ? "dark" : "light")}
            className={`w-6 h-6 flex items-center justify-center rounded-full p-1
              ${mode === "light" ? "bg-dark text-light" : "bg-light text-dark"}`}
            aria-label="Toggle theme">
            {mode === "light" ? <SunIcon className="fill-dark" /> : <MoonIcon className="fill-dark" />}
          </button>
        </nav>
      </div>

      {/* Mobile nav */}
      {isOpen && (
        <motion.div
          className="min-w-[70vw] sm:min-w-[90vw] flex justify-between items-center flex-col
            fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
            py-24 bg-dark/90 dark:bg-light/75 rounded-lg z-50 backdrop-blur-md"
          initial={{ scale: 0, x: "-50%", y: "-50%", opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}>
          <nav className="flex flex-col items-center">
            <MobileNavLink toggle={() => setIsOpen(false)} href="/" title="Home" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/instrument" title="Instrument" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/about" title="Theory" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/projects" title="Models" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/articles" title="Papers" />
          </nav>
          <div className="flex items-center gap-3 mt-4">
            <motion.a target="_blank" className="w-5 bg-light rounded-full dark:bg-dark"
              href="https://github.com/fullscreen-triangle/shakespear"
              whileHover={{ y: -2 }} whileTap={{ scale: 0.9 }}>
              <GithubIcon />
            </motion.a>
            <button
              onClick={() => setMode(mode === "light" ? "dark" : "light")}
              className={`w-6 h-6 flex items-center justify-center rounded-full p-1
                ${mode === "light" ? "bg-dark text-light" : "bg-light text-dark"}`}>
              {mode === "light" ? <SunIcon className="fill-dark" /> : <MoonIcon className="fill-dark" />}
            </button>
          </div>
        </motion.div>
      )}
    </header>
  );
};

export default Navbar;
