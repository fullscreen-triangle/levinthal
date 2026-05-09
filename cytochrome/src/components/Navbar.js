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

const NavLink = ({ href, title, headline = false, className = "" }) => {
  const router = useRouter();
  const isActive = router.asPath === href;

  return (
    <Link href={href}
      className={`${className} relative text-sm tracking-wider uppercase
        ${isActive
          ? "text-primaryDark dark:text-primaryDark"
          : headline
          ? "text-primary dark:text-primaryDark hover:text-primaryDark"
          : "text-dark/60 dark:text-light/50 hover:text-dark dark:hover:text-light"}
        transition-colors duration-200`}>
      {title}
      {headline && !isActive && (
        <span className="ml-1 text-[8px] tracking-widest align-super">★</span>
      )}
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
      dark:text-light xl:px-16 lg:px-12 relative md:px-8 sm:px-6">

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
            cytochrome
          </span>
          <span className="text-[10px] text-dark/30 dark:text-light/30 tracking-widest uppercase hidden xl:inline">
            P450 monograph
          </span>
        </Link>

        {/* Nav links */}
        <nav className="flex items-center gap-5 xl:gap-3">
          <NavLink href="/" title="Home" />
          <NavLink href="/foundations" title="Foundations" />
          <NavLink href="/manifold" title="Manifold" />
          <NavLink href="/glb-input" title="GLB" />
          <NavLink href="/equilibrium" title="Equilibrium" />
          <NavLink href="/transfer" title="Transfer" headline />
          <NavLink href="/compound-i" title="Compound I" />
          <NavLink href="/ch-activation" title="C–H" />
          <NavLink href="/polymorphisms" title="DDI" />
          <NavLink href="/spectroscopy" title="Spectroscopy" />
          <NavLink href="/database" title="Database" />
          <NavLink href="/apparatus" title="Apparatus" />
          <NavLink href="/gallery" title="Gallery" />
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
            py-12 px-8 bg-dark/90 dark:bg-light/75 rounded-lg z-50 backdrop-blur-md"
          initial={{ scale: 0, x: "-50%", y: "-50%", opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}>
          <nav className="flex flex-col items-center">
            <MobileNavLink toggle={() => setIsOpen(false)} href="/" title="Home" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/foundations" title="Foundations · P1" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/manifold" title="Manifold · P2" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/glb-input" title="GLB · P2.5" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/equilibrium" title="Equilibrium · P3" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/transfer" title="Transfer · P4 ★" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/compound-i" title="Compound I · P5" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/ch-activation" title="C–H Activation · P6" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/polymorphisms" title="DDI · P15" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/spectroscopy" title="Spectroscopy · P13" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/database" title="Database · P14" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/apparatus" title="Apparatus" />
            <MobileNavLink toggle={() => setIsOpen(false)} href="/gallery" title="Gallery" />
          </nav>
          <div className="flex items-center gap-3 mt-4">
            <motion.a target="_blank" className="w-5 bg-light rounded-full dark:bg-dark"
              href="https://github.com/fullscreen-triangle/levinthal"
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
