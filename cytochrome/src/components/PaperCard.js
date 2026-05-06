import Link from "next/link";
import { motion } from "framer-motion";

const PaperCard = ({ paper }) => {
  const isHeadline = !!paper.headline_paper;

  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2 }}
      className={`relative rounded-2xl border p-6
        ${isHeadline
          ? "border-primaryDark bg-primaryDark/5 dark:bg-primaryDark/10"
          : "border-dark/10 dark:border-light/10 bg-light dark:bg-dark"}`}
    >
      {isHeadline && (
        <span className="absolute -top-3 left-6 rounded-full bg-primaryDark px-3 py-0.5
          text-[10px] font-bold uppercase tracking-widest text-dark">
          headline
        </span>
      )}

      <div className="flex items-baseline gap-2 mb-3">
        <span className="text-[10px] uppercase tracking-widest text-dark/40 dark:text-light/40">
          Paper {paper.id}
        </span>
        <span className="text-[10px] uppercase tracking-widest text-dark/40 dark:text-light/40">
          · {paper.short}
        </span>
      </div>

      <Link href={paper.href}>
        <h3 className={`text-lg font-bold leading-snug mb-2
          ${isHeadline
            ? "text-primaryDark"
            : "text-dark dark:text-light hover:text-primary"}`}>
          {paper.title}
        </h3>
      </Link>

      <p className="text-xs text-dark/60 dark:text-light/55 mb-4 line-clamp-4">
        {paper.role}
      </p>

      <div className="space-y-1 mb-4">
        {paper.headline?.slice(0, 2).map((h, i) => (
          <div key={i} className="flex items-baseline justify-between
            text-[11px] text-dark/70 dark:text-light/65 font-mono">
            <span className="truncate pr-2">{h.label}</span>
            <span className="text-primaryDark font-semibold">{h.computed}</span>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between text-[11px]">
        <span className="text-dark/40 dark:text-light/40">
          {paper.status}
        </span>
        <Link
          href={paper.href}
          className="text-primary dark:text-primaryDark hover:underline uppercase tracking-wider"
        >
          read &rarr;
        </Link>
      </div>
    </motion.div>
  );
};

export default PaperCard;
