import { useState } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";

const PanelGrid = ({ panels, panelDir, columns = 2 }) => {
  const [open, setOpen] = useState(null);

  return (
    <>
      <div className={`grid gap-4 ${columns === 2 ? "grid-cols-2 md:grid-cols-1" : "grid-cols-3 lg:grid-cols-2 sm:grid-cols-1"}`}>
        {panels.map(([file, caption], idx) => (
          <motion.figure
            key={file}
            whileHover={{ y: -2 }}
            className="cursor-pointer rounded-xl overflow-hidden border
              border-dark/10 dark:border-light/10 bg-white dark:bg-light/5"
            onClick={() => setOpen({ src: `${panelDir}/${file}`, caption, idx })}
          >
            <div className="relative aspect-[4/1] bg-white">
              <Image
                src={`${panelDir}/${file}`}
                alt={caption}
                fill
                sizes="(max-width: 768px) 100vw, 50vw"
                className="object-contain"
              />
            </div>
            <figcaption className="px-3 py-2 text-[11px] text-dark/65 dark:text-light/55">
              <span className="font-mono text-primaryDark mr-2">
                {String(idx + 1).padStart(2, "0")}
              </span>
              {caption}
            </figcaption>
          </motion.figure>
        ))}
      </div>

      {/* Lightbox */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center
              bg-dark/80 backdrop-blur-sm p-8"
            onClick={() => setOpen(null)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="relative max-w-6xl w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="relative aspect-[4/1] bg-white rounded-lg overflow-hidden">
                <Image
                  src={open.src}
                  alt={open.caption}
                  fill
                  sizes="100vw"
                  className="object-contain"
                />
              </div>
              <p className="mt-3 text-light text-sm">
                <span className="font-mono text-primaryDark mr-2">
                  Panel {String(open.idx + 1).padStart(2, "0")}
                </span>
                {open.caption}
              </p>
              <button
                className="absolute -top-2 -right-2 w-8 h-8 rounded-full
                  bg-light text-dark text-lg font-bold"
                onClick={() => setOpen(null)}
                aria-label="Close"
              >
                ×
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default PanelGrid;
