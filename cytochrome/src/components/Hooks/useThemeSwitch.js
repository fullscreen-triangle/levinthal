import { useEffect } from "react";

export function useThemeSwitch() {
  useEffect(() => {
    document.documentElement.classList.add("dark");
    window.localStorage.setItem("theme", "dark");
  }, []);
  return ["dark", () => {}];
}
