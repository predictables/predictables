import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
    },
  },
  // daisyui: {
  //   themes: [
  //     {
  //       mytheme: {
  //         primary: "#1d4ed8",
  //         secondary: "#3b82f6",
  //         accent: "#ffffff",
  //         neutral: "#1d4ed8",
  //         "base-100": "#ffffff",
  //         // "base-100": "#1d4ed8",
  //         info: "#f59e0b",
  //         success: "#00ffff",
  //         warning: "#fca5a5",
  //         error: "#991b1b",
  //       },
  //     },
  //   ],
  // },
  plugins: [require("daisyui")],
};
export default config;
