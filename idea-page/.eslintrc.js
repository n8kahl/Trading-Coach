/** @type {import("eslint").Linter.BaseConfig} */
module.exports = {
  root: true,
  extends: ["next", "next/core-web-vitals", "plugin:@typescript-eslint/recommended", "prettier"],
  parser: "@typescript-eslint/parser",
  plugins: ["@typescript-eslint"],
  rules: {
    "@typescript-eslint/explicit-module-boundary-types": "off",
    "@typescript-eslint/no-unused-vars": ["error", { "argsIgnorePattern": "^_", "varsIgnorePattern": "^_" }],
    "react/jsx-sort-props": "off",
    "react-hooks/exhaustive-deps": "off",
    "react/no-unescaped-entities": "off"
  }
};
